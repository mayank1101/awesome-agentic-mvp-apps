"""The analysis pipeline: two documents in, one scored report out.

Six steps, three of them model calls:

1. Scan both documents for injection attempts.
2. Parse the resume into structure (model call).
3. Parse the posting into requirements (model call).
4. Match each requirement to the resume lines that could support it (embeddings
   or lexical overlap -- no model).
5. Judge each requirement against its evidence, and draft advice (model call).
6. Reconcile the verdicts against measured similarity and compute the score
   (arithmetic -- no model).

Synchronous, and called from Streamlit's own script thread. The `progress`
callback therefore paints safely; that is only true because nothing here runs on
a worker thread. If a future change moves this pipeline onto a background thread,
the callback has to become a queue of events the UI drains -- painting Streamlit
from another thread raises `NoSessionContext`.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.exceptions import InputBlocked, ModelRequestTooLarge, RunDeadlineExceeded
from app.core.logging import get_logger
from app.models.schemas import (
    AssessmentBatch,
    FitReport,
    JobPosting,
    JobRequirement,
    ResumeProfile,
)
from app.prompts import ASSESSOR, JD_PARSER, RESUME_PARSER, system
from app.services import scoring
from app.services.guardrails import Finding, fence, has_severity, scan_input
from app.services.llm import complete_model
from app.services.matching import RequirementMatch, match_requirements

logger = get_logger(__name__)

ProgressCallback = Callable[[str], None]


@dataclass
class AnalysisResult:
    """Everything one analysis produced.

    Attributes:
        report: The scored report, for display.
        profile: The parsed resume, reused by the rewrite so it is not parsed
            twice.
        posting: The parsed job description, same reason.
        resume_text: The extracted resume text, kept because the fabrication
            guard compares against the *original text*, not the parse of it.
        findings: Guardrail findings that were warned about rather than blocked.
        elapsed_seconds: Wall-clock time for the run, shown in the UI footer.
    """

    report: FitReport
    profile: ResumeProfile
    posting: JobPosting
    resume_text: str
    findings: list[Finding] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def analyze(
    resume_text: str,
    job_description: str,
    *,
    truncated_resume: bool = False,
    truncated_jd: bool = False,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    """Run the full analysis.

    Args:
        resume_text: Extracted resume text.
        job_description: Normalised job description.
        truncated_resume: Whether the resume text was capped, for the report.
        truncated_jd: Whether the posting was capped, for the report.
        progress: Called with a short status line before each step.

    Returns:
        The analysis result.

    Raises:
        InputBlocked: A high-severity injection pattern was found and blocking is
            enabled.
        RunDeadlineExceeded: The run passed its global deadline between steps.
        ModelError: Any provider failure, already classified.
    """
    settings = get_settings()
    started = time.monotonic()

    def step(message: str) -> None:
        # Monotonic, not wall clock: a system clock adjustment mid-run must not
        # turn a 30-second analysis into an expired one.
        if time.monotonic() - started > settings.run_deadline_seconds:
            raise RunDeadlineExceeded(
                f"The analysis passed its {settings.run_deadline_seconds:.0f}s deadline."
            )
        if progress:
            progress(message)

    findings: list[Finding] = []
    if settings.guardrails_enabled:
        step("Checking the documents")
        findings = scan_input(resume_text, job_description)
        if settings.block_flagged_input and has_severity(findings, "high"):
            raise InputBlocked(
                "One of the documents contains text that tries to instruct the "
                "assistant rather than describe experience.",
                findings=findings,
            )

    step("Reading the resume")
    profile = complete_model(
        system=system(RESUME_PARSER),
        user=f"Resume text:\n{fence(resume_text)}",
        schema=ResumeProfile,
        max_tokens=settings.max_tokens_extraction,
    )

    step("Reading the job description")
    posting = complete_model(
        system=system(JD_PARSER % {"max_requirements": settings.max_requirements}),
        user=f"Job posting:\n{fence(job_description)}",
        schema=JobPosting,
        max_tokens=settings.max_tokens_extraction,
    )
    posting.requirements = _normalize_requirements(posting.requirements, settings.max_requirements)

    step("Matching requirements to your experience")
    evidence = profile.evidence_texts()
    matches, mode = match_requirements(posting.requirements, evidence)

    step("Judging each requirement")
    batch = _assess(profile, posting, matches)

    step("Scoring")
    assessments = scoring.reconcile(batch.assessments, matches, mode)
    dimensions = scoring.build_dimensions(posting, assessments, resume_text)
    score = scoring.overall_score(dimensions)

    keyword_advice = scoring.keyword_actions(posting.keywords, resume_text, evidence)
    actions = scoring.merge_actions(
        sorted(batch.actions, key=lambda action: action.priority),
        scoring.fallback_actions(posting, assessments, matches, keyword_advice),
    )

    report = FitReport(
        overall_score=score,
        band=scoring.band_for(score),
        dimensions=dimensions,
        assessments=assessments,
        strengths=batch.strengths[:5],
        gaps=batch.gaps[:6],
        actions=actions,
        keyword_actions=keyword_advice,
        matching_mode=mode,
        truncated_resume=truncated_resume,
        truncated_jd=truncated_jd,
    )

    elapsed = time.monotonic() - started
    logger.info(
        "Analysis complete: score=%d requirements=%d mode=%s elapsed=%.1fs",
        score,
        len(posting.requirements),
        mode,
        elapsed,
    )

    return AnalysisResult(
        report=report,
        profile=profile,
        posting=posting,
        resume_text=resume_text,
        findings=findings,
        elapsed_seconds=elapsed,
    )


def _normalize_requirements(
    requirements: list[JobRequirement],
    cap: int,
) -> list[JobRequirement]:
    """Give every requirement a unique sequential id and apply the cap.

    The prompt asks for `R-01`-style ids, and models mostly comply -- but "mostly"
    is not a property to build the assessment join on, and a duplicate id would
    silently make two requirements share one verdict.
    """
    normalized: list[JobRequirement] = []
    for index, requirement in enumerate(requirements[:cap], start=1):
        normalized.append(requirement.model_copy(update={"id": f"R-{index:02d}"}))
    return normalized


def _assess(
    profile: ResumeProfile,
    posting: JobPosting,
    matches: list[RequirementMatch],
) -> AssessmentBatch:
    """Ask the model for a verdict on each requirement.

    The prompt carries one block per requirement, each with only the evidence
    matched to it, plus a short profile header for context the line-level
    evidence cannot give (seniority, current role). Sending the whole resume
    again would double the prompt and invite the model to go find support
    elsewhere -- which is exactly the behaviour the similarity floor exists to
    catch.

    **This call adapts to the provider's size limits instead of failing at
    them.** Groq's free tier caps tokens per *minute* and counts the requested
    output reservation toward that cap, so on the smallest model a two-page
    resume against a dozen requirements is rejected outright: 6444 requested
    against a 6000 ceiling. Three escalating attempts:

    1. full evidence, full output reservation;
    2. one evidence line per requirement, trimmed, and a smaller reservation;
    3. the requirements split into batches small enough to fit, assessed
       separately and merged.

    The last one costs more calls but is the only thing that works when the
    requirement list itself is what does not fit.
    """
    if not posting.requirements:
        return AssessmentBatch()

    settings = get_settings()
    full_tokens = settings.max_tokens_assessment

    try:
        return _assess_once(profile, posting, posting.requirements, matches, full_tokens)
    except ModelRequestTooLarge:
        logger.warning("Assessment prompt too large; retrying with trimmed evidence")

    reduced_tokens = max(700, int(full_tokens * 0.6))
    try:
        return _assess_once(
            profile,
            posting,
            posting.requirements,
            matches,
            reduced_tokens,
            evidence_per_requirement=1,
            evidence_char_cap=160,
        )
    except ModelRequestTooLarge:
        logger.warning("Still too large; splitting the requirements into batches")

    return _assess_in_batches(profile, posting, matches, reduced_tokens)


def _assess_in_batches(
    profile: ResumeProfile,
    posting: JobPosting,
    matches: list[RequirementMatch],
    max_tokens: int,
) -> AssessmentBatch:
    """Assess the requirements in chunks and merge the replies.

    The advice fields are concatenated and deduplicated rather than taken from
    one batch: each batch only saw part of the posting, so each has a partial
    view of what the candidate should do about it.
    """
    size = max(2, len(posting.requirements) // 2)
    merged = AssessmentBatch()

    while True:
        chunks = [
            posting.requirements[start : start + size]
            for start in range(0, len(posting.requirements), size)
        ]
        try:
            for chunk in chunks:
                batch = _assess_once(
                    profile,
                    posting,
                    chunk,
                    matches,
                    max_tokens,
                    evidence_per_requirement=1,
                    evidence_char_cap=160,
                )
                merged.assessments.extend(batch.assessments)
                merged.strengths.extend(batch.strengths)
                merged.gaps.extend(batch.gaps)
                merged.actions.extend(batch.actions)
        except ModelRequestTooLarge:
            if size <= 2:
                raise
            size = max(2, size // 2)
            logger.warning("Batch still too large; halving to %d requirements", size)
            merged = AssessmentBatch()
            continue
        break

    merged.strengths = _dedupe(merged.strengths)[:5]
    merged.gaps = _dedupe(merged.gaps)[:6]
    return merged


def _dedupe(items: list[str]) -> list[str]:
    """Drop repeats while keeping order, case-insensitively."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = " ".join(item.lower().split())
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _assess_once(
    profile: ResumeProfile,
    posting: JobPosting,
    requirements: list[JobRequirement],
    matches: list[RequirementMatch],
    max_tokens: int,
    *,
    evidence_per_requirement: int | None = None,
    evidence_char_cap: int | None = None,
) -> AssessmentBatch:
    """Make one assessment call over `requirements`.

    Args:
        profile: The parsed resume, for the one-line header.
        posting: The posting, for the role line.
        requirements: The subset to assess in this call.
        matches: Every match, indexed by requirement id.
        max_tokens: Output reservation for this call.
        evidence_per_requirement: Cap on evidence lines shown per requirement.
        evidence_char_cap: Cap on the length of each evidence line.

    Returns:
        The reply, validated.

    Raises:
        ModelRequestTooLarge: The provider refused the request on size.
    """
    by_id = {match.requirement_id: match for match in matches}
    blocks: list[str] = []

    for requirement in requirements:
        match = by_id.get(requirement.id)
        evidence = list(match.evidence) if match else []
        if evidence_per_requirement is not None:
            evidence = evidence[:evidence_per_requirement]
        if evidence_char_cap is not None:
            evidence = [text[:evidence_char_cap] for text in evidence]
        lines = "\n".join(f"  - {text}" for text in evidence) or "  (no similar line found)"
        blocks.append(
            f"{requirement.id} [{'must-have' if requirement.must_have else 'preferred'}] "
            f"{requirement.text}\n{lines}"
        )

    header_parts = [part for part in (profile.headline, profile.summary) if part]
    header_cap = 300 if evidence_char_cap else 600
    header = " | ".join(header_parts)[:header_cap] or "(no summary on the resume)"

    user = (
        f"Role: {posting.title or 'unspecified'}"
        + (f" at {posting.company}" if posting.company else "")
        + (f" ({posting.seniority})" if posting.seniority else "")
        + "\n\nCandidate profile line:\n"
        + fence(header)
        + "\n\nRequirements, each followed by the closest lines from the resume:\n"
        + fence("\n\n".join(blocks))
    )

    return complete_model(
        system=system(ASSESSOR),
        user=user,
        schema=AssessmentBatch,
        max_tokens=max_tokens,
    )
