"""Rewriting the resume for one posting, then proving it invented nothing.

The rewrite is one model call. The interesting part is what happens after it:
the output goes through :mod:`app.services.validation`, which compares every
number, name, and contact detail against the original resume text. A rewrite that
introduces one gets **one** repair attempt -- the model is shown exactly which
fragments were not in the original and asked to remove them -- and if the second
attempt also fails, strict mode refuses to produce a resume at all.

Refusing is the right default. The alternative is handing a candidate a document
that reads well, that they did not write, and that they will have to defend in an
interview. A rewrite that cannot be produced honestly is a rewrite that should
not be produced.

The model is shown the original resume *text*, not the parsed profile. The parse
is a lossy intermediate; the text is what the candidate actually wrote, and it is
also what the guard compares against, so rewriting from anything else would mean
checking the output against a document the model never saw.
"""

from collections.abc import Callable

from app.core.config import get_settings
from app.core.exceptions import FabricationDetected
from app.core.logging import get_logger
from app.models.schemas import FitReport, TailoredResume, TailoredResumeDraft
from app.prompts import TAILOR, system
from app.services import provenance
from app.services.analyzer import AnalysisResult
from app.services.guardrails import fence, sanitize_markdown
from app.services.llm import complete_model
from app.services.validation import check_tailored_resume

logger = get_logger(__name__)

ProgressCallback = Callable[[str], None]


def tailor_resume(
    analysis: AnalysisResult,
    *,
    progress: ProgressCallback | None = None,
) -> TailoredResume:
    """Produce a tailored resume that states no fact the original did not.

    Args:
        analysis: The finished analysis, which carries the original resume text,
            the posting, and the gap list that tells the rewrite what to
            emphasise.
        progress: Called with a short status line before each step.

    Returns:
        The tailored resume, sanitised, with the change log.

    Raises:
        FabricationDetected: The rewrite introduced unsupported facts twice and
            ``STRICT_FABRICATION_GUARD`` is on.
        ModelError: Any provider failure, already classified.
    """
    settings = get_settings()

    def step(message: str) -> None:
        if progress:
            progress(message)

    step("Rewriting your resume for this role")
    draft = _request_rewrite(analysis)
    outcome = check_tailored_resume(draft.markdown, analysis.resume_text)

    if not outcome.passed and outcome.violations:
        logger.warning(
            "Rewrite failed the fabrication guard (%s engine); requesting a repair",
            outcome.engine,
        )
        step("Checking every claim against your resume")
        draft = _request_rewrite(analysis, offenders=provenance.describe(outcome.violations))
        outcome = check_tailored_resume(draft.markdown, analysis.resume_text)

    if outcome.violations and settings.strict_fabrication_guard:
        raise FabricationDetected(
            "The rewrite kept introducing details that are not in your resume, so it "
            "was not produced. Your original resume is unchanged.",
            offenders=provenance.describe(outcome.violations),
        )

    return TailoredResume(
        markdown=sanitize_markdown(draft.markdown),
        changes=draft.changes,
        flagged=provenance.describe(outcome.violations),
    )


def _request_rewrite(
    analysis: AnalysisResult,
    *,
    offenders: list[str] | None = None,
) -> TailoredResumeDraft:
    """Make one rewrite call.

    Args:
        analysis: The finished analysis.
        offenders: Fragments a previous attempt invented. When present, the
            prompt names them and demands their removal -- naming them beats
            repeating "do not invent things", which the first attempt already
            ignored.

    Returns:
        The draft, unvalidated.
    """
    posting = analysis.posting
    report = analysis.report

    covered = [a for a in report.assessments if a.status in ("covered", "partial")]
    emphasise = [
        f"- {requirement.text}"
        for requirement in posting.requirements
        if any(a.requirement_id == requirement.id for a in covered)
    ]
    absent = [
        f"- {requirement.text}"
        for requirement in posting.requirements
        if any(
            a.requirement_id == requirement.id and a.status == "missing" for a in report.assessments
        )
    ]

    instructions = (
        f"Target role: {posting.title or 'unspecified'}"
        + (f" at {posting.company}" if posting.company else "")
        + "\n\nWhat this posting asks for that the resume DOES support -- lead with these, "
        "in the posting's vocabulary:\n"
        + ("\n".join(emphasise) or "- (nothing matched; keep the resume as it is)")
        + "\n\nWhat this posting asks for that the resume DOES NOT support -- these must NOT "
        "appear anywhere in your output:\n"
        + ("\n".join(absent) or "- (none)")
        + (
            "\n\nPosting keywords, usable ONLY where the original resume already describes "
            "that work: " + ", ".join(posting.keywords[:15])
            if posting.keywords
            else ""
        )
        + _actions_block(report)
    )

    if offenders:
        instructions += (
            "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED. These fragments do not appear in the "
            "original resume and you wrote them anyway:\n"
            + "\n".join(f"- {offender}" for offender in offenders)
            + "\nRewrite again without them. Do not substitute different invented details "
            "for these; remove the claim entirely and use only what the original says."
        )

    user = (
        f"{instructions}\n\nOriginal resume, verbatim -- this is the only source of facts you "
        f"may use:\n{fence(analysis.resume_text)}"
    )

    return complete_model(
        system=system(TAILOR),
        user=user,
        schema=TailoredResumeDraft,
        max_tokens=get_settings().max_tokens_tailor,
    )


def _actions_block(report: FitReport) -> str:
    """Render the analysis's action plan as instructions for the rewrite.

    Gap actions are deliberately excluded. They describe things the candidate
    does not have, and handing "the resume does not show Kubernetes" to a model
    whose job is to make the resume fit this posting is an invitation to write
    Kubernetes into it -- which the fabrication guard would then reject, costing
    a call and a repair round for no reason.
    """
    actionable = [action for action in report.actions if not action.is_gap]
    if not actionable:
        return ""

    lines = "\n".join(
        f"- [{action.category}] {action.section + ': ' if action.section else ''}{action.change}"
        for action in actionable
    )
    return (
        "\n\nThe analysis produced these specific edits. Apply the ones that are possible "
        "without inventing anything:\n" + lines
    )
