"""Turning per-requirement verdicts into a number the app is willing to defend.

The model is never asked for a score. It is asked, per requirement, whether the
resume covers it and which line shows that. The number is arithmetic over those
answers, here, in code. That ordering is the whole design:

* the same resume and posting produce the same score twice, because the only
  non-deterministic step is a per-requirement verdict with three possible values;
* the score can be explained line by line, which is what a candidate looking at
  thirty links actually needs -- "68" ranks jobs, "68, and here are the two
  must-haves you are missing" decides which one to spend an hour on;
* a model in a generous mood can inflate individual verdicts and not a total.

**Two checks the model does not control run before the arithmetic.**

*Provenance*: a ``covered`` verdict quoting a line that is not in the resume is
demoted. This is the failure mode the sibling app measured and this one inherits
-- an evidence quote that reads perfectly and appears nowhere in the document.

*Similarity backstop*: a ``covered`` verdict for a requirement that matches no
resume line at all, in either similarity mode, is demoted. Calibrated as a
**gross-mismatch** check rather than a judge, for a measured reason recorded in
the sibling app: with hosted embeddings, absolute cosine cannot separate "the
resume covers this" from "the resume is also professional English" -- genuine
coverage measured 0.706-0.899 and genuine misses 0.618-0.755, overlapping. Any
threshold aggressive enough to catch the misses also demotes real matches, and
demoting a real match costs a candidate a job they could have got.

Must-haves and nice-to-haves are weighted differently because they mean different
things: a missing must-have is a screen-out, a missing nice-to-have is a
conversation. A posting that states no preferred requirements does not lose the
points allocated to them -- the weights renormalise.
"""

import re

from app.core.logging import get_logger
from app.models.schemas import (
    JobAssessment,
    MatchingMode,
    RequirementAssessment,
    ScoreBreakdown,
)
from app.services.matching import RequirementMatch, match_requirements

logger = get_logger(__name__)

#: Credit per coverage status.
_STATUS_CREDIT: dict[str, float] = {"covered": 1.0, "partial": 0.5, "missing": 0.0}

#: Share of the total that required items carry. Preferred items are worth
#: something -- they are what separates two candidates who both clear the bar --
#: but not much, and a posting with none of them is scored on must-haves alone.
_MUST_HAVE_WEIGHT = 0.8

#: Below this best-similarity *and* below the margin floor, a claimed coverage is
#: treated as unsupported. Both conditions, not either: a requirement can be
#: genuinely covered by a line that reads nothing like it ("shipped services in
#: Go" against "Golang"), and the margin is what distinguishes that from a
#: resume that simply has nothing to say about the requirement.
_SEMANTIC_SIMILARITY_FLOOR = 0.70
_SEMANTIC_MARGIN_FLOOR = 0.06

#: Lexical mode has a real zero -- vocabulary overlap with an unrelated
#: requirement genuinely is 0.0 -- so a plain floor works and no margin is
#: needed.
_LEXICAL_SIMILARITY_FLOOR = 0.15

#: Resume lines shorter than this are headings, dates, and section labels. They
#: are evidence of nothing and they drag a baseline down.
_MIN_EVIDENCE_LINE = 25

#: Whitespace and punctuation are squashed on both sides before an evidence quote
#: is looked for in the resume. PDF extraction inserts kerning spaces inside
#: words and glues icon glyphs onto values, so a quote that is character-for-
#: character correct still fails a naive containment check.
_SQUASH = re.compile(r"[^a-z0-9]+")


def resume_lines(resume_text: str) -> list[str]:
    """Split a resume into the lines a requirement can be matched against."""
    seen: set[str] = set()
    lines: list[str] = []
    for raw in resume_text.splitlines():
        line = raw.strip(" -*•\t")
        if len(line) < _MIN_EVIDENCE_LINE or line.lower() in seen:
            continue
        seen.add(line.lower())
        lines.append(line)
    return lines


def squash(text: str) -> str:
    """Reduce text to lowercase alphanumerics for tolerant containment checks."""
    return _SQUASH.sub("", text.lower())


def score_assessment(
    assessment: JobAssessment,
    resume_text: str,
) -> tuple[float, ScoreBreakdown, JobAssessment, MatchingMode]:
    """Score one assessed job, after checking the verdicts that claim coverage.

    Args:
        assessment: The normalised assessment -- one verdict per requirement.
        resume_text: The resume the verdicts claim to be about.

    Returns:
        The 0-100 score, the arithmetic behind it, the assessment with any
        demoted verdicts rewritten (so the screen shows what was actually
        counted, not what was claimed), and the mode the similarity numbers came
        from.
    """
    if not assessment.requirements:
        return 0.0, ScoreBreakdown(), assessment, "lexical"

    evidence_lines = resume_lines(resume_text)
    matches, mode = match_requirements(assessment.requirements, evidence_lines)
    by_id = {match.requirement_id: match for match in matches}

    checked: list[RequirementAssessment] = []
    demoted: list[str] = []

    for verdict in assessment.assessments:
        adjusted = _check(verdict, by_id.get(verdict.requirement_id), resume_text, mode)
        if adjusted.status != verdict.status:
            demoted.append(verdict.requirement_id)
        checked.append(adjusted)

    breakdown = _tally(assessment, checked, demoted)
    score = _combine(breakdown)

    if demoted:
        logger.info("Demoted %d unsupported verdict(s): %s", len(demoted), ", ".join(demoted))

    return score, breakdown, assessment.model_copy(update={"assessments": checked}), mode


def _check(
    verdict: RequirementAssessment,
    match: RequirementMatch | None,
    resume_text: str,
    mode: MatchingMode,
) -> RequirementAssessment:
    """Demote a claimed coverage that the resume does not support.

    Only ``covered`` is checked. ``partial`` is already a hedge, and demoting it
    to ``missing`` on a similarity number would be the scorer overruling the
    reader on exactly the judgement calls the reader is better at.

    The two checks are **ordered, not independent**: a quote that is verifiably
    in the resume settles the question, and the similarity backstop is skipped.
    Running both would demote correct verdicts wherever similarity is blind to
    morphology -- lexical mode scores "payments domain" against "two payment
    integrations" at zero, because the words differ by one letter.
    """
    if verdict.status != "covered":
        return verdict

    if verdict.evidence:
        if _evidence_in_resume(verdict.evidence, resume_text):
            return verdict
        return verdict.model_copy(
            update={
                "status": "partial",
                "note": (
                    f"{verdict.note} (Counted as partial: the quoted evidence does not appear "
                    "in the resume.)"
                ).strip(),
            }
        )

    if match is not None and _below_floor(match, mode):
        return verdict.model_copy(
            update={
                "status": "partial",
                "note": (
                    f"{verdict.note} (Counted as partial: no resume line is close to this "
                    "requirement.)"
                ).strip(),
            }
        )

    return verdict


def _evidence_in_resume(evidence: str, resume_text: str) -> bool:
    """Whether a quoted line really appears in the resume.

    Squashed on both sides -- see :data:`_SQUASH`. Very short quotes are accepted
    without checking: below a handful of characters, containment says nothing
    and the check would demote correct verdicts whose evidence was a single
    token like "Go".
    """
    needle = squash(evidence)
    if len(needle) < 12:
        return True
    return needle in squash(resume_text)


def _below_floor(match: RequirementMatch, mode: MatchingMode) -> bool:
    """Whether a requirement is a gross mismatch for every line in the resume."""
    if mode == "semantic":
        return (
            match.similarity < _SEMANTIC_SIMILARITY_FLOOR and match.margin < _SEMANTIC_MARGIN_FLOOR
        )
    return match.similarity < _LEXICAL_SIMILARITY_FLOOR


def _tally(
    assessment: JobAssessment,
    verdicts: list[RequirementAssessment],
    demoted: list[str],
) -> ScoreBreakdown:
    """Count the checked verdicts into the two weighted groups."""
    must_have_ids = {
        requirement.id for requirement in assessment.requirements if requirement.must_have
    }

    breakdown = ScoreBreakdown(demoted=demoted)
    must_credit = 0.0
    nice_credit = 0.0

    for verdict in verdicts:
        credit = _STATUS_CREDIT.get(verdict.status, 0.0)
        if verdict.requirement_id in must_have_ids:
            breakdown.must_have_total += 1
            must_credit += credit
            breakdown.must_have_covered += verdict.status == "covered"
            breakdown.must_have_partial += verdict.status == "partial"
        else:
            breakdown.nice_to_have_total += 1
            nice_credit += credit
            breakdown.nice_to_have_covered += verdict.status == "covered"
            breakdown.nice_to_have_partial += verdict.status == "partial"

    breakdown.must_have_score = (
        round(100.0 * must_credit / breakdown.must_have_total, 1)
        if breakdown.must_have_total
        else 0.0
    )
    breakdown.nice_to_have_score = (
        round(100.0 * nice_credit / breakdown.nice_to_have_total, 1)
        if breakdown.nice_to_have_total
        else None
    )
    return breakdown


def _combine(breakdown: ScoreBreakdown) -> float:
    """Weight the two groups into one 0-100 score, renormalising when needed."""
    if not breakdown.must_have_total and breakdown.nice_to_have_score is None:
        return 0.0
    if not breakdown.must_have_total:
        # A posting that states everything as preferred is unusual but real, and
        # scoring it zero would be an artefact of the weighting rather than a
        # fact about the candidate.
        return round(breakdown.nice_to_have_score or 0.0, 1)
    if breakdown.nice_to_have_score is None:
        return round(breakdown.must_have_score, 1)

    return round(
        _MUST_HAVE_WEIGHT * breakdown.must_have_score
        + (1.0 - _MUST_HAVE_WEIGHT) * breakdown.nice_to_have_score,
        1,
    )


def explain(assessment: JobAssessment, breakdown: ScoreBreakdown) -> str:
    """Write the one-line reason shown next to a deep score.

    Names the missing must-haves rather than counting them: "missing Kubernetes,
    Terraform" is a sentence a candidate can act on, "covers 6 of 8" is a
    sentence they have to expand the row to understand.
    """
    if not assessment.requirements:
        return "The posting stated no requirements that could be scored."

    requirement_text = {requirement.id: requirement.text for requirement in assessment.requirements}
    must_have_ids = {
        requirement.id for requirement in assessment.requirements if requirement.must_have
    }

    missing = [
        requirement_text.get(verdict.requirement_id, verdict.requirement_id)
        for verdict in assessment.assessments
        if verdict.requirement_id in must_have_ids and verdict.status == "missing"
    ]

    covered = breakdown.must_have_covered
    total = breakdown.must_have_total

    if not missing:
        if breakdown.must_have_partial:
            return (
                f"Covers {covered} of {total} must-haves outright and "
                f"{breakdown.must_have_partial} partially."
            )
        return f"Covers all {total} must-haves."

    named = ", ".join(_shorten(text) for text in missing[:3])
    if len(missing) > 3:
        named += f", and {len(missing) - 3} more"
    return f"Covers {covered} of {total} must-haves. Missing: {named}."


def _shorten(text: str, limit: int = 48) -> str:
    """Trim a requirement to something that fits on one line."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 1].rstrip()}…"
