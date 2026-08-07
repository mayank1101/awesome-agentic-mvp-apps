"""The deep tier: one model call that reads a posting and judges it.

One call per job, not two. The sibling app extracts requirements in one call and
judges them in a second, which is cleaner and costs twice as much -- and the
second call re-reads text the first one already had in front of it. Here the
expensive tier runs once per job, several times per run, so the pair is collapsed
into a single request whose reply carries both halves.

What comes back is normalised hard before anything downstream sees it. Models
number things inconsistently: ids like ``1``, ``R1``, ``Req-3``, assessments for
requirements that were never emitted, requirements with no assessment at all,
and occasionally forty requirements when the prompt asked for twenty. Every one
of those parses as valid JSON and every one of them corrupts a score quietly, so
:func:`_normalise` re-indexes the ids itself and guarantees the invariant the
scorer relies on: **exactly one assessment per requirement, in order.**
"""

from app.core.config import get_settings
from app.core.exceptions import ModelRequestTooLarge
from app.core.logging import get_logger
from app.models.schemas import JobAssessment, JobHit, JobRequirement, RequirementAssessment
from app.prompts import ASSESSMENT_SYSTEM, assessment_user_message
from app.services.guardrails import fence
from app.services.llm import complete_model

logger = get_logger(__name__)

#: How much of the posting survives the retry after a too-large request. Groq's
#: free tier counts the output reservation toward a per-minute token ceiling, so
#: a long posting plus a two-page resume can exceed it on the first attempt. The
#: requirements live near the top of a posting, so the front half is the half
#: worth keeping.
_SHRINK_FACTOR = 0.5


def assess_job(
    hit: JobHit,
    posting_text: str,
    resume_text: str,
) -> JobAssessment:
    """Read one posting and judge it against the resume.

    Args:
        hit: The search result, used for the fallback title and the source URL.
        posting_text: The posting's text. Full page text where the fetch worked;
            the search snippet where it did not, in which case the caller is
            responsible for keeping the row's tier honest.
        resume_text: The candidate's resume text.

    Returns:
        The assessment, normalised so that requirement ids are dense and every
        requirement has exactly one verdict.

    Raises:
        ModelError: The call failed, or the reply could not be validated after a
            repair attempt. Fatal for this job only -- the run continues and the
            row carries the error.
    """
    settings = get_settings()

    try:
        assessment = _call(hit, posting_text, resume_text, settings.max_tokens_assessment)
    except ModelRequestTooLarge as exc:
        shortened = posting_text[: int(len(posting_text) * _SHRINK_FACTOR)]
        logger.warning(
            "Assessment request too large (%s); retrying with %d characters of posting",
            exc,
            len(shortened),
        )
        assessment = _call(hit, shortened, resume_text, settings.max_tokens_assessment)

    return _normalise(assessment)


def _call(hit: JobHit, posting_text: str, resume_text: str, max_tokens: int) -> JobAssessment:
    """Issue one assessment call with both documents fenced."""
    return complete_model(
        system=ASSESSMENT_SYSTEM,
        user=assessment_user_message(
            posting_text=fence(posting_text),
            resume_text=fence(resume_text),
            fallback_title=hit.title,
            source_url=hit.url,
        ),
        schema=JobAssessment,
        max_tokens=max_tokens,
    )


def _normalise(assessment: JobAssessment) -> JobAssessment:
    """Re-index requirements and pair each with exactly one verdict.

    Must-haves are ordered first and the cap is applied after that ordering, so
    a posting with thirty bullets loses its "nice to have" tail rather than the
    requirements that decide whether the candidate is screened out.
    """
    cap = get_settings().max_requirements

    requirements = [
        requirement for requirement in assessment.requirements if requirement.text.strip()
    ]
    verdicts = {
        _id_key(verdict.requirement_id): verdict
        for verdict in assessment.assessments
        if verdict.requirement_id
    }

    ordered = sorted(requirements, key=lambda requirement: not requirement.must_have)[:cap]

    renumbered: list[JobRequirement] = []
    paired: list[RequirementAssessment] = []

    for index, requirement in enumerate(ordered, start=1):
        new_id = f"R-{index:02d}"
        original = _id_key(requirement.id)
        verdict = verdicts.get(original)

        renumbered.append(requirement.model_copy(update={"id": new_id}))
        paired.append(
            verdict.model_copy(update={"requirement_id": new_id})
            if verdict is not None
            else RequirementAssessment(
                requirement_id=new_id,
                status="missing",
                note="The model returned no verdict for this requirement.",
            )
        )

    if len(ordered) < len(requirements):
        logger.info("Capped requirements at %d (posting stated %d)", cap, len(requirements))

    return assessment.model_copy(update={"requirements": renumbered, "assessments": paired})


def _id_key(raw: str) -> str:
    """Reduce an id to its digits, so ``R-01``, ``R1``, and ``1`` agree.

    Ids that carry no digits fall back to the lowercased string, which still
    pairs a consistent-but-unconventional scheme correctly.
    """
    digits = "".join(character for character in raw if character.isdigit())
    return digits.lstrip("0") or raw.strip().lower()
