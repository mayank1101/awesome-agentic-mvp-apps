"""Reading the resume once, into the object the rest of the run works from.

One model call per run, and everything after it -- the queries, the ranking, the
per-job judgement -- depends on this being right. Which is why the profile is
shown to the user *before* the search runs: a resume read as "data analyst" when
the candidate is a data engineer produces thirty plausible, wrong results, and
the mistake is invisible from the results themselves.

The post-processing below is not cosmetic. A model asked for "at most 15 skills"
sometimes returns forty; asked for a seniority label it occasionally returns
"Senior Engineer". Both parse fine and both are wrong in ways that only show up
much later -- as a 400-character search query, or as a level the query builder
silently ignores.
"""

import re

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import CandidateProfile
from app.prompts import PROFILE_SYSTEM, profile_user_message
from app.services.guardrails import fence
from app.services.llm import complete_model

logger = get_logger(__name__)

#: Caps applied after parsing, matching what the prompt asks for. The prompt is a
#: request; this is the guarantee.
_LIMITS = {"titles": 4, "skills": 15, "domains": 5, "locations": 3, "highlights": 4}

#: Longest a single title or skill may be. Anything past this is a sentence the
#: model put in a list field, and a sentence in a search query matches nothing.
_MAX_TERM_CHARS = 60

#: Seniority words, longest first, so "senior" is not found inside a string that
#: says "senior lead".
_SENIORITY_WORDS: tuple[tuple[str, str], ...] = (
    ("junior", "junior"),
    ("entry", "junior"),
    ("intern", "junior"),
    ("principal", "lead"),
    ("staff", "lead"),
    ("lead", "lead"),
    ("head", "lead"),
    ("director", "lead"),
    ("senior", "senior"),
    ("sr", "senior"),
    ("mid", "mid"),
)


def extract_profile(resume_text: str) -> CandidateProfile:
    """Parse a resume into the structured profile the run is built on.

    Args:
        resume_text: Normalised text from the uploaded PDF.

    Returns:
        The profile, cleaned and capped.

    Raises:
        ModelError: The call failed, or the reply could not be validated after a
            repair attempt. Fatal for the run -- without a profile there is
            nothing to search for and nothing to score against.
    """
    settings = get_settings()

    profile = complete_model(
        system=PROFILE_SYSTEM,
        user=profile_user_message(fence(resume_text)),
        schema=CandidateProfile,
        max_tokens=settings.max_tokens_profile,
    )

    cleaned = _clean(profile)
    logger.info(
        "Resume parsed: %d titles, %d skills, seniority=%s",
        len(cleaned.titles),
        len(cleaned.skills),
        cleaned.seniority,
    )
    return cleaned


def _clean(profile: CandidateProfile) -> CandidateProfile:
    """Apply the caps and normalisations the prompt asked for but cannot enforce."""
    data = profile.model_dump()

    for field, limit in _LIMITS.items():
        values = [_tidy(value) for value in data.get(field, [])]
        if field != "highlights":
            values = [value for value in values if len(value) <= _MAX_TERM_CHARS]
        data[field] = _dedupe(values)[:limit]

    years = data.get("years_experience")
    if years is not None and not 0 <= years <= 60:
        # A model that reads a graduation year as a duration returns numbers like
        # 2018. Dropping it is better than carrying a number that would put the
        # candidate at lead level on arithmetic alone.
        logger.warning("Discarding implausible years_experience=%s", years)
        data["years_experience"] = None

    data["summary"] = _tidy(data.get("summary", ""))
    return CandidateProfile.model_validate(data)


def _tidy(value: str) -> str:
    """Collapse whitespace and strip list punctuation from one value."""
    return re.sub(r"\s+", " ", str(value or "")).strip(" -*•\t")


def _dedupe(values: list[str]) -> list[str]:
    """De-duplicate case-insensitively, preserving first-seen order."""
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            kept.append(value)
    return kept


def infer_seniority(text: str) -> str | None:
    """Read a seniority level out of a job title, if it states one.

    Used to sanity-check a search result against the level the user asked for.
    Returns ``None`` for a title that says nothing about level, which is the
    common case and must not be treated as a mismatch.
    """
    lowered = f" {text.lower()} "
    for word, level in _SENIORITY_WORDS:
        if re.search(rf"[^a-z]{word}[^a-z]", lowered):
            return level
    return None
