"""The cheap tier: ordering every search result against the resume.

This is what decides which jobs get read properly, so its job is *ordering*, not
measurement. It sees a title and a few hundred characters of search snippet --
never enough to say whether a candidate meets a requirement, easily enough to say
that a Kubernetes platform role fits a backend resume better than a data
analytics one does.

Two things here are worth understanding before changing the numbers.

**Hosted embeddings do not start at zero.** Two unrelated pieces of professional
English sit around 0.6 cosine before either says anything, so a raw cosine
rendered as a percentage would put an irrelevant job at 60/100. The mapping below
subtracts that floor and stretches what is left, and it deliberately tops out
below 100: no amount of snippet similarity justifies a number that reads like a
verified match.

**A shallow score is a different quantity from a deep one and is never blended
with it.** The band this produces (roughly 15-75) overlaps a deep score's range
on purpose -- they are both "how well does this fit" -- but the tier travels with
the number everywhere it goes, and the UI states which one it is on every row.
"""

from dataclasses import dataclass

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger
from app.models.schemas import CandidateProfile, JobHit, MatchingMode, SearchFilters
from app.services.embeddings import embed_texts
from app.services.matching import cosine, tokenize
from app.services.profile import infer_seniority

logger = get_logger(__name__)

#: Cosine below which a hit is treated as unrelated. Measured, not chosen by
#: feel: hosted embeddings put any two professional documents in the 0.55-0.65
#: band regardless of subject.
_COSINE_FLOOR = 0.58

#: Cosine at which the mapping saturates. Above this the differences are noise.
_COSINE_CEILING = 0.88

#: The band a shallow score is rendered into. The floor is not zero because a
#: result that survived the posting filter is a real job on a whitelisted site,
#: and the ceiling is not 100 because nothing here read the posting.
_SCORE_FLOOR = 15.0
_SCORE_CEILING = 75.0

#: Penalty applied when a title states a level the user did not ask for. Small on
#: purpose: it should reorder near-ties, not bury a good match whose title
#: happens to say "staff".
_SENIORITY_PENALTY = 8.0

#: Adjacent levels, which should not be penalised against each other -- a senior
#: candidate applying to a lead role is a normal application, not a mismatch.
_ADJACENT: dict[str, set[str]] = {
    "junior": {"mid"},
    "mid": {"junior", "senior"},
    "senior": {"mid", "lead"},
    "lead": {"senior"},
}


@dataclass(frozen=True)
class Ranking:
    """One search result with its cheap-tier numbers.

    Attributes:
        hit: The result.
        similarity: Raw similarity, 0-1, in whichever mode produced it.
            Comparable within a run and within a mode, not across modes.
        score: The similarity mapped into the shallow band, 0-100.
    """

    hit: JobHit
    similarity: float
    score: float


def rank_hits(
    profile: CandidateProfile,
    hits: list[JobHit],
    filters: SearchFilters | None = None,
) -> tuple[list[Ranking], MatchingMode]:
    """Order search results by how well they fit the resume.

    Args:
        profile: The parsed resume.
        hits: Every result that survived filtering.
        filters: The user's filters, used only for the seniority nudge.

    Returns:
        Rankings sorted best first, and the mode that produced the similarities.
        The mode is ``lexical`` whenever embeddings were unavailable *or* failed:
        an embedding outage degrades a run, it does not end one.
    """
    if not hits:
        return [], "lexical"

    profile_text = profile.profile_text()
    if not profile_text.strip():
        # No profile text means nothing to compare against. Order by the search
        # provider's own relevance rather than returning an arbitrary order.
        ordered = sorted(hits, key=lambda hit: hit.provider_score, reverse=True)
        return [Ranking(hit=hit, similarity=0.0, score=_SCORE_FLOOR) for hit in ordered], "lexical"

    try:
        similarities = _semantic_similarities(profile_text, hits)
        mode: MatchingMode = "semantic"
    except EmbeddingError as exc:
        logger.warning("Falling back to lexical ranking: %s", exc)
        similarities = _lexical_similarities(profile_text, hits)
        mode = "lexical"

    wanted_level = (filters.seniority if filters else None) or profile.seniority

    rankings = [
        Ranking(
            hit=hit,
            similarity=round(similarity, 4),
            score=_to_score(similarity, mode, hit, wanted_level),
        )
        for hit, similarity in zip(hits, similarities, strict=True)
    ]

    # Provider score breaks ties, so an ordering is never arbitrary: two jobs the
    # embeddings cannot separate fall back to what the search engine thought.
    rankings.sort(key=lambda ranking: (ranking.score, ranking.hit.provider_score), reverse=True)
    return rankings, mode


def _semantic_similarities(profile_text: str, hits: list[JobHit]) -> list[float]:
    """Cosine between the profile and each hit, in one batched embedding call."""
    vectors = embed_texts([profile_text] + [hit.ranking_text() for hit in hits])
    profile_vector, hit_vectors = vectors[0], vectors[1:]
    return [cosine(profile_vector, vector) for vector in hit_vectors]


def _lexical_similarities(profile_text: str, hits: list[JobHit]) -> list[float]:
    """Vocabulary overlap between the profile and each hit.

    Overlap relative to the *hit*, not the profile: a resume has a far larger
    vocabulary than a job snippet, so measuring the profile's coverage would
    score every job near zero and rank on noise.
    """
    profile_tokens = tokenize(profile_text)
    if not profile_tokens:
        return [0.0] * len(hits)

    similarities: list[float] = []
    for hit in hits:
        hit_tokens = tokenize(hit.ranking_text())
        similarities.append(
            len(profile_tokens & hit_tokens) / len(hit_tokens) if hit_tokens else 0.0
        )
    return similarities


def _to_score(
    similarity: float,
    mode: MatchingMode,
    hit: JobHit,
    wanted_level: str,
) -> float:
    """Map a similarity into the shallow band, then apply the seniority nudge."""
    if mode == "semantic":
        span = _COSINE_CEILING - _COSINE_FLOOR
        fraction = (similarity - _COSINE_FLOOR) / span
    else:
        # Lexical overlap has no comparable floor -- an unrelated snippet really
        # does share close to none of the profile's vocabulary -- so it is
        # stretched from zero, saturating at 0.4, which a genuinely on-topic
        # snippet reaches.
        fraction = similarity / 0.4

    fraction = max(0.0, min(1.0, fraction))
    score = _SCORE_FLOOR + fraction * (_SCORE_CEILING - _SCORE_FLOOR)

    stated = infer_seniority(hit.title)
    if stated and stated != wanted_level and stated not in _ADJACENT.get(wanted_level, set()):
        score -= _SENIORITY_PENALTY

    return round(max(0.0, min(_SCORE_CEILING, score)), 1)
