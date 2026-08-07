"""Matching each job requirement to the resume lines that could support it.

This runs *before* the model is asked for a verdict, and it exists for two
reasons.

**It keeps the assessment call small and grounded.** Instead of "here is a whole
resume and 20 requirements, decide", the model is shown each requirement beside
the three resume lines most likely to be relevant. That is a smaller prompt, a
cheaper call on a free tier, and a question with a much narrower space of wrong
answers.

**It gives the score a component the model cannot flatter.** The similarity
number is computed here, from text, and the pipeline uses it to override a model
verdict that claims coverage no resume line supports. An instruction is a
preference; arithmetic over the candidate's own words is a check.

Two modes, chosen by whether a Mistral key is configured. Semantic matching is
better -- "shipped microservices in Go" should match "Golang experience", and
lexical overlap says 0.0 to that -- but the lexical path means the app degrades
instead of failing, and the report always says which mode it used.
"""

import math
import re
import statistics
from dataclasses import dataclass, field

from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger
from app.models.schemas import JobRequirement, MatchingMode
from app.services.embeddings import embed_texts

logger = get_logger(__name__)

#: Words that carry no signal about whether a requirement is met. Kept short on
#: purpose: an aggressive stoplist deletes exactly the domain terms that matter
#: ("C", "R", "Go" are all languages).
_STOPWORDS = frozenset(
    # One string, split at import time -- see the note in `provenance._ALLOWED`.
    """
    a an the and or of in on at to for with by from as is are was were be been being
    you your our we they it this that these those will shall should would can could
    have has had do does did not no if then than so such about into over under
    experience experienced years year strong excellent good ability able skills skill
    knowledge understanding working work works worked using use used plus preferred
    required requirements must nice familiarity proficiency proficient demonstrated
    """.split()
)

#: Tokens are alphanumerics plus the punctuation that lives inside real technical
#: terms: `node.js`, `ci/cd`, `c++`, `scikit-learn`.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#./_-]*")


@dataclass
class RequirementMatch:
    """The resume lines that best support one requirement.

    Attributes:
        requirement_id: The `R-nn` this belongs to.
        similarity: Best similarity found, 0-1. Comparable within a run and
            within a mode; not comparable across modes, which is why the mode is
            reported alongside it.
        baseline: Mean similarity between this requirement and *every* resume
            line. With hosted embeddings this is never near zero -- two pieces of
            professional English are ~0.6 similar before either says anything --
            so the interesting quantity is how far the best line stands above the
            resume's own floor, not the raw number.
        evidence: The best-matching resume lines, strongest first.
    """

    requirement_id: str
    similarity: float = 0.0
    baseline: float = 0.0
    evidence: list[str] = field(default_factory=list)

    @property
    def margin(self) -> float:
        """How far the best line stands above this requirement's baseline."""
        return max(0.0, self.similarity - self.baseline)


def tokenize(text: str) -> set[str]:
    """Reduce text to the content tokens used for lexical matching."""
    return {
        token
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def lexical_similarity(requirement: str, evidence: str) -> float:
    """How much of the requirement's vocabulary appears in the evidence.

    Coverage of the requirement, not symmetric overlap: a long bullet that
    happens to contain every word of a short requirement *does* satisfy it, and
    Jaccard would punish that for being long.

    Args:
        requirement: The requirement text.
        evidence: One resume line.

    Returns:
        0-1.
    """
    wanted = tokenize(requirement)
    if not wanted:
        return 0.0
    found = tokenize(evidence)
    return len(wanted & found) / len(wanted)


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two vectors, clamped to 0-1.

    Negative cosines are clamped rather than kept: below zero, "less similar than
    unrelated" is not a distinction this app has any use for, and a negative
    number in a score column reads as a bug.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / norm))


def match_requirements(
    requirements: list[JobRequirement],
    evidence_texts: list[str],
    *,
    top_k: int = 3,
) -> tuple[list[RequirementMatch], MatchingMode]:
    """Find the best resume evidence for each requirement.

    Args:
        requirements: Extracted requirements, in order.
        evidence_texts: Every resume line that could support a requirement.
        top_k: How many evidence lines to keep per requirement.

    Returns:
        One match per requirement in the same order, and the mode that produced
        them. The mode is ``lexical`` whenever embeddings were unavailable *or*
        failed -- an embedding failure degrades the run, it does not end it.
    """
    if not requirements:
        return [], "lexical"
    if not evidence_texts:
        return [RequirementMatch(requirement_id=r.id) for r in requirements], "lexical"

    try:
        return _semantic_match(requirements, evidence_texts, top_k), "semantic"
    except EmbeddingError as exc:
        logger.warning("Falling back to lexical matching: %s", exc)
        return _lexical_match(requirements, evidence_texts, top_k), "lexical"


def _semantic_match(
    requirements: list[JobRequirement],
    evidence_texts: list[str],
    top_k: int,
) -> list[RequirementMatch]:
    """Match by cosine similarity over `mistral-embed` vectors.

    One request for everything: requirements and evidence go in the same batched
    call, because two calls is two chances to hit a per-minute rate limit.
    """
    vectors = embed_texts([r.text for r in requirements] + evidence_texts)
    split = len(requirements)
    requirement_vectors, evidence_vectors = vectors[:split], vectors[split:]

    matches: list[RequirementMatch] = []
    for requirement, vector in zip(requirements, requirement_vectors, strict=True):
        scored = sorted(
            (
                (cosine(vector, other), text)
                for other, text in zip(evidence_vectors, evidence_texts, strict=True)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        matches.append(_to_match(requirement.id, scored, top_k))
    return matches


def _lexical_match(
    requirements: list[JobRequirement],
    evidence_texts: list[str],
    top_k: int,
) -> list[RequirementMatch]:
    """Match by requirement-vocabulary coverage. The offline fallback."""
    matches: list[RequirementMatch] = []
    for requirement in requirements:
        scored = sorted(
            ((lexical_similarity(requirement.text, text), text) for text in evidence_texts),
            key=lambda pair: pair[0],
            reverse=True,
        )
        matches.append(_to_match(requirement.id, scored, top_k))
    return matches


def _to_match(
    requirement_id: str,
    scored: list[tuple[float, str]],
    top_k: int,
) -> RequirementMatch:
    """Build a match from a ranked list, dropping evidence that scored nothing.

    The baseline is the mean over *every* line, computed before the top-k cut --
    it is the resume's own noise floor for this requirement, and cutting first
    would measure the noise floor of the best three lines instead.
    """
    kept = [(score, text) for score, text in scored[:top_k] if score > 0.0]
    baseline = statistics.fmean(score for score, _ in scored) if scored else 0.0
    return RequirementMatch(
        requirement_id=requirement_id,
        similarity=round(kept[0][0], 4) if kept else 0.0,
        baseline=round(baseline, 4),
        evidence=[text for _, text in kept],
    )
