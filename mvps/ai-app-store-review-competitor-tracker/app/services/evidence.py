"""Turning fetched reviews into the fenced, budgeted evidence the model reads.

Only critical (≤3★) reviews are packed -- see PRD §10 Q-2: gap analysis is a
question about what is failing users, and mixing in five-star praise would
dilute the one thing this tool is for. The full fetched sample, including
positive reviews, still feeds `app/services/stats.py` and the raw-review
browser; this module only decides what the model sees.
"""

from app.core.config import Settings
from app.models.schemas import Review
from app.services.guardrails import defang_fence_markers, fence


def select_critical(reviews: list[Review]) -> list[Review]:
    """Return the ≤3★ reviews, most recent first.

    Args:
        reviews: The fetched sample, in feed order (already most-recent-first).

    Returns:
        The critical subset, order preserved.
    """
    return [r for r in reviews if r.is_critical]


def _trim(text: str, cap: int) -> str:
    """Cut text to `cap` characters, preferring a word boundary."""
    if len(text) <= cap:
        return text
    cut = text[:cap]
    space = cut.rfind(" ")
    if space > cap - 80:
        cut = cut[:space]
    return cut.rstrip() + "…"


def format_evidence(reviews: list[Review], *, settings: Settings) -> str:
    """Render packed critical reviews as the fenced block the model reads.

    Each review is labelled with its id, star rating, and version -- but the
    model is never told the id is something to reproduce; it is a citation
    key, resolved back to the real text by the renderer (see
    `app/services/guardrails.py` module docstring).

    Args:
        reviews: The critical reviews to pack, already selected.
        settings: Runtime configuration, for the per-review and total caps.

    Returns:
        The fenced evidence block.
    """
    per_review_cap = settings.review_char_cap
    budget = settings.evidence_char_budget

    blocks: list[str] = []
    used = 0
    for review in reviews:
        version = f", v{review.version}" if review.version else ""
        # Android reviews have no title field at all, unlike iOS.
        title = f" — {defang_fence_markers(review.title)}" if review.title else ""
        content = defang_fence_markers(_trim(review.content, per_review_cap))
        block = f"[{review.id}] {review.rating}★{version}{title}\n{content}"

        if used + len(block) > budget and blocks:
            # Proportional trimming is overkill for a single flat pool of
            # reviews (unlike the multi-section evidence in the competitor
            # analyzer) -- once the budget is spent, stop including reviews
            # rather than shrink every one of them.
            break

        blocks.append(block)
        used += len(block)

    return fence("\n\n".join(blocks))
