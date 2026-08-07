"""Rating arithmetic over the fetched sample (SC-3).

Deliberately the simplest module in the app: a star count is either right or a
bug, and there is no reason to route arithmetic through a model call that
could get it wrong, cost latency, and cost tokens for no benefit.
"""

from app.models.schemas import Review, ReviewStats


def compute_stats(reviews: list[Review]) -> ReviewStats:
    """Summarise a fetched sample of reviews.

    Args:
        reviews: The fetched reviews, in any order.

    Returns:
        Distribution, critical count, and the sample's date range. All-zero
        fields for an empty sample rather than raising -- an app with zero
        reviews in the sample is a legitimate, renderable outcome.
    """
    distribution = {stars: 0 for stars in range(1, 6)}
    for review in reviews:
        distribution[review.rating] += 1

    dated = [r.updated for r in reviews if r.updated is not None]

    return ReviewStats(
        fetched_count=len(reviews),
        distribution=distribution,
        critical_count=sum(1 for r in reviews if r.is_critical),
        oldest=min(dated) if dated else None,
        newest=max(dated) if dated else None,
    )
