"""Tests for rating arithmetic (SC-3)."""

from datetime import datetime

from app.models.schemas import Review
from app.services.stats import compute_stats


def _review(rating: int, updated: str | None = None, review_id: str = "1") -> Review:
    return Review(
        id=review_id,
        rating=rating,
        title="t",
        content="c",
        author="a",
        updated=datetime.fromisoformat(updated) if updated else None,
    )


def test_empty_sample_is_all_zero():
    stats = compute_stats([])
    assert stats.fetched_count == 0
    assert stats.critical_count == 0
    assert stats.average == 0.0
    assert stats.critical_share == 0.0


def test_distribution_counts_each_star():
    stats = compute_stats([_review(5), _review(5), _review(1)])
    assert stats.distribution == {1: 1, 2: 0, 3: 0, 4: 0, 5: 2}


def test_critical_count_is_three_stars_and_below():
    reviews = [_review(1), _review(2), _review(3), _review(4), _review(5)]
    stats = compute_stats(reviews)
    assert stats.critical_count == 3


def test_average_is_exact_arithmetic():
    stats = compute_stats([_review(1), _review(5)])
    assert stats.average == 3.0


def test_critical_share_is_a_fraction():
    stats = compute_stats([_review(1), _review(5)])
    assert stats.critical_share == 0.5


def test_date_range_uses_min_and_max_of_dated_reviews():
    stats = compute_stats(
        [
            _review(1, updated="2026-01-15T00:00:00+00:00"),
            _review(2, updated="2026-03-01T00:00:00+00:00"),
            _review(3),  # undated, must not break min/max
        ]
    )
    assert stats.oldest == datetime.fromisoformat("2026-01-15T00:00:00+00:00")
    assert stats.newest == datetime.fromisoformat("2026-03-01T00:00:00+00:00")


def test_all_undated_yields_no_range():
    stats = compute_stats([_review(1), _review(2)])
    assert stats.oldest is None
    assert stats.newest is None
