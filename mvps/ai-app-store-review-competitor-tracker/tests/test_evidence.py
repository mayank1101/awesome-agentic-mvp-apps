"""Tests for critical-review selection and evidence packing."""

from app.core.config import Settings
from app.models.schemas import Review
from app.services.evidence import format_evidence, select_critical
from app.services.guardrails import FENCE_CLOSE, FENCE_OPEN


def _review(rating: int, review_id: str, content: str = "It crashes on launch.") -> Review:
    return Review(id=review_id, rating=rating, title="Bad", content=content, author="a")


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="gsk_test", **overrides)


def test_select_critical_keeps_only_three_stars_and_below():
    reviews = [_review(5, "1"), _review(3, "2"), _review(1, "3")]
    critical = select_critical(reviews)
    assert {r.id for r in critical} == {"2", "3"}


def test_select_critical_preserves_order():
    reviews = [_review(2, "a"), _review(5, "b"), _review(1, "c")]
    assert [r.id for r in select_critical(reviews)] == ["a", "c"]


def test_format_evidence_labels_each_review_with_its_id():
    block = format_evidence([_review(1, "999")], settings=_settings())
    assert "[999]" in block


def test_format_evidence_is_fenced():
    block = format_evidence([_review(1, "1")], settings=_settings())
    assert block.startswith(FENCE_OPEN)
    assert block.endswith(FENCE_CLOSE)


def test_format_evidence_stops_once_the_budget_is_spent():
    reviews = [_review(1, str(i), content="x" * 500) for i in range(50)]
    block = format_evidence(reviews, settings=_settings(evidence_char_budget=1000))
    # Well under including all 50 reviews' worth of content.
    assert len(block) < 50 * 500


def test_format_evidence_always_includes_at_least_one_review():
    # Even if one review alone exceeds the budget, the loop must not emit nothing.
    reviews = [_review(1, "1", content="x" * 5000)]
    block = format_evidence(reviews, settings=_settings(evidence_char_budget=100))
    assert "[1]" in block
