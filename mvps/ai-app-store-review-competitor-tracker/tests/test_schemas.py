"""Tests for the domain models' validation behaviour."""

import pytest
from pydantic import ValidationError

from app.models.schemas import FeatureGap, GapAnalysisResult, Report, Review, ReviewStats


def test_feature_gap_coerces_integer_ids_to_strings():
    gap = FeatureGap(title="t", description="d", severity="high", review_ids=[1, 2, 3])
    assert gap.review_ids == ("1", "2", "3")


def test_feature_gap_coerces_a_single_id_to_a_tuple():
    gap = FeatureGap(title="t", description="d", severity="low", review_ids="42")
    assert gap.review_ids == ("42",)


def test_feature_gap_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        FeatureGap(title="t", description="d", severity="low", extra_field="nope")


def test_feature_gap_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        FeatureGap(title="t", description="d", severity="critical")


def test_gap_analysis_result_empty_has_no_gaps():
    assert GapAnalysisResult.empty().gaps == ()


def test_review_rejects_out_of_range_rating():
    with pytest.raises(ValidationError):
        Review(id="1", rating=6, title="t", content="c", author="a")


def test_review_is_critical_property():
    assert Review(id="1", rating=3, title="t", content="c", author="a").is_critical is True
    assert Review(id="1", rating=4, title="t", content="c", author="a").is_critical is False


def test_report_rejects_empty_markdown():
    with pytest.raises(ValidationError):
        Report(
            identity={
                "platform": "ios",
                "track_id": 1,
                "track_name": "App",
                "artist_name": "Dev",
                "app_store_url": "https://apps.apple.com/app/id1",
            },
            stats=ReviewStats(fetched_count=0, distribution={}, critical_count=0),
            markdown="   ",
            generated_on="2026-01-01",
        )
