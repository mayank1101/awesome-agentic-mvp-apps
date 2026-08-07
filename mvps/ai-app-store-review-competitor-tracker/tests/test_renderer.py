"""Tests for the renderer -- the grounding guarantees (SC-1, SC-2)."""

from datetime import date

from app.models.schemas import AppIdentity, FeatureGap, Platform, Review, ReviewStats
from app.services.renderer import render_report


def _identity() -> AppIdentity:
    return AppIdentity(
        platform=Platform.IOS,
        track_id=1,
        track_name="Acme App",
        artist_name="Acme Inc",
        app_store_url="https://apps.apple.com/app/id1",
        published_average_rating=4.2,
        published_rating_count=1000,
    )


def _stats(critical=2, total=5) -> ReviewStats:
    return ReviewStats(
        fetched_count=total,
        distribution={5: total - critical, 4: 0, 3: 0, 2: 0, 1: critical},
        critical_count=critical,
    )


def _review(review_id: str, content: str = "This app crashes constantly on launch.") -> Review:
    return Review(id=review_id, rating=1, title="Crashes", content=content, author="a")


def test_a_gap_with_a_resolvable_citation_is_rendered_with_the_real_excerpt():
    reviews = (_review("1", content="The sync feature loses my data every single time."),)
    gap = FeatureGap(title="Sync loses data", description="Users report data loss.", severity="high", review_ids=("1",))

    markdown, resolved = render_report(
        _identity(), _stats(), reviews, (gap,), generated_on=date(2026, 1, 1)
    )

    assert "Sync loses data" in markdown
    assert "sync feature loses my data" in markdown
    assert len(resolved) == 1


def test_a_gap_with_no_resolvable_citation_is_dropped():
    # SC-2: the model cited an id that was never in the fetched sample.
    reviews = (_review("1"),)
    gap = FeatureGap(title="Invented gap", description="x", severity="high", review_ids=("999",))

    markdown, resolved = render_report(
        _identity(), _stats(), reviews, (gap,), generated_on=date(2026, 1, 1)
    )

    assert "Invented gap" not in markdown
    assert resolved == ()


def test_no_excerpt_text_appears_that_was_not_in_a_fetched_review():
    # SC-1, structurally: every quoted line under a gap is a substring of some
    # fetched review's content.
    reviews = (
        _review("1", content="Notifications never arrive on time for me."),
        _review("2", content="The export button does nothing when tapped."),
    )
    gaps = (
        FeatureGap(title="Notifications broken", description="d", severity="high", review_ids=("1",)),
        FeatureGap(title="Export broken", description="d", severity="medium", review_ids=("2",)),
    )

    markdown, _ = render_report(_identity(), _stats(), reviews, gaps, generated_on=date(2026, 1, 1))

    for review in reviews:
        # The renderer must have pulled this exact text in somewhere.
        assert review.content in markdown


def test_insufficient_signal_shows_a_message_and_no_gap_section_content():
    markdown, resolved = render_report(
        _identity(),
        _stats(critical=1),
        (_review("1"),),
        (),
        generated_on=date(2026, 1, 1),
        insufficient_signal=True,
    )
    assert "Not enough critical reviews" in markdown
    assert resolved == ()


def test_analysis_failed_is_stated_plainly():
    markdown, _ = render_report(
        _identity(), _stats(), (_review("1"),), (), generated_on=date(2026, 1, 1), analysis_failed=True
    )
    assert "Analysis failed" in markdown


def test_snapshot_includes_published_rating_and_sample_stats():
    markdown, _ = render_report(
        _identity(), _stats(critical=2, total=5), (), (), generated_on=date(2026, 1, 1)
    )
    assert "4.2" in markdown
    assert "5 most recent reviews" in markdown


def test_gap_excerpts_are_sanitized():
    hostile = _review("1", content="<script>alert(1)</script> also crashes")
    gap = FeatureGap(title="Crashes", description="d", severity="high", review_ids=("1",))
    markdown, _ = render_report(_identity(), _stats(), (hostile,), (gap,), generated_on=date(2026, 1, 1))
    assert "<script>" not in markdown


def test_there_is_no_raw_review_dump():
    # Replaced by the closing summary -- a fifty-review list added length
    # without adding information the gap excerpts weren't already providing.
    reviews = tuple(_review(str(i), content=f"unique review body {i}") for i in range(5))
    markdown, _ = render_report(_identity(), _stats(), reviews, (), generated_on=date(2026, 1, 1))
    assert "All fetched reviews" not in markdown
    # None of the un-cited review bodies should appear verbatim -- only a gap
    # citation earns a review a place in the document.
    for review in reviews:
        assert review.content not in markdown


def test_summary_section_reports_sample_composition():
    markdown, _ = render_report(
        _identity(), _stats(critical=2, total=5), (), (), generated_on=date(2026, 1, 1)
    )
    assert "## Summary" in markdown
    assert "5 reviews sampled" in markdown
    assert "2 critical" in markdown


def test_summary_section_lists_found_gap_titles():
    reviews = (_review("1"),)
    gap = FeatureGap(title="Sync loses data", description="d", severity="high", review_ids=("1",))
    markdown, _ = render_report(_identity(), _stats(), reviews, (gap,), generated_on=date(2026, 1, 1))

    summary = markdown.split("## Summary")[1]
    assert "1 gap(s) found" in summary
    assert "Sync loses data" in summary


def test_summary_section_states_insufficient_signal():
    markdown, _ = render_report(
        _identity(),
        _stats(critical=1),
        (_review("1"),),
        (),
        generated_on=date(2026, 1, 1),
        insufficient_signal=True,
    )
    summary = markdown.split("## Summary")[1]
    assert "Not enough critical reviews" in summary
