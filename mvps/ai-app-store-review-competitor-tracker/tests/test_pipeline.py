"""End-to-end tests for the orchestrator, with a fake fetcher and a fake model.

This is the last stage before anything imports Streamlit, so these tests prove
the app works without a UI.
"""

import json
from typing import Any

import pytest

from app.agents import analyzer
from app.core.config import Settings
from app.models.schemas import AppIdentity, Platform, Review
from app.services.pipeline import run_to_completion
from tests.test_analyzer import FakeAgent


def _identity() -> AppIdentity:
    return AppIdentity(
        platform=Platform.IOS,
        track_id=1,
        track_name="Acme App",
        artist_name="Acme Inc",
        app_store_url="https://apps.apple.com/app/id1",
    )


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, groq_api_key="gsk_test", **overrides)


def _review(review_id: str, rating: int, content: str = "It crashes on launch.") -> Review:
    return Review(id=review_id, rating=rating, title="t", content=content, author="a")


def _valid_reply(review_ids: list[str]) -> str:
    return json.dumps(
        {
            "gaps": [
                {
                    "title": "App crashes on launch",
                    "description": "Several users report a crash.",
                    "severity": "high",
                    "review_ids": review_ids,
                }
            ]
        }
    )


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch):
    def install(replies: list[Any] | None = None) -> None:
        monkeypatch.setattr(analyzer, "_build_agent", lambda: FakeAgent(replies or [_valid_reply(["1", "2"])]))
        monkeypatch.setattr(analyzer, "build_options", lambda **_: {})

    install()
    return install


def _fetcher(reviews: list[Review]):
    def fetch(identity: AppIdentity, settings: Settings) -> list[Review]:
        return reviews

    return fetch


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_a_full_run_produces_a_report_with_a_gap(fake_model):
    reviews = [_review(str(i), 1) for i in range(6)]
    events, report = run_to_completion(
        _identity(), settings=_settings(min_critical_reviews=5), fetch_reviews=_fetcher(reviews)
    )

    assert report.stats.fetched_count == 6
    assert len(report.gaps) == 1
    assert "App crashes on launch" in report.markdown
    assert events[-1].stage == "done"


def test_progress_events_cover_every_stage(fake_model):
    reviews = [_review(str(i), 1) for i in range(6)]
    events, _ = run_to_completion(
        _identity(), settings=_settings(min_critical_reviews=5), fetch_reviews=_fetcher(reviews)
    )
    stages = [e.stage for e in events]
    assert stages[0] == "resolving"
    assert stages[-1] == "done"
    for stage in ("fetching", "analyzing", "rendering"):
        assert stage in stages


# --------------------------------------------------------------------------- #
# Insufficient signal (SC-4)
# --------------------------------------------------------------------------- #


def test_too_few_critical_reviews_skips_the_model_call(fake_model):
    reviews = [_review("1", 1), _review("2", 5), _review("3", 5)]
    _, report = run_to_completion(
        _identity(), settings=_settings(min_critical_reviews=5), fetch_reviews=_fetcher(reviews)
    )

    assert report.insufficient_signal is True
    assert report.gaps == ()
    assert "Not enough critical reviews" in report.markdown


def test_zero_reviews_is_a_renderable_report(fake_model):
    _, report = run_to_completion(
        _identity(), settings=_settings(min_critical_reviews=5), fetch_reviews=_fetcher([])
    )
    assert report.stats.fetched_count == 0
    assert report.insufficient_signal is True
    assert report.markdown


# --------------------------------------------------------------------------- #
# Analysis failure
# --------------------------------------------------------------------------- #


def test_analysis_failure_still_produces_a_report(fake_model):
    fake_model(["not json", "still not json"])
    reviews = [_review(str(i), 1) for i in range(6)]
    _, report = run_to_completion(
        _identity(), settings=_settings(min_critical_reviews=5), fetch_reviews=_fetcher(reviews)
    )

    assert report.analysis_failed is True
    assert report.gaps == ()
    assert "Analysis failed" in report.markdown
    # Reviews and stats are unaffected by the model failure.
    assert report.stats.fetched_count == 6


def test_gaps_citing_unknown_ids_are_dropped_from_the_final_report(fake_model):
    fake_model([_valid_reply(["999"])])
    reviews = [_review(str(i), 1) for i in range(6)]
    _, report = run_to_completion(
        _identity(), settings=_settings(min_critical_reviews=5), fetch_reviews=_fetcher(reviews)
    )

    assert report.gaps == ()
    assert "App crashes on launch" not in report.markdown
