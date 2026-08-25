import pytest

from app.core.exceptions import SearchError
from app.services import search


def test_build_queries_has_three_categories_at_minimum() -> None:
    queries = search.build_queries("Lisbon", days=2)
    categories = {c for c, _ in queries}
    assert categories == {"activities", "accommodation", "tips"}
    assert len(queries) == 3


def test_longer_trips_add_more_activity_queries() -> None:
    short = search.build_queries("Lisbon", days=2)
    week = search.build_queries("Lisbon", days=7)
    long = search.build_queries("Lisbon", days=10)

    assert len(week) > len(short)
    assert len(long) > len(week)


def test_canonical_url_strips_tracking_params() -> None:
    a = search.canonical_url("https://example.com/page?utm_source=x&id=42")
    b = search.canonical_url("https://www.example.com/page/?id=42")
    assert a == b


def test_canonical_url_handles_unparseable_input() -> None:
    assert search.canonical_url("not a url") == "not a url"


def test_domain_of_strips_www() -> None:
    assert search.domain_of("https://www.example.com/x") == "example.com"


def test_gather_evidence_dedupes_and_categorises(
    monkeypatch: pytest.MonkeyPatch, trip_request, tavily_response: dict
) -> None:
    monkeypatch.setattr(search, "post_json", lambda path, payload: tavily_response)

    sections = search.gather_evidence(trip_request)

    assert {s.category for s in sections} == {"activities", "accommodation", "tips"}
    activities = next(s for s in sections if s.category == "activities")
    assert len(activities.items) == 2
    # ids are unique across the whole evidence set
    all_ids = [item.id for section in sections for item in section.items]
    assert len(all_ids) == len(set(all_ids))


def test_gather_evidence_drops_duplicate_urls_across_queries(
    monkeypatch: pytest.MonkeyPatch, trip_request, tavily_response: dict
) -> None:
    # Every query returns the same two results; the same URL must not appear twice.
    monkeypatch.setattr(search, "post_json", lambda path, payload: tavily_response)

    sections = search.gather_evidence(trip_request)
    seen_urls = [item.url for section in sections for item in section.items]
    assert len(seen_urls) == len(set(seen_urls))


def test_gather_evidence_survives_one_failed_query(
    monkeypatch: pytest.MonkeyPatch, trip_request, tavily_response: dict
) -> None:
    calls = {"n": 0}

    def fake_post_json(path: str, payload: dict) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise SearchError("boom")
        return tavily_response

    monkeypatch.setattr(search, "post_json", fake_post_json)

    sections = search.gather_evidence(trip_request)
    assert any(section.items for section in sections)


def test_gather_evidence_raises_when_every_query_fails(
    monkeypatch: pytest.MonkeyPatch, trip_request
) -> None:
    def always_fails(path: str, payload: dict) -> dict:
        raise SearchError("boom")

    monkeypatch.setattr(search, "post_json", always_fails)

    with pytest.raises(SearchError):
        search.gather_evidence(trip_request)


def test_gather_evidence_caps_items_per_category(
    monkeypatch: pytest.MonkeyPatch, trip_request
) -> None:
    from app.core.config import Settings

    settings = Settings(groq_api_key="x", tavily_api_key="y", max_evidence_per_category=1)
    monkeypatch.setattr(search, "get_settings", lambda: settings)
    monkeypatch.setattr(
        search,
        "post_json",
        lambda path, payload: {
            "results": [
                {"title": "A", "url": "https://a.example.com", "content": "a", "score": 0.5},
                {"title": "B", "url": "https://b.example.com", "content": "b", "score": 0.9},
            ]
        },
    )

    sections = search.gather_evidence(trip_request)
    for section in sections:
        assert len(section.items) <= 1
