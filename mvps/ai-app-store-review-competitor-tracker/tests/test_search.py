"""Tests for app resolution: id/URL parsing, search, and lookup."""

import httpx
import pytest

from app.appstore.search import lookup_app, parse_track_id, resolve, search_apps
from app.core.config import Settings
from app.core.exceptions import AppNotFound, InvalidAppReference
from app.models.schemas import AppCandidate, AppIdentity


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="gsk_test", **overrides)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _search_result(**overrides) -> dict:
    payload = {
        "trackId": 1232780281,
        "trackName": "Notion",
        "artistName": "Notion Labs",
        "primaryGenreName": "Productivity",
        "artworkUrl100": "https://example.com/icon.png",
        "trackViewUrl": "https://apps.apple.com/us/app/notion/id1232780281",
        "averageUserRating": 4.7,
        "userRatingCount": 89837,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_bare_numeric_id_parses():
    assert parse_track_id("1232780281") == 1232780281


def test_bare_numeric_id_tolerates_whitespace():
    assert parse_track_id("  1232780281  ") == 1232780281


def test_app_store_url_parses():
    url = "https://apps.apple.com/us/app/notion-notes-tasks-ai/id1232780281?uo=4"
    assert parse_track_id(url) == 1232780281


def test_url_without_id_raises():
    with pytest.raises(InvalidAppReference):
        parse_track_id("https://apps.apple.com/us/app/notion/")


def test_a_name_is_not_a_track_id():
    assert parse_track_id("Notion") is None


def test_short_digit_strings_fall_through_to_search():
    # 6-12 digits is the id shape; shorter numeric strings are not treated as an id.
    assert parse_track_id("12345") is None


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def test_search_returns_candidates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_search_result()]})

    candidates = search_apps("notion", settings=_settings(), client=_client(handler))

    assert len(candidates) == 1
    assert isinstance(candidates[0], AppCandidate)
    assert candidates[0].track_id == 1232780281


def test_search_drops_results_missing_required_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"trackName": "No id or url"}]})

    assert search_apps("x", settings=_settings(), client=_client(handler)) == []


def test_search_empty_results_is_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    assert search_apps("zzzqqx", settings=_settings(), client=_client(handler)) == []


def test_search_network_failure_raises_app_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(AppNotFound):
        search_apps("notion", settings=_settings(), client=_client(handler))


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def test_lookup_returns_an_identity():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_search_result()]})

    identity = lookup_app(1232780281, settings=_settings(), client=_client(handler))

    assert isinstance(identity, AppIdentity)
    assert identity.track_name == "Notion"
    assert identity.published_rating_count == 89837


def test_lookup_with_no_results_raises_app_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    with pytest.raises(AppNotFound) as caught:
        lookup_app(999999999, settings=_settings(), client=_client(handler))
    assert "999999999" in caught.value.query


# --------------------------------------------------------------------------- #
# resolve()
# --------------------------------------------------------------------------- #


def test_resolve_with_an_id_skips_search():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"results": [_search_result()]})

    result = resolve("1232780281", settings=_settings(), client=_client(handler))

    assert isinstance(result, AppIdentity)
    assert calls == ["/lookup"]


def test_resolve_with_a_name_returns_candidates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_search_result(), _search_result(trackId=2)]})

    result = resolve("notion", settings=_settings(), client=_client(handler))

    assert isinstance(result, list)
    assert len(result) == 2


def test_resolve_with_a_name_and_no_matches_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    with pytest.raises(AppNotFound):
        resolve("zzzqqx", settings=_settings(), client=_client(handler))
