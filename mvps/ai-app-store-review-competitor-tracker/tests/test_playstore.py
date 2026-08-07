"""Tests for the Google Play resolver and review fetch.

`google-play-scraper` has no injectable transport (it wraps `urllib` calls
directly), so these monkeypatch the three functions our modules import from
it -- `search`, `app`, `reviews` -- rather than mocking HTTP. That is the same
boundary discipline as `app/appstore`'s `client: httpx.Client` injection: the
fake stands in for exactly what our code calls, nothing deeper.
"""

from datetime import datetime
from typing import Any

import pytest
from google_play_scraper.exceptions import GooglePlayScraperException, NotFoundError

from app import playstore
from app.core.config import Settings
from app.core.exceptions import AppNotFound, InvalidAppReference, ReviewFetchError
from app.models.schemas import AppCandidate, AppIdentity, Platform
from app.playstore.reviews import fetch_reviews
from app.playstore.search import lookup_app, parse_package_name, resolve, search_apps


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, groq_api_key="gsk_test", **overrides)


def _search_result(**overrides: Any) -> dict:
    payload = {
        "appId": "com.spotify.music",
        "title": "Spotify: Music and Podcasts",
        "developer": "Spotify AB",
        "genre": "Music & Audio",
        "icon": "https://example.com/icon.png",
        "url": "https://play.google.com/store/apps/details?id=com.spotify.music",
        "score": 4.3,
        "ratings": 36_000_000,
    }
    payload.update(overrides)
    return payload


def _review_payload(**overrides: Any) -> dict:
    payload = {
        "reviewId": "abc-123",
        "userName": "Someone",
        "content": "It crashes constantly since the update.",
        "score": 1,
        "reviewCreatedVersion": "9.1.0",
        "at": datetime(2026, 8, 5, 12, 0, 0),
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_a_package_name_parses():
    assert parse_package_name("com.spotify.music") == "com.spotify.music"


def test_play_store_url_parses():
    url = "https://play.google.com/store/apps/details?id=com.spotify.music&hl=en&gl=us"
    assert parse_package_name(url) == "com.spotify.music"


def test_url_without_id_raises():
    with pytest.raises(InvalidAppReference):
        parse_package_name("https://play.google.com/store/apps/collection/topselling_free")


def test_a_free_text_name_is_not_a_package_name():
    assert parse_package_name("Spotify") is None
    assert parse_package_name("music player app") is None


def test_a_single_dotted_word_is_not_a_package_name():
    # Package names need at least two dots (reverse-DNS shaped); this rules
    # out accidentally matching something like a decimal-looking fragment.
    assert parse_package_name("spotify.music") is None


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def test_search_returns_candidates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(playstore.search, "gp_search", lambda *a, **k: [_search_result()])

    candidates = search_apps("spotify", settings=_settings())

    assert len(candidates) == 1
    assert isinstance(candidates[0], AppCandidate)
    assert candidates[0].platform is Platform.ANDROID
    assert candidates[0].package_name == "com.spotify.music"


def test_search_drops_results_with_no_app_id(monkeypatch: pytest.MonkeyPatch):
    # A known google-play-scraper fragility: appId is sometimes None.
    monkeypatch.setattr(
        playstore.search, "gp_search", lambda *a, **k: [_search_result(appId=None)]
    )
    assert search_apps("x", settings=_settings()) == []


def test_search_empty_results_is_not_an_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(playstore.search, "gp_search", lambda *a, **k: [])
    assert search_apps("zzzqqx", settings=_settings()) == []


def test_search_failure_raises_app_not_found(monkeypatch: pytest.MonkeyPatch):
    def boom(*a, **k):
        raise GooglePlayScraperException("boom")

    monkeypatch.setattr(playstore.search, "gp_search", boom)
    with pytest.raises(AppNotFound):
        search_apps("spotify", settings=_settings())


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def test_lookup_returns_an_identity(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(playstore.search, "gp_app", lambda *a, **k: _search_result())

    identity = lookup_app("com.spotify.music", settings=_settings())

    assert isinstance(identity, AppIdentity)
    assert identity.platform is Platform.ANDROID
    assert identity.package_name == "com.spotify.music"
    assert identity.published_rating_count == 36_000_000


def test_lookup_carries_the_requested_country(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(playstore.search, "gp_app", lambda *a, **k: _search_result())
    identity = lookup_app("com.spotify.music", country="in", settings=_settings())
    assert identity.country == "in"


def test_lookup_not_found_raises_app_not_found(monkeypatch: pytest.MonkeyPatch):
    def boom(*a, **k):
        raise NotFoundError("404")

    monkeypatch.setattr(playstore.search, "gp_app", boom)
    with pytest.raises(AppNotFound):
        lookup_app("com.fake.nonexistent", settings=_settings())


# --------------------------------------------------------------------------- #
# resolve()
# --------------------------------------------------------------------------- #


def test_resolve_with_a_package_name_skips_search(monkeypatch: pytest.MonkeyPatch):
    calls = {"search": 0, "app": 0}
    monkeypatch.setattr(
        playstore.search, "gp_app", lambda *a, **k: calls.__setitem__("app", calls["app"] + 1) or _search_result()
    )
    monkeypatch.setattr(
        playstore.search,
        "gp_search",
        lambda *a, **k: calls.__setitem__("search", calls["search"] + 1) or [],
    )

    result = resolve("com.spotify.music", settings=_settings())

    assert isinstance(result, AppIdentity)
    assert calls == {"search": 0, "app": 1}


def test_resolve_with_a_name_returns_candidates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        playstore.search,
        "gp_search",
        lambda *a, **k: [_search_result(), _search_result(appId="com.other.app")],
    )
    result = resolve("spotify", settings=_settings())
    assert isinstance(result, list)
    assert len(result) == 2


def test_resolve_with_a_name_and_no_matches_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(playstore.search, "gp_search", lambda *a, **k: [])
    with pytest.raises(AppNotFound):
        resolve("zzzqqx", settings=_settings())


# --------------------------------------------------------------------------- #
# Reviews
# --------------------------------------------------------------------------- #


def test_fetch_parses_reviews(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        playstore.reviews, "gp_reviews", lambda *a, **k: ([_review_payload()], None)
    )

    reviews = fetch_reviews("com.spotify.music", settings=_settings())

    assert len(reviews) == 1
    assert reviews[0].rating == 1
    assert reviews[0].id == "abc-123"
    assert reviews[0].title is None  # Play reviews have no title, unlike iOS.


def test_fetch_drops_entries_with_no_score(monkeypatch: pytest.MonkeyPatch):
    malformed = {"reviewId": "x", "content": "no score field"}
    monkeypatch.setattr(
        playstore.reviews, "gp_reviews", lambda *a, **k: ([_review_payload(), malformed], None)
    )
    reviews = fetch_reviews("com.spotify.music", settings=_settings())
    assert len(reviews) == 1


def test_fetch_respects_max_reviews(monkeypatch: pytest.MonkeyPatch):
    payloads = [_review_payload(reviewId=str(i)) for i in range(10)]
    monkeypatch.setattr(playstore.reviews, "gp_reviews", lambda *a, **k: (payloads, None))
    reviews = fetch_reviews("com.spotify.music", settings=_settings(max_reviews=3))
    assert len(reviews) == 3


def test_fetch_trims_content_to_the_char_cap(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        playstore.reviews,
        "gp_reviews",
        lambda *a, **k: ([_review_payload(content="x" * 5000)], None),
    )
    reviews = fetch_reviews("com.spotify.music", settings=_settings(review_char_cap=100))
    assert len(reviews[0].content) <= 100


def test_fetch_empty_is_not_an_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(playstore.reviews, "gp_reviews", lambda *a, **k: ([], None))
    assert fetch_reviews("com.spotify.music", settings=_settings()) == []


def test_fetch_failure_raises_review_fetch_error(monkeypatch: pytest.MonkeyPatch):
    def boom(*a, **k):
        raise GooglePlayScraperException("boom")

    monkeypatch.setattr(playstore.reviews, "gp_reviews", boom)
    with pytest.raises(ReviewFetchError):
        fetch_reviews("com.spotify.music", settings=_settings())


def test_fetch_bare_network_error_raises_review_fetch_error(monkeypatch: pytest.MonkeyPatch):
    # google-play-scraper's own HTTP layer raises un-subclassed exceptions
    # (urllib errors) rather than always wrapping them.
    def boom(*a, **k):
        raise OSError("network unreachable")

    monkeypatch.setattr(playstore.reviews, "gp_reviews", boom)
    with pytest.raises(ReviewFetchError):
        fetch_reviews("com.spotify.music", settings=_settings())


def test_fetch_passes_through_the_requested_country(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_reviews(package, **kwargs):
        captured.update(kwargs)
        return [], None

    monkeypatch.setattr(playstore.reviews, "gp_reviews", fake_reviews)
    fetch_reviews("com.spotify.music", country="in", settings=_settings())
    assert captured["country"] == "in"
