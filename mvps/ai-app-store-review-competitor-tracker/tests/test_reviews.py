"""Tests for the review feed fetch.

Covers the parsing shape confirmed by direct testing against the live
endpoint (docs/01-prd.md §7): entries missing `im:rating` are dropped, a
single-review feed's `entry` arrives as an object rather than a list, and an
empty feed (no `entry` key at all -- the shape Apple's feed returns for a
broken `page=` parameter or a genuinely empty app) is a legitimate empty
result, not an error.
"""

import httpx
import pytest

from app.appstore import reviews as reviews_module
from app.appstore.reviews import fetch_reviews
from app.core.config import Settings
from app.core.exceptions import ReviewFetchError, ReviewFetchTimeout


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="gsk_test", **overrides)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch):
    """Skip the real backoff before the empty-response retry in every test."""
    monkeypatch.setattr(reviews_module.time, "sleep", lambda _seconds: None)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _entry(review_id="111", rating="1", title="Bad", content="It crashes.", version="1.0"):
    return {
        "author": {"name": {"label": "Someone"}},
        "updated": {"label": "2026-08-05T19:22:55-07:00"},
        "im:rating": {"label": rating},
        "im:version": {"label": version},
        "id": {"label": review_id},
        "title": {"label": title},
        "content": {"label": content},
    }


def test_fetch_parses_entries_into_reviews():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"feed": {"entry": [_entry(), _entry(review_id="222")]}})

    reviews = fetch_reviews(1, settings=_settings(), client=_client(handler))

    assert len(reviews) == 2
    assert reviews[0].rating == 1
    assert reviews[0].id == "111"


def test_a_single_review_arrives_as_an_object_not_a_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"feed": {"entry": _entry()}})

    reviews = fetch_reviews(1, settings=_settings(), client=_client(handler))
    assert len(reviews) == 1


def test_an_empty_feed_with_no_entry_key_is_an_empty_list_not_an_error():
    # The exact shape Apple's feed returns for a broken `page=` parameter, or
    # for the intermittent per-id emptiness described in the module docstring.
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"feed": {"author": {}, "title": {}}})

    assert fetch_reviews(1, settings=_settings(), client=_client(handler)) == []
    # Retried once before accepting the empty result as genuine.
    assert len(calls) == 2


def test_a_second_attempt_recovers_from_a_transient_empty_response():
    responses = [
        httpx.Response(200, json={"feed": {"author": {}}}),
        httpx.Response(200, json={"feed": {"entry": [_entry()]}}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    reviews = fetch_reviews(1, settings=_settings(), client=_client(handler))
    assert len(reviews) == 1


def test_a_genuinely_populated_first_response_does_not_retry():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"feed": {"entry": [_entry()]}})

    fetch_reviews(1, settings=_settings(), client=_client(handler))
    assert len(calls) == 1


def test_entries_missing_a_rating_are_dropped():
    def handler(request: httpx.Request) -> httpx.Response:
        malformed = {"id": {"label": "999"}, "title": {"label": "no rating field"}}
        return httpx.Response(200, json={"feed": {"entry": [_entry(), malformed]}})

    reviews = fetch_reviews(1, settings=_settings(), client=_client(handler))
    assert len(reviews) == 1


def test_results_are_capped_at_max_reviews():
    def handler(request: httpx.Request) -> httpx.Response:
        entries = [_entry(review_id=str(i)) for i in range(10)]
        return httpx.Response(200, json={"feed": {"entry": entries}})

    reviews = fetch_reviews(1, settings=_settings(max_reviews=3), client=_client(handler))
    assert len(reviews) == 3


def test_content_is_trimmed_to_the_char_cap():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"feed": {"entry": [_entry(content="x" * 5000)]}})

    reviews = fetch_reviews(1, settings=_settings(review_char_cap=100), client=_client(handler))
    assert len(reviews[0].content) <= 100


def test_timeout_raises_review_fetch_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    with pytest.raises(ReviewFetchTimeout):
        fetch_reviews(1, settings=_settings(), client=_client(handler))


def test_http_error_raises_review_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    with pytest.raises(ReviewFetchError):
        fetch_reviews(1, settings=_settings(), client=_client(handler))


def test_non_json_body_raises_review_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(ReviewFetchError):
        fetch_reviews(1, settings=_settings(), client=_client(handler))


def test_unparseable_date_does_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        entry = _entry()
        entry["updated"] = {"label": "not-a-date"}
        return httpx.Response(200, json={"feed": {"entry": [entry]}})

    reviews = fetch_reviews(1, settings=_settings(), client=_client(handler))
    assert reviews[0].updated is None
