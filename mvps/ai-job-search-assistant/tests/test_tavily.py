import httpx
import pytest

from app.core.exceptions import SearchAuthError, SearchError, SearchQuotaExhausted
from app.services import tavily


def _respond(monkeypatch: pytest.MonkeyPatch, response: httpx.Response) -> dict:
    """Patch the transport and capture the request that was made."""
    captured: dict = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return response

    monkeypatch.setattr(httpx, "post", fake_post)
    return captured


def test_missing_key_is_an_auth_error() -> None:
    with pytest.raises(SearchAuthError):
        tavily.post_json("/search", {"query": "x"})


def test_key_is_sent_as_a_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    captured = _respond(monkeypatch, httpx.Response(200, json={"results": []}))

    tavily.post_json("/search", {"query": "backend engineer"})

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["headers"]["Authorization"] == "Bearer tvly-test-key"
    assert captured["json"] == {"query": "backend engineer"}


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_is_classified(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    _respond(monkeypatch, httpx.Response(status, text="forbidden"))

    with pytest.raises(SearchAuthError):
        tavily.post_json("/search", {})


def test_rate_limit_and_credit_exhaustion_are_the_same_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    _respond(monkeypatch, httpx.Response(429, text="Too many requests"))

    with pytest.raises(SearchQuotaExhausted):
        tavily.post_json("/search", {})


def test_credit_message_on_a_400_is_still_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    # The status code alone is not enough: a plan limit arrives as a 4xx with the
    # explanation only in the body.
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    _respond(monkeypatch, httpx.Response(400, text="You have exceeded your plan limit"))

    with pytest.raises(SearchQuotaExhausted):
        tavily.post_json("/search", {})


def test_server_error_is_a_plain_search_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    _respond(monkeypatch, httpx.Response(503, text="unavailable"))

    with pytest.raises(SearchError):
        tavily.post_json("/search", {})


def test_non_json_reply_is_a_search_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    _respond(monkeypatch, httpx.Response(200, text="<html>hello</html>"))

    with pytest.raises(SearchError):
        tavily.post_json("/search", {})


def test_timeout_is_reported_with_the_configured_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(SearchError, match="did not respond"):
        tavily.post_json("/search", {})
