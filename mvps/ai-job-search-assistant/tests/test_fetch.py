from typing import Any

import pytest

from app.core.exceptions import SearchQuotaExhausted
from app.services import fetch

_LONG = "We are hiring a backend engineer. " * 40


def _patch_extract(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    queue = list(replies)

    def fake_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append({"path": path, **payload})
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(fetch, "post_json", fake_post)
    return sent


def test_no_urls_makes_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _patch_extract(monkeypatch, [])
    assert fetch.fetch_postings([]) == {}
    assert sent == []


def test_full_text_is_normalised_and_marked_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://boards.greenhouse.io/acme/jobs/1"
    _patch_extract(monkeypatch, [{"results": [{"url": url, "raw_content": _LONG}]}])

    postings = fetch.fetch_postings([url])

    assert postings[url].ok is True
    assert "backend engineer" in postings[url].text.lower()


def test_a_javascript_shell_is_not_scoreable(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://boards.greenhouse.io/acme/jobs/2"
    _patch_extract(monkeypatch, [{"results": [{"url": url, "raw_content": "Loading…"}]}])

    posting = fetch.fetch_postings([url])[url]

    assert posting.ok is False
    assert "JavaScript" in posting.reason


def test_failed_results_carry_their_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://boards.greenhouse.io/acme/jobs/3"
    _patch_extract(
        monkeypatch, [{"results": [], "failed_results": [{"url": url, "error": "404 not found"}]}]
    )

    posting = fetch.fetch_postings([url])[url]

    assert posting.ok is False
    assert posting.reason == "404 not found"


def test_a_redirected_url_is_matched_back_to_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = "https://boards.greenhouse.io/acme/jobs/4"
    _patch_extract(monkeypatch, [{"results": [{"url": f"{asked}/", "raw_content": _LONG}]}])

    postings = fetch.fetch_postings([asked])

    assert postings[asked].ok is True


def test_a_provider_failure_degrades_every_url_in_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls = ["https://boards.greenhouse.io/acme/jobs/5", "https://jobs.lever.co/acme/abcdef12"]
    _patch_extract(monkeypatch, [SearchQuotaExhausted("no credits")])

    postings = fetch.fetch_postings(urls)

    assert set(postings) == set(urls)
    assert all(posting.ok is False for posting in postings.values())


def test_every_requested_url_gets_an_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = ["https://boards.greenhouse.io/acme/jobs/6", "https://boards.greenhouse.io/acme/jobs/7"]
    _patch_extract(monkeypatch, [{"results": [{"url": asked[0], "raw_content": _LONG}]}])

    postings = fetch.fetch_postings(asked)

    assert set(postings) == set(asked)
    assert postings[asked[1]].ok is False


def test_urls_are_batched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACT_BATCH_SIZE", "2")
    urls = [f"https://boards.greenhouse.io/acme/jobs/{index}" for index in range(5)]
    sent = _patch_extract(monkeypatch, [{"results": []}] * 3)

    fetch.fetch_postings(urls)

    assert [len(call["urls"]) for call in sent] == [2, 2, 1]
