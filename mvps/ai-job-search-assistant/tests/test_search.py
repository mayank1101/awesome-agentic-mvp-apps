from typing import Any

import pytest

from app.core.exceptions import SearchError
from app.services import search


def _result(url: str, title: str = "Backend Engineer", **extra: Any) -> dict[str, Any]:
    return {"url": url, "title": title, "content": "snippet", "score": 0.5, **extra}


def _patch_search(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> list[dict[str, Any]]:
    """Serve one canned reply per query, recording the payloads that were sent."""
    sent: list[dict[str, Any]] = []
    queue = list(replies)

    def fake_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append({"path": path, **payload})
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(search, "post_json", fake_post)
    return sent


def test_whitelist_is_sent_to_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _patch_search(monkeypatch, [{"results": []}])

    search.search_jobs(["backend engineer"], ["jobs.lever.co"], recency_days=30)

    assert sent[0]["include_domains"] == ["jobs.lever.co"]
    assert sent[0]["time_range"] == "month"


def test_empty_whitelist_is_refused() -> None:
    # An empty include_domains searches the whole web, which is the one thing
    # this app promises not to do.
    with pytest.raises(SearchError):
        search.search_jobs(["backend engineer"], [])


@pytest.mark.parametrize(
    ("days", "expected"),
    [(None, None), (0, None), (1, "day"), (7, "week"), (30, "month"), (45, "year"), (900, None)],
)
def test_recency_maps_to_a_provider_window(days: int | None, expected: str | None) -> None:
    assert search.time_range_for(days) == expected


def test_off_whitelist_results_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(
        monkeypatch,
        [
            {
                "results": [
                    _result("https://jobs.lever.co/acme/2f1c9a44-1111-2222-3333-444455556666"),
                    _result("https://scam.example/jobs/12345"),
                ]
            }
        ],
    )

    hits, raw = search.search_jobs(["q"], ["jobs.lever.co"])

    assert raw == 2
    assert [hit.domain for hit in hits] == ["jobs.lever.co"]


def test_subdomains_of_a_whitelisted_host_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(
        monkeypatch,
        [{"results": [_result("https://boards.greenhouse.io/acme/jobs/123456")]}],
    )

    hits, _ = search.search_jobs(["q"], ["greenhouse.io"])

    assert len(hits) == 1


def test_listing_pages_are_filtered_out(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(
        monkeypatch,
        [
            {
                "results": [
                    _result("https://boards.greenhouse.io/acme", title="Jobs at Acme"),
                    _result("https://boards.greenhouse.io/acme/jobs/123456"),
                ]
            }
        ],
    )

    hits, raw = search.search_jobs(["q"], ["boards.greenhouse.io"])

    assert raw == 2
    assert len(hits) == 1


def test_the_same_job_from_two_queries_appears_once(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://boards.greenhouse.io/acme/jobs/123456"
    _patch_search(
        monkeypatch,
        [
            {"results": [_result(url)]},
            {"results": [_result(f"{url}?utm_source=alerts")]},
        ],
    )

    hits, raw = search.search_jobs(["q1", "q2"], ["boards.greenhouse.io"])

    assert raw == 2
    assert len(hits) == 1
    assert hits[0].query == "q1"


def test_the_same_role_on_two_boards_appears_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(
        monkeypatch,
        [
            {
                "results": [
                    _result(
                        "https://boards.greenhouse.io/acme/jobs/123456", "Senior Backend Engineer"
                    ),
                    _result(
                        "https://www.linkedin.com/jobs/view/9988776655", "Senior Backend Engineer"
                    ),
                ]
            }
        ],
    )

    hits, _ = search.search_jobs(["q"], ["boards.greenhouse.io", "linkedin.com"])

    assert len(hits) == 1
    assert hits[0].domain == "boards.greenhouse.io"


def test_one_failing_query_does_not_lose_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(
        monkeypatch,
        [
            SearchError("provider hiccup"),
            {"results": [_result("https://boards.greenhouse.io/acme/jobs/123456")]},
        ],
    )

    hits, _ = search.search_jobs(["q1", "q2"], ["boards.greenhouse.io"])

    assert len(hits) == 1


def test_every_query_failing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(monkeypatch, [SearchError("down"), SearchError("down")])

    with pytest.raises(SearchError):
        search.search_jobs(["q1", "q2"], ["boards.greenhouse.io"])


def test_result_cap_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_RESULTS_TOTAL", "2")
    _patch_search(
        monkeypatch,
        [
            {
                "results": [
                    _result(f"https://boards.greenhouse.io/acme/jobs/{index}00000", f"Role {index}")
                    for index in range(5)
                ]
            }
        ],
    )

    hits, _ = search.search_jobs(["q"], ["boards.greenhouse.io"])

    assert len(hits) == 2


def test_hits_carry_canonical_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(
        monkeypatch,
        [{"results": [_result("https://boards.greenhouse.io/acme/jobs/123456/?utm_source=x")]}],
    )

    hits, _ = search.search_jobs(["q"], ["boards.greenhouse.io"])

    assert hits[0].url == "https://boards.greenhouse.io/acme/jobs/123456"


def test_a_reply_without_a_results_array_fails_that_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_search(monkeypatch, [{"unexpected": True}])

    with pytest.raises(SearchError):
        search.search_jobs(["q"], ["boards.greenhouse.io"])
