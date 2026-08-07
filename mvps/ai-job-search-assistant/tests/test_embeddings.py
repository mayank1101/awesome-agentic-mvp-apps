import httpx
import pytest

from app.core.exceptions import EmbeddingError
from app.services import embeddings


def _patch_transport(monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]) -> list[int]:
    """Serve canned replies, counting how many requests were actually made."""
    calls: list[int] = []
    queue = list(responses)

    def fake_post(self: httpx.Client, url: str, **kwargs: object) -> httpx.Response:
        calls.append(1)
        return queue.pop(0)

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    monkeypatch.setattr(embeddings, "_RETRY_SECONDS", 0.0)
    return calls


def _vectors(count: int) -> dict:
    return {"data": [{"embedding": [0.1, 0.2]} for _ in range(count)]}


def test_no_key_is_an_error() -> None:
    with pytest.raises(EmbeddingError, match="MISTRAL_API_KEY"):
        embeddings.embed_texts(["anything"])


def test_empty_input_makes_no_call() -> None:
    assert embeddings.embed_texts([]) == []


def test_a_rate_limit_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Observed live: a run makes one embedding call to rank and one per scored
    # job, in a burst. On a free tier that draws a 429 constantly, and a single
    # 429 used to drop the whole run to word overlap.
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    calls = _patch_transport(
        monkeypatch,
        [httpx.Response(429, text="rate limited"), httpx.Response(200, json=_vectors(1))],
    )

    vectors = embeddings.embed_texts(["backend engineer"])

    assert len(calls) == 2
    assert vectors == [[0.1, 0.2]]


def test_a_second_rate_limit_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    calls = _patch_transport(monkeypatch, [httpx.Response(429, text="rate limited")] * 2)

    with pytest.raises(EmbeddingError, match="rate-limiting"):
        embeddings.embed_texts(["backend engineer"])

    assert len(calls) == 2


def test_a_rejected_key_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    calls = _patch_transport(monkeypatch, [httpx.Response(401, text="unauthorized")])

    with pytest.raises(EmbeddingError, match="rejected"):
        embeddings.embed_texts(["backend engineer"])

    assert len(calls) == 1


def test_a_short_reply_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    _patch_transport(monkeypatch, [httpx.Response(200, json=_vectors(1))])

    with pytest.raises(EmbeddingError, match="1 vectors for 2 inputs"):
        embeddings.embed_texts(["one", "two"])
