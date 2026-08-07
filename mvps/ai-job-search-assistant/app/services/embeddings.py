"""Embeddings via Mistral's hosted `mistral-embed`.

Hosted rather than local, for one reason: the deployment target is a ~1GB
Streamlit Community Cloud container with no GPU, and a local sentence-transformer
plus `torch` does not fit it -- nor does it coexist peacefully with other native
wheels on macOS. A hosted embedding endpoint costs one HTTP round trip and keeps
the image at "Python plus pure-Python wheels".

Called through `httpx` rather than the `mistralai` SDK. One endpoint, one request
shape, and the SDK's transitive dependency set is larger than the app's own.

**Embeddings are optional here.** Without a Mistral key the matcher falls back to
lexical overlap (:mod:`app.services.matching`), the report says which mode
produced it, and the app still works. That is a deliberate trade: an app whose
demo dies because a second free-tier key expired is worse than an app whose
similarity numbers get coarser.
"""

import time

import httpx

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)

_ENDPOINT = "https://api.mistral.ai/v1/embeddings"

#: Attempts for one batch. Two, not more: a second 429 means the key is
#: saturated rather than momentarily busy, and the lexical path is a better
#: answer than a longer wait.
_RATE_LIMIT_ATTEMPTS = 2

#: Wait between those attempts. Sized to Mistral's free-tier window, which is
#: per-second rather than per-minute.
_RETRY_SECONDS = 1.5

#: Mistral rejects empty strings in a batch, and a resume genuinely can produce
#: one after normalisation, so blanks are replaced rather than sent.
_PLACEHOLDER = "(blank)"


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with `mistral-embed`.

    Args:
        texts: The strings to embed, in order. May be empty.

    Returns:
        One vector per input, in the same order.

    Raises:
        EmbeddingError: No key is configured, the provider refused, or the reply
            did not contain one vector per input. Callers treat this as "fall
            back to lexical matching", never as a fatal error.
    """
    if not texts:
        return []

    settings = get_settings()
    if not settings.mistral_api_key:
        raise EmbeddingError("MISTRAL_API_KEY is not set.")

    payload_texts = [text if text.strip() else _PLACEHOLDER for text in texts]
    vectors: list[list[float]] = []

    with httpx.Client(timeout=settings.embedding_timeout_seconds) as client:
        for start in range(0, len(payload_texts), settings.embedding_batch_size):
            batch = payload_texts[start : start + settings.embedding_batch_size]
            vectors.extend(_embed_batch(client, batch))

    if len(vectors) != len(texts):
        raise EmbeddingError(f"Mistral returned {len(vectors)} vectors for {len(texts)} inputs.")
    return vectors


def _embed_batch(client: httpx.Client, batch: list[str]) -> list[list[float]]:
    """Embed one batch, retrying a rate limit once before degrading the run.

    The retry is here because of what a live run of this app looks like from
    Mistral's side: one call to rank the search results, then one per job as each
    is scored, arriving in a burst. On the free tier that is enough to draw a 429
    on nearly every call, and a single 429 drops the *whole run* to word overlap
    -- which is a much worse result than waiting a second and a half.

    One retry, not a loop: past that, the key is genuinely saturated and the
    lexical path is the honest answer rather than a spinner.
    """
    for attempt in range(_RATE_LIMIT_ATTEMPTS):
        try:
            return _post_batch(client, batch)
        except EmbeddingError as exc:
            if "rate-limiting" not in str(exc) or attempt == _RATE_LIMIT_ATTEMPTS - 1:
                raise
            logger.info("Embedding call rate-limited, retrying in %.1fs", _RETRY_SECONDS)
            time.sleep(_RETRY_SECONDS)

    raise EmbeddingError("Mistral is rate-limiting this key.")


def _post_batch(client: httpx.Client, batch: list[str]) -> list[list[float]]:
    """Make one embedding request, translating every failure into an error."""
    settings = get_settings()
    try:
        response = client.post(
            _ENDPOINT,
            headers={
                "Authorization": f"Bearer {settings.mistral_api_key}",
                "Content-Type": "application/json",
            },
            json={"model": settings.embedding_model, "input": batch},
        )
    except httpx.HTTPError as exc:
        raise EmbeddingError(
            f"Could not reach the embedding service: {type(exc).__name__}"
        ) from exc

    if response.status_code == 401:
        raise EmbeddingError("The Mistral API key was rejected.")
    if response.status_code == 429:
        raise EmbeddingError("Mistral is rate-limiting this key.")
    if response.status_code >= 400:
        raise EmbeddingError(f"The embedding service returned HTTP {response.status_code}.")

    try:
        data = response.json()["data"]
        return [item["embedding"] for item in data]
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingError("The embedding service returned an unexpected shape.") from exc
