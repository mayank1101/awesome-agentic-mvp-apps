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

import httpx

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)

_ENDPOINT = "https://api.mistral.ai/v1/embeddings"

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
    """Embed one batch, translating every failure into :class:`EmbeddingError`."""
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
