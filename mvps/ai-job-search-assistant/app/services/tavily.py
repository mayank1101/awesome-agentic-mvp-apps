"""Transport for the one service this app uses to reach the web.

Tavily, over plain `httpx`. Two endpoints, two request shapes, so the SDK would
be a dependency that saves nothing -- the same call this repo makes to Mistral's
embedding endpoint for the same reason.

Everything network-shaped that can go wrong is classified here, once, into this
app's own error vocabulary. The classification matters more than it looks:
Tavily returns HTTP 429 both for "you are going too fast" and for "your monthly
credits are gone", and those need opposite handling -- the first clears in
seconds, the second does not clear at all this month. A retry loop that cannot
tell them apart spends a minute proving the obvious and then reports the wrong
thing to the user.
"""

from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import SearchAuthError, SearchError, SearchQuotaExhausted
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Body markers that mean the credit balance is gone rather than the request
#: rate being too high. Checked before the status code is trusted, because both
#: arrive as 429.
_EXHAUSTED_MARKERS = (
    "usage limit",
    "credit",
    "quota",
    "exceeded your",
    "plan limit",
    "upgrade",
)


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST a JSON body to a Tavily endpoint and return the decoded reply.

    Args:
        path: Endpoint path, e.g. ``/search``.
        payload: The request body. The API key is added here rather than by
            callers, so no caller has to hold it and no caller can log it.

    Returns:
        The decoded JSON object.

    Raises:
        SearchAuthError: The key was missing or rejected.
        SearchQuotaExhausted: The account is out of credits or rate-limited in
            a way that will not clear quickly.
        SearchError: Anything else -- network failure, timeout, 5xx, or a reply
            that was not a JSON object.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        raise SearchAuthError("TAVILY_API_KEY is not set.")

    url = f"{settings.tavily_base_url.rstrip('/')}{path}"

    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.tavily_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.tavily_timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        raise SearchError(
            f"The search service did not respond within {settings.tavily_timeout_seconds:.0f}s."
        ) from exc
    except httpx.HTTPError as exc:
        raise SearchError(f"Could not reach the search service ({type(exc).__name__}).") from exc

    _raise_for_status(response)

    try:
        body = response.json()
    except ValueError as exc:
        raise SearchError("The search service returned a reply that was not JSON.") from exc

    if not isinstance(body, dict):
        raise SearchError("The search service returned JSON that was not an object.")
    return body


def _raise_for_status(response: httpx.Response) -> None:
    """Turn a non-2xx reply into the right exception, reading the body first."""
    if response.status_code < 400:
        return

    body = _safe_body(response)
    lowered = body.lower()

    if response.status_code in (401, 403):
        raise SearchAuthError(
            "The Tavily API key was rejected. Check TAVILY_API_KEY at https://tavily.com."
        )

    if response.status_code in (402, 429) or any(
        marker in lowered for marker in _EXHAUSTED_MARKERS
    ):
        raise SearchQuotaExhausted(
            "This Tavily key has no search credits left, or is being rate-limited. "
            "Free-tier credits are a monthly pool, so this does not clear in a minute."
        )

    if response.status_code >= 500:
        raise SearchError(f"The search service is failing (HTTP {response.status_code}).")

    raise SearchError(f"The search service refused the request (HTTP {response.status_code}).")


def _safe_body(response: httpx.Response) -> str:
    """Read a response body for classification without ever raising.

    Capped: an error body is occasionally an entire HTML page, and the only part
    that carries a marker is the beginning.
    """
    try:
        return response.text[:2000]
    except Exception:  # noqa: BLE001 - a body that cannot be read is not a reason to crash
        return ""
