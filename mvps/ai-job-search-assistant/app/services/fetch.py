"""Fetching the full text of the postings that are about to be scored.

Only the jobs selected for deep scoring get here, and that ordering is the whole
cost model of the app: extraction is charged per URL, so fetching everything the
search returned would spend most of a run's budget on pages nobody reads.

**Nothing in this module can fail a run.** Every failure -- a provider error, a
JavaScript shell with no text, a login wall, an expired listing, a page of pure
navigation -- produces a :class:`PostingText` with ``ok=False`` and a reason.
The job then keeps its snippet-based score and the row says which basis produced
it. That is the honest outcome: the job exists, the link works, and the app could
not read the page. Dropping it would hide a job the user might want; scoring it
deeply anyway would put a confident number on 200 characters of navigation.
"""

from typing import Any

from app.core.config import get_settings
from app.core.exceptions import JobSearchError
from app.core.logging import get_logger
from app.models.schemas import PostingText
from app.services.pdf_extract import prepare_posting_text
from app.services.tavily import post_json

logger = get_logger(__name__)


def fetch_postings(urls: list[str]) -> dict[str, PostingText]:
    """Fetch the text of several postings, batched.

    Args:
        urls: Canonical posting URLs, in the order they were ranked.

    Returns:
        One entry per requested URL, keyed by the URL as passed in. Every URL
        gets an entry -- a missing key would make callers write the same
        "did the fetch happen?" branch over and over.
    """
    if not urls:
        return {}

    settings = get_settings()
    results: dict[str, PostingText] = {}

    for start in range(0, len(urls), settings.extract_batch_size):
        batch = urls[start : start + settings.extract_batch_size]
        try:
            results.update(_fetch_batch(batch))
        except JobSearchError as exc:
            # A failed batch is a batch of shallow rows, not a failed run.
            logger.warning("Extract batch of %d failed: %s", len(batch), exc)
            for url in batch:
                results[url] = PostingText(
                    url=url, ok=False, reason="The page could not be fetched."
                )

    for url in urls:
        results.setdefault(
            url, PostingText(url=url, ok=False, reason="The page was not returned by the fetcher.")
        )

    readable = sum(1 for posting in results.values() if posting.ok)
    logger.info("Fetched %d/%d postings with usable text", readable, len(urls))
    return results


def _fetch_batch(urls: list[str]) -> dict[str, PostingText]:
    """Fetch one batch and normalise every entry into a :class:`PostingText`."""
    body = post_json(
        "/extract",
        {"urls": urls, "extract_depth": "basic"},
    )

    postings: dict[str, PostingText] = {}

    for entry in _as_dicts(body.get("results")):
        url = str(entry.get("url") or "").strip()
        raw = entry.get("raw_content") or entry.get("content") or ""
        key = _match_requested(url, urls) or url
        postings[key] = _to_posting(key, str(raw))

    for entry in _as_dicts(body.get("failed_results")):
        url = str(entry.get("url") or "").strip()
        key = _match_requested(url, urls) or url
        if key in postings:
            continue
        reason = str(entry.get("error") or "").strip() or "The page could not be read."
        postings[key] = PostingText(url=key, ok=False, reason=reason)

    return postings


def _to_posting(url: str, raw: str) -> PostingText:
    """Normalise, cap, and judge whether the text is worth scoring against."""
    text, truncated = prepare_posting_text(raw)
    minimum = get_settings().min_posting_chars

    if len(text) < minimum:
        return PostingText(
            url=url,
            text=text,
            truncated=truncated,
            ok=False,
            reason=(
                f"Only {len(text)} characters of text came back, which usually means the "
                "posting is rendered by JavaScript, behind a login, or already closed."
            ),
        )

    return PostingText(url=url, text=text, truncated=truncated, ok=True)


def _match_requested(returned: str, requested: list[str]) -> str | None:
    """Map a URL the provider echoed back onto the one that was asked for.

    Extraction follows redirects and normalises as it goes, so the URL in the
    reply is often not byte-identical to the one sent -- a trailing slash, a
    resolved shortlink, an added locale segment. Keying results on the returned
    string would then lose the association with the job that asked for it.
    """
    if returned in requested:
        return returned
    stripped = returned.rstrip("/")
    for candidate in requested:
        if candidate.rstrip("/") == stripped:
            return candidate
    return None


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    """Return `value` as a list of dicts, tolerating anything else."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
