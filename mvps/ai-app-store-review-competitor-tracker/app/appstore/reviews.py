"""The review fetch: one HTTP request, no pagination.

Apple's customer-reviews feed is unofficial and undocumented. Direct testing
against several apps on 2026-08-07 (see docs/01-prd.md §7) established two
things with different confidence levels:

* **Deterministic and reproduced repeatedly:** the **bare URL** --
  ``https://itunes.apple.com/rss/customerreviews/id={id}/sortby=mostrecent/json``,
  no country prefix, no ``page=`` segment -- is the only request shape that
  ever returns real data. Adding **any** ``page=N`` path segment (including
  the theoretically inert ``page=1``) or a non-US `cc` value (as a query
  param or a path prefix) makes the feed return an empty shell instead of an
  error, every time it was tried.
* **Not deterministic, and not fully explained:** even the bare URL against a
  known-good id is *unreliable* under sustained request volume from one
  source. Early, sparsely-spaced requests in this investigation succeeded
  consistently; the same exact request, hit repeatedly while narrowing down
  the finding above, started returning empty shells for apps (and even the
  `cc` presence/absence) that had just worked. A User-Agent header and the
  `cc` param were both tested as possible explanations and **neither held up
  under a controlled retest** -- the same "known good" request failed anyway
  minutes later. The pattern is consistent with informal, IP-based rate
  limiting or bot-suspicion on Apple's side that this app has no way to
  detect or negotiate with, rather than anything about the request itself.

The practical conclusion is the same either way: this module cannot
distinguish "this app genuinely has zero reviews in the sample" from
"the feed is temporarily unwilling to answer," so it does not try to. It sends
a browser-shaped `User-Agent` (harmless, and was a plausible fix before the
retest above ruled it out as *the* fix) and retries once after a short pause,
then accepts whatever it gets as the answer -- see `render_report` in
`app/services/renderer.py` for how an empty sample is shown to the user as a
plain fact, never as an error page.

This module therefore makes at most two requests per call, both to the bare
URL, and does not attempt to paginate or switch storefronts -- see the module
docstring in `docs/01-prd.md` for why the latter is a documented product
decision and not a gap to be worked around later with more retries or
scraping.
"""

import time
from datetime import datetime

import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import ReviewFetchError, ReviewFetchTimeout
from app.core.logging import get_logger
from app.models.schemas import Review

logger = get_logger(__name__)

_REVIEWS_URL_TEMPLATE = "https://itunes.apple.com/rss/customerreviews/id={track_id}/sortby=mostrecent/json"

#: A plausible fix for the empty-shell problem, tested and kept even though a
#: controlled retest showed it is not sufficient by itself -- see the module
#: docstring. Costs nothing to send and may still help against whatever part
#: of Apple's edge behavior is header-sensitive.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
}

#: How long to wait before the one retry of an empty-looking response, for the
#: residual edge-inconsistency the header above does not fully eliminate.
_EMPTY_RETRY_DELAY_SECONDS = 1.5


def _parse_entry(entry: dict, *, char_cap: int) -> Review | None:
    """Convert one feed entry into a :class:`Review`, or `None` if unusable.

    A handful of feeds have been seen to include a non-review entry (feed
    metadata masquerading as the first array item) alongside genuine reviews;
    anything missing a rating is filtered out rather than trusted.
    """
    rating_label = entry.get("im:rating", {}).get("label")
    review_id = entry.get("id", {}).get("label")
    if not rating_label or not review_id:
        return None

    updated_label = entry.get("updated", {}).get("label")
    updated = None
    if updated_label:
        try:
            updated = datetime.fromisoformat(updated_label)
        except ValueError:
            updated = None

    content = entry.get("content", {}).get("label", "")
    return Review(
        id=str(review_id),
        rating=int(rating_label),
        title=entry.get("title", {}).get("label", "").strip() or "(no title)",
        content=content[:char_cap].strip(),
        author=entry.get("author", {}).get("name", {}).get("label", "Anonymous"),
        version=entry.get("im:version", {}).get("label"),
        updated=updated,
    )


def _request_entries(track_id: int, *, settings: Settings, client: httpx.Client) -> list[dict]:
    """Make one request and return the raw feed entries (possibly empty).

    Raises:
        ReviewFetchTimeout: The request exceeded its timeout.
        ReviewFetchError: The request failed for any other reason (network,
            non-2xx, unparseable body).
    """
    url = _REVIEWS_URL_TEMPLATE.format(track_id=track_id)
    params = {"cc": settings.appstore_country} if settings.appstore_country else {}

    try:
        response = client.get(url, params=params, headers=_BROWSER_HEADERS)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise ReviewFetchTimeout(
            f"the review feed timed out after {settings.request_timeout_seconds}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise ReviewFetchError(f"the review feed request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise ReviewFetchError("the review feed returned a response that was not JSON") from exc

    entries = data.get("feed", {}).get("entry", [])
    if isinstance(entries, dict):
        # A feed with exactly one review comes back as a single object rather
        # than a one-item array -- Apple's Atom-to-JSON conversion does this
        # for every singular child element, not only this one.
        entries = [entries]
    return entries


def fetch_reviews(
    track_id: int,
    *,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> list[Review]:
    """Fetch the most recent reviews for one app.

    Sends a browser-shaped `User-Agent` and retries once, briefly, if the
    first response is empty -- neither is a confirmed fix for the
    unreliability described in the module docstring, but both are cheap and
    cannot make it worse. A genuinely review-less app still ends up with an
    empty list; a temporarily uncooperative feed does too, and this module has
    no way to tell the two apart from the response alone.

    Args:
        track_id: The App Store id.
        settings: Runtime configuration; defaults to the process settings.
        client: Injected HTTP client, for tests.

    Returns:
        Reviews, most recent first, capped at `settings.max_reviews`.
        Possibly empty -- an app with no reviews in the sample is a legitimate
        outcome, not an error.

    Raises:
        ReviewFetchTimeout: A request exceeded its timeout.
        ReviewFetchError: A request failed for any other reason (network,
            non-2xx, unparseable body).
    """
    settings = settings or get_settings()

    owns_client = client is None
    client = client or httpx.Client(timeout=settings.request_timeout_seconds)
    try:
        entries = _request_entries(track_id, settings=settings, client=client)
        if not entries:
            logger.info("empty response for app %d; retrying once", track_id)
            time.sleep(_EMPTY_RETRY_DELAY_SECONDS)
            entries = _request_entries(track_id, settings=settings, client=client)
    finally:
        if owns_client:
            client.close()

    reviews = [_parse_entry(entry, char_cap=settings.review_char_cap) for entry in entries]
    kept = [r for r in reviews if r is not None][: settings.max_reviews]

    logger.info("fetched %d review(s) for app %d", len(kept), track_id)
    return kept
