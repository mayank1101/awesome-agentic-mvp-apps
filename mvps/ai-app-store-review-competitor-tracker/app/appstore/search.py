"""App resolution: turning whatever the user typed into one `trackId`.

Three input shapes, one exit:

* A **numeric App Store id** (`1232780281`) needs no network call at all until
  the lookup that confirms it exists.
* A **full App Store URL** (`https://apps.apple.com/us/app/notion/id1232780281`)
  has the id embedded in it; parsed locally, then confirmed the same way.
* A **free-text name** goes to Apple's search endpoint and comes back as a list
  of candidates for the caller to disambiguate -- there is no heuristic
  ranking here, because Apple's own relevance ranking is the correct one to
  defer to, and picking silently would risk exactly the wrong-app failure this
  repo's other apps guard against for companies.

Both endpoints used here (`/search` and `/lookup`) are the documented,
stable half of Apple's public iTunes API -- confirmed working as documented by
direct testing (see docs/01-prd.md §7), unlike the review feed in
`reviews.py`.
"""

import re
from urllib.parse import urlparse

import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import AppNotFound, InvalidAppReference
from app.core.logging import get_logger
from app.models.schemas import AppCandidate, AppIdentity, Platform

logger = get_logger(__name__)

_SEARCH_URL = "https://itunes.apple.com/search"
_LOOKUP_URL = "https://itunes.apple.com/lookup"

#: Matches the numeric id in an App Store URL path segment, e.g. `/id1232780281`.
_URL_ID_PATTERN = re.compile(r"/id(\d+)")

#: A bare numeric id typed directly, allowing surrounding whitespace.
_BARE_ID_PATTERN = re.compile(r"^\s*(\d{6,12})\s*$")

_APP_STORE_HOSTS = frozenset({"apps.apple.com", "itunes.apple.com"})


def parse_track_id(raw: str) -> int | None:
    """Extract a `trackId` from a bare id or an App Store URL.

    Args:
        raw: Whatever the user typed.

    Returns:
        The parsed id, or `None` if `raw` looks like a name search instead.

    Raises:
        InvalidAppReference: `raw` looks like an App Store URL but has no
            parseable id in it -- distinct from "not a URL at all", so the
            caller can say what actually went wrong.
    """
    bare = _BARE_ID_PATTERN.match(raw)
    if bare:
        return int(bare.group(1))

    parsed = urlparse(raw if "//" in raw else f"//{raw}")
    if parsed.netloc.lower().removeprefix("www.") not in _APP_STORE_HOSTS:
        return None

    match = _URL_ID_PATTERN.search(parsed.path)
    if not match:
        raise InvalidAppReference(
            "that looks like an App Store link, but no app id could be found in it"
        )
    return int(match.group(1))


def _to_candidate(raw: dict) -> AppCandidate | None:
    """Convert one iTunes API result into a candidate, or `None` if unusable."""
    track_id = raw.get("trackId")
    url = raw.get("trackViewUrl")
    name = raw.get("trackName")
    if not (track_id and url and name):
        return None
    return AppCandidate(
        platform=Platform.IOS,
        track_id=track_id,
        track_name=name,
        artist_name=raw.get("artistName", "Unknown developer"),
        primary_genre_name=raw.get("primaryGenreName"),
        artwork_url=raw.get("artworkUrl100") or raw.get("artworkUrl60"),
        app_store_url=url,
        average_user_rating=raw.get("averageUserRating"),
        user_rating_count=raw.get("userRatingCount", 0),
    )


def _to_identity(raw: dict) -> AppIdentity:
    """Convert one iTunes lookup result into an :class:`AppIdentity`."""
    return AppIdentity(
        platform=Platform.IOS,
        track_id=raw["trackId"],
        track_name=raw["trackName"],
        artist_name=raw.get("artistName", "Unknown developer"),
        primary_genre_name=raw.get("primaryGenreName"),
        artwork_url=raw.get("artworkUrl100") or raw.get("artworkUrl60"),
        app_store_url=raw["trackViewUrl"],
        published_average_rating=raw.get("averageUserRating"),
        published_rating_count=raw.get("userRatingCount", 0),
    )


def search_apps(
    term: str,
    *,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
    limit: int = 5,
) -> list[AppCandidate]:
    """Search Apple's catalog by name.

    Args:
        term: A free-text app name.
        settings: Runtime configuration; defaults to the process settings.
        client: Injected HTTP client, for tests.
        limit: Maximum candidates to return.

    Returns:
        Candidates in Apple's own relevance order, possibly empty.

    Raises:
        AppNotFound: The request itself failed (network, non-2xx). An empty
            *result set* is not this -- that is a normal empty list, and the
            caller decides what to do with zero candidates.
    """
    settings = settings or get_settings()
    params = {
        "term": term,
        "country": settings.appstore_country,
        "entity": "software",
        "limit": limit,
    }
    try:
        owns_client = client is None
        client = client or httpx.Client(timeout=settings.request_timeout_seconds)
        try:
            response = client.get(_SEARCH_URL, params=params)
            response.raise_for_status()
        finally:
            if owns_client:
                client.close()
    except httpx.HTTPError as exc:
        raise AppNotFound(f"the App Store search request failed: {exc}", query=term) from exc

    data = response.json()
    candidates = [_to_candidate(raw) for raw in data.get("results", [])]
    return [c for c in candidates if c is not None]


def lookup_app(
    track_id: int,
    *,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> AppIdentity:
    """Confirm an app id and fetch its published identity.

    Args:
        track_id: The App Store id to confirm.
        settings: Runtime configuration; defaults to the process settings.
        client: Injected HTTP client, for tests.

    Returns:
        The app's identity.

    Raises:
        AppNotFound: The id does not resolve to any app, or the request failed.
    """
    settings = settings or get_settings()
    params = {"id": track_id, "country": settings.appstore_country}
    try:
        owns_client = client is None
        client = client or httpx.Client(timeout=settings.request_timeout_seconds)
        try:
            response = client.get(_LOOKUP_URL, params=params)
            response.raise_for_status()
        finally:
            if owns_client:
                client.close()
    except httpx.HTTPError as exc:
        raise AppNotFound(
            f"the App Store lookup request failed: {exc}", query=str(track_id)
        ) from exc

    results = response.json().get("results", [])
    if not results:
        raise AppNotFound(f"no app found for id {track_id}", query=str(track_id))

    logger.info("resolved id %d to %s", track_id, results[0].get("trackName"))
    return _to_identity(results[0])


def resolve(
    raw: str,
    *,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> AppIdentity | list[AppCandidate]:
    """Resolve user input to one app, or a list to disambiguate.

    Args:
        raw: A name, an App Store URL, or a numeric id.
        settings: Runtime configuration; defaults to the process settings.
        client: Injected HTTP client, for tests.

    Returns:
        An :class:`AppIdentity` when the input was already unambiguous (an id
        or a URL), or a list of :class:`AppCandidate` for the caller to
        present when it was a name search.

    Raises:
        AppNotFound: A name search returned nothing, or an id/URL did not
            resolve to a real app.
        InvalidAppReference: The input looked like an App Store URL but had no
            parseable id.
    """
    track_id = parse_track_id(raw)
    if track_id is not None:
        return lookup_app(track_id, settings=settings, client=client)

    candidates = search_apps(raw, settings=settings, client=client)
    if not candidates:
        raise AppNotFound(f"no apps found matching “{raw}”", query=raw)
    return candidates
