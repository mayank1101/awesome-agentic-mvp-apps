"""App resolution for Google Play, mirroring `app/appstore/search.py`.

There is no official free API for looking up a competitor's Play Store
listing -- the Play Developer API only covers apps you own. This module wraps
`google-play-scraper`, an unofficial package that scrapes Play's own public
app and search pages, confirmed by direct testing (docs/01-prd.md §7-addendum)
to work reliably for both the US and India storefronts with no rate-limiting
observed, unlike the iOS review feed.

Three input shapes, same as iOS: a bare **package name** (`com.spotify.music`
-- reverse-DNS shaped, so it is detected rather than guessed), a **Play Store
URL** (`https://play.google.com/store/apps/details?id=...`), or a **free-text
name**, which returns candidates to disambiguate.
"""

import re
from urllib.parse import parse_qs, urlparse

from google_play_scraper import app as gp_app
from google_play_scraper import search as gp_search
from google_play_scraper.exceptions import GooglePlayScraperException, NotFoundError

from app.core.config import Settings, get_settings
from app.core.exceptions import AppNotFound, InvalidAppReference
from app.core.logging import get_logger
from app.models.schemas import AppCandidate, AppIdentity, Platform

logger = get_logger(__name__)

#: A Play Store package name: reverse-DNS, lowercase, digits, underscores.
#: `com.spotify.music` matches; `Spotify` or `spotify music` does not, so a
#: free-text name never gets misread as a package id.
_PACKAGE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+){2,}$")

_PLAY_HOSTS = frozenset({"play.google.com"})


def parse_package_name(raw: str) -> str | None:
    """Extract a package name from a bare id or a Play Store URL.

    Args:
        raw: Whatever the user typed.

    Returns:
        The package name, or `None` if `raw` looks like a name search instead.

    Raises:
        InvalidAppReference: `raw` looks like a Play Store URL but carries no
            `id` query parameter.
    """
    stripped = raw.strip()
    if _PACKAGE_NAME_PATTERN.match(stripped):
        return stripped

    parsed = urlparse(stripped if "//" in stripped else f"//{stripped}")
    if parsed.netloc.lower().removeprefix("www.") not in _PLAY_HOSTS:
        return None

    package = parse_qs(parsed.query).get("id", [None])[0]
    if not package:
        raise InvalidAppReference(
            "that looks like a Play Store link, but no app id could be found in it"
        )
    return package


def _to_candidate(raw: dict) -> AppCandidate | None:
    """Convert one Play search result into a candidate, or `None` if unusable.

    `appId` has been observed `None` for a subset of results -- an unofficial
    scraper's page-structure parsing occasionally misses it -- and those
    results are dropped rather than guessed at, the same rule
    `app/appstore/search.py` applies to a missing `trackId`.
    """
    package = raw.get("appId")
    title = raw.get("title")
    if not (package and title):
        return None
    return AppCandidate(
        platform=Platform.ANDROID,
        package_name=package,
        track_name=title,
        artist_name=raw.get("developer", "Unknown developer"),
        primary_genre_name=raw.get("genre"),
        artwork_url=raw.get("icon"),
        app_store_url=raw.get("url", f"https://play.google.com/store/apps/details?id={package}"),
        average_user_rating=raw.get("score"),
        user_rating_count=raw.get("ratings") or 0,
    )


def _to_identity(raw: dict, *, package: str, country: str) -> AppIdentity:
    """Convert one Play `app()` result into an :class:`AppIdentity`."""
    return AppIdentity(
        platform=Platform.ANDROID,
        package_name=package,
        track_name=raw.get("title", package),
        artist_name=raw.get("developer", "Unknown developer"),
        primary_genre_name=raw.get("genre"),
        artwork_url=raw.get("icon"),
        app_store_url=raw.get("url", f"https://play.google.com/store/apps/details?id={package}"),
        published_average_rating=raw.get("score"),
        published_rating_count=raw.get("ratings") or 0,
        country=country,
    )


def search_apps(
    term: str, *, country: str | None = None, settings: Settings | None = None, limit: int = 5
) -> list[AppCandidate]:
    """Search Play's catalog by name.

    Args:
        term: A free-text app name.
        country: Storefront to search, e.g. "us", "in". Defaults to
            `settings.playstore_country`.
        settings: Runtime configuration; defaults to the process settings.
        limit: Maximum candidates to return.

    Returns:
        Candidates, possibly empty.

    Raises:
        AppNotFound: The request itself failed (network, parse error). An
            empty *result set* is not this -- that is a normal empty list.
    """
    settings = settings or get_settings()
    resolved_country = country or settings.playstore_country

    try:
        results = gp_search(term, n_hits=limit, lang="en", country=resolved_country)
    except GooglePlayScraperException as exc:
        raise AppNotFound(f"the Play Store search request failed: {exc}", query=term) from exc

    candidates = [_to_candidate(raw) for raw in results]
    return [c for c in candidates if c is not None]


def lookup_app(
    package: str, *, country: str | None = None, settings: Settings | None = None
) -> AppIdentity:
    """Confirm a package name and fetch its published identity.

    Args:
        package: The Play Store package name to confirm.
        country: Storefront to look up against. Defaults to
            `settings.playstore_country`.
        settings: Runtime configuration; defaults to the process settings.

    Returns:
        The app's identity.

    Raises:
        AppNotFound: The package does not resolve to any app, or the request
            failed.
    """
    settings = settings or get_settings()
    resolved_country = country or settings.playstore_country

    try:
        raw = gp_app(package, lang="en", country=resolved_country)
    except NotFoundError as exc:
        raise AppNotFound(f"no app found for package {package!r}", query=package) from exc
    except GooglePlayScraperException as exc:
        raise AppNotFound(f"the Play Store lookup request failed: {exc}", query=package) from exc

    logger.info("resolved package %s to %s", package, raw.get("title"))
    return _to_identity(raw, package=package, country=resolved_country)


def resolve(
    raw: str, *, country: str | None = None, settings: Settings | None = None
) -> AppIdentity | list[AppCandidate]:
    """Resolve user input to one app, or a list to disambiguate.

    Args:
        raw: A package name, a Play Store URL, or a free-text name.
        country: Storefront to resolve against. Defaults to
            `settings.playstore_country`.
        settings: Runtime configuration; defaults to the process settings.

    Returns:
        An :class:`AppIdentity` when the input was already unambiguous (a
        package name or a URL), or a list of :class:`AppCandidate` for the
        caller to present when it was a name search.

    Raises:
        AppNotFound: A name search returned nothing, or a package/URL did not
            resolve to a real app.
        InvalidAppReference: The input looked like a Play Store URL but had no
            `id` parameter.
    """
    package = parse_package_name(raw)
    if package is not None:
        return lookup_app(package, country=country, settings=settings)

    candidates = search_apps(raw, country=country, settings=settings)
    if not candidates:
        raise AppNotFound(f"no apps found matching “{raw}”", query=raw)
    return candidates
