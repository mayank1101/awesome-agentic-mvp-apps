"""Building the run's queries, searching, and packing the results as evidence.

Unlike this repo's job-search app, there is no domain whitelist here: travel
guidance is spread across airline blogs, city tourism boards, and personal
travel writers in a way a handful of ATS-style domains cannot capture, and the
harm model is different -- a bad job-board result costs someone an
application; a bad travel-blog result costs a paragraph a human reviews before
booking anything. What *is* enforced, the same way as every search-and-
synthesise app in this repo, is that the model never sees the URLs (see
`app.prompts`) and every named place it writes must trace to a labelled
evidence item.

Query count scales with trip length: a 3-day city break and a 12-day trip need
different amounts of raw material, and asking the same three queries for both
either wastes credits on the short trip or starves the long one.
"""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.core.config import Settings, get_settings
from app.core.exceptions import SearchError
from app.core.logging import get_logger
from app.models.schemas import (
    CategoryEvidence,
    EvidenceCategory,
    EvidenceItem,
    SearchHit,
    TripRequest,
)
from app.services.tavily import post_json

logger = get_logger(__name__)

#: Tracking parameters stripped before two URLs are compared for dedup.
_TRACKING_PARAMS = frozenset(
    "utm_source utm_medium utm_campaign utm_term utm_content utm_id "
    "ref refid ref_src source src gclid fbclid msclkid mc_cid mc_eid "
    "_hsenc _hsmi".split()
)


def canonical_url(url: str) -> str:
    """Normalise a URL enough that two links to the same page compare equal."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    if not parts.netloc:
        return url.strip()

    host = parts.netloc.lower().removeprefix("www.")
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if key.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def domain_of(url: str) -> str:
    """Return the lowercased host of a URL, without ``www.``."""
    return urlsplit(url).netloc.lower().removeprefix("www.")


def build_queries(destination: str, days: int) -> list[tuple[EvidenceCategory, str]]:
    """Build the run's queries, most important first, scaled to trip length.

    Args:
        destination: The destination as the user typed it.
        days: Trip length.

    Returns:
        (category, query) pairs. Always three categories at minimum;
        longer trips add a fourth query so there is enough material for a
        multi-day itinerary rather than repeating the same handful of stops.
    """
    queries: list[tuple[EvidenceCategory, str]] = [
        ("activities", f"top attractions and things to do in {destination}"),
        ("accommodation", f"best areas and neighborhoods to stay in {destination} for visitors"),
        ("tips", f"{destination} travel tips getting around first time visitors"),
    ]
    if days >= 4:
        queries.append(("activities", f"hidden gems local recommendations {destination}"))
    if days >= 7:
        queries.append(("activities", f"day trips near {destination}"))
    if days >= 10:
        queries.append(("activities", f"{destination} itinerary off the beaten path"))
    return queries


def _search_once(query: str, settings: Settings) -> list[SearchHit]:
    """Issue one search request and return its hits."""
    payload = {
        "query": query,
        "max_results": settings.results_per_query,
        "search_depth": settings.tavily_search_depth,
    }
    body = post_json("/search", payload)
    results = body.get("results")
    if not isinstance(results, list):
        raise SearchError("The search service returned no results array.")

    hits: list[SearchHit] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        hits.append(
            SearchHit(
                title=str(result.get("title") or url).strip(),
                url=url,
                domain=domain_of(url),
                content=str(result.get("content") or "").strip(),
                score=_as_float(result.get("score")),
            )
        )
    logger.info("Query %r returned %d usable hit(s)", query, len(hits))
    return hits


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def gather_evidence(request: TripRequest) -> list[CategoryEvidence]:
    """Run every query for a trip and pack the results into evidence, by category.

    One failed query does not fail the run -- the same rule this repo's other
    search apps apply -- but every query failing does, since a synthesis call
    with no evidence at all would just be the model writing from memory with
    extra steps.

    Args:
        request: The validated trip request.

    Returns:
        One :class:`CategoryEvidence` per category (activities, accommodation,
        tips), in that order, each capped and deduplicated, with globally
        unique item ids the prompt can reference.

    Raises:
        SearchError: Every query failed.
    """
    settings = get_settings()
    queries = build_queries(request.destination, request.days)[: settings.max_queries]

    by_category: dict[EvidenceCategory, list[SearchHit]] = {
        "activities": [],
        "accommodation": [],
        "tips": [],
    }
    query_by_category: dict[EvidenceCategory, str] = {}
    seen_urls: set[str] = set()
    failures = 0

    for category, query in queries:
        query_by_category.setdefault(category, query)
        try:
            hits = _search_once(query, settings)
        except SearchError as exc:
            logger.warning("Query failed, continuing: %s", exc)
            failures += 1
            continue

        for hit in hits:
            key = canonical_url(hit.url)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            by_category[category].append(hit)

    if failures == len(queries):
        raise SearchError("Every search failed. Check TAVILY_API_KEY and try again.")

    next_id = 1
    sections: list[CategoryEvidence] = []
    for category in ("activities", "accommodation", "tips"):
        capped = sorted(by_category[category], key=lambda h: h.score, reverse=True)[
            : settings.max_evidence_per_category
        ]
        items: list[EvidenceItem] = []
        for hit in capped:
            items.append(
                EvidenceItem(
                    id=next_id,
                    category=category,
                    title=hit.title,
                    content=hit.content[: settings.max_snippet_chars],
                    url=hit.url,
                    domain=hit.domain,
                )
            )
            next_id += 1
        sections.append(
            CategoryEvidence(
                category=category,
                query=query_by_category.get(category, ""),
                items=items,
            )
        )

    return sections
