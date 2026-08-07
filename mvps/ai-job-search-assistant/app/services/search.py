"""Running the run's queries and turning results into job hits.

The search itself is one request per query. Everything interesting here happens
to what comes back.

**The whitelist is enforced twice.** Once at the API, via ``include_domains``,
which is what makes it a restriction rather than a filter -- off-list pages are
never fetched and never paid for. And once here, on every result, because a
provider that ignores or partially honours the parameter would otherwise put an
off-list link in front of a user who was promised the opposite. The second check
has never fired in testing. It stays because "the API honoured it last time" is
not the guarantee this app made.

**Listing pages are dropped before they cost anything.** Roughly a third of what
a job-site search returns is not a job: board landing pages, company profiles,
"42 open roles at Acme". They are rejected on URL shape (:mod:`app.services.sites`),
before any page is fetched.

**Duplicates are collapsed across queries and across boards.** Four queries
returning ten results each do not produce forty jobs; they produce the same
strong postings several times, plus the same role listed on both an ATS and
LinkedIn. Both kinds of duplicate are collapsed, and the *first* seen wins --
which, because queries are issued most-important-first, is the one found by the
most on-target query.
"""

from typing import Any

from app.core.config import get_settings
from app.core.exceptions import SearchError
from app.core.logging import get_logger
from app.models.schemas import JobHit
from app.services import sites as site_rules
from app.services.tavily import post_json

logger = get_logger(__name__)

#: Tavily's general search takes a coarse recency window rather than a day
#: count (`days` applies only to its news topic). Mapping to the nearest window
#: that does not *exclude* what the user asked for: a 45-day request becomes a
#: year rather than a month, because silently dropping postings the user asked
#: to see is worse than including some they did not.
_TIME_RANGES: tuple[tuple[int, str], ...] = (
    (1, "day"),
    (7, "week"),
    (31, "month"),
    (366, "year"),
)


def time_range_for(days: int | None) -> str | None:
    """Map a recency window in days onto the provider's coarse ranges."""
    if not days or days <= 0:
        return None
    for ceiling, label in _TIME_RANGES:
        if days <= ceiling:
            return label
    return None


def search_jobs(
    queries: list[str],
    sites: list[str],
    *,
    recency_days: int | None = None,
) -> tuple[list[JobHit], int]:
    """Run every query against the whitelist and return the hits worth keeping.

    Args:
        queries: The run's queries, most important first.
        sites: The domain whitelist. Must not be empty -- an empty
            ``include_domains`` searches the whole web, which is the one thing
            this app promises not to do.
        recency_days: Restrict to postings from the last N days, if given.

    Returns:
        The surviving hits in first-seen order, and the raw number of results
        the provider returned across all queries. The second number is what
        lets the UI say "38 results, 22 postings, 8 read in full" instead of
        presenting eight jobs as if that were everything there was.

    Raises:
        SearchError: Every query failed. A single query failing is survivable
            and logged; all of them failing is the run.
    """
    if not sites:
        raise SearchError("No sites to search. Add at least one job site.")
    if not queries:
        return [], 0

    settings = get_settings()
    allowed = set(sites)
    time_range = time_range_for(recency_days)

    hits: list[JobHit] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    raw_count = 0
    failures: list[str] = []

    for query in queries:
        try:
            results = _search_once(query, sites, time_range)
        except SearchError as exc:
            # One bad query should not lose the run. Four queries exist partly
            # so that a provider hiccup on one still leaves a usable shortlist.
            logger.warning("Query failed, continuing: %s", exc)
            failures.append(str(exc))
            continue

        raw_count += len(results)
        for result in results:
            hit = _to_hit(result, query)
            if hit is None:
                continue
            if hit.domain not in allowed and not _is_subdomain_of_allowed(hit.domain, allowed):
                logger.warning("Dropping off-whitelist result from %s", hit.domain)
                continue

            url_key, title_key = site_rules.dedupe_key(hit.url, hit.title)
            if url_key in seen_urls or (title_key and title_key in seen_titles):
                continue
            seen_urls.add(url_key)
            if title_key:
                seen_titles.add(title_key)

            hits.append(hit)
            if len(hits) >= settings.max_results_total:
                logger.info("Result cap of %d reached", settings.max_results_total)
                return hits, raw_count

    if failures and not hits and len(failures) == len(queries):
        raise SearchError(failures[0])

    return hits, raw_count


def _search_once(query: str, sites: list[str], time_range: str | None) -> list[dict[str, Any]]:
    """Issue one search request and return its raw results."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "query": query,
        "max_results": settings.results_per_query,
        "search_depth": settings.tavily_search_depth,
        "include_domains": sites,
    }
    if time_range:
        payload["time_range"] = time_range

    body = post_json("/search", payload)
    results = body.get("results")
    if not isinstance(results, list):
        raise SearchError("The search service returned no results array.")

    logger.info("Query %r returned %d results", query, len(results))
    return [result for result in results if isinstance(result, dict)]


def _to_hit(result: dict[str, Any], query: str) -> JobHit | None:
    """Build a :class:`JobHit`, or ``None`` if the result is not a posting."""
    url = str(result.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return None
    if not site_rules.is_probable_posting(url):
        return None

    return JobHit(
        url=site_rules.canonical_url(url),
        title=_clean(result.get("title")),
        domain=site_rules.domain_of(url),
        snippet=_clean(result.get("content")),
        published=_clean(result.get("published_date")),
        provider_score=_as_float(result.get("score")),
        query=query,
    )


def _is_subdomain_of_allowed(domain: str, allowed: set[str]) -> bool:
    """Whether `domain` sits under a whitelisted host.

    A whitelist entry of ``greenhouse.io`` is a statement about the company, and
    postings live on ``boards.greenhouse.io`` and ``job-boards.greenhouse.io``.
    Rejecting those because the string does not match exactly would make the
    obvious whitelist entry the wrong one.
    """
    return any(domain.endswith(f".{entry}") for entry in allowed)


def _clean(value: object) -> str:
    """Coerce a provider field to a stripped string."""
    return str(value).strip() if isinstance(value, str) else ""


def _as_float(value: object) -> float:
    """Coerce a provider score to a float, defaulting to zero."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
