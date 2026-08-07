"""The whitelist, and what counts as a job posting.

Three jobs, all of them boring and all of them load-bearing:

**Normalising domains.** The user edits the site list by hand, so it arrives as
whatever they typed -- ``https://boards.greenhouse.io/``, ``www.linkedin.com``,
``Naukri.com``. The search provider wants bare hosts. A domain that fails to
normalise is dropped with a notice rather than sent, because a malformed entry
silently widens or narrows the search and neither is visible from the results.

**Canonicalising URLs.** The same posting reaches a search index under several
URLs -- tracking parameters, a trailing slash, an uppercase host, a fragment.
De-duplicating on the raw string keeps all of them, and a shortlist with the same
job three times is a shortlist the user stops trusting on sight.

**Deciding what is a posting.** A search restricted to job sites returns plenty
of pages that are not jobs: board landing pages, company profiles, "42 jobs at
Acme" list pages, blog posts about hiring. Scoring a resume against a category
page produces a confident number about nothing. The check is deliberately a
*URL-shape* check rather than a content check: it runs before anything is
fetched, so a rejected page costs nothing.

The per-site patterns are the part most likely to rot -- job boards restructure
URLs -- so they are all in one table with the shape they match, and the generic
fallback is written to be safe when a pattern goes stale: an unrecognised host is
judged by structure, not rejected.
"""

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Query parameters that identify a referrer rather than a document. Stripping
#: them is what makes two links to the same job compare equal. Anything not
#: listed here is kept -- several boards put the job id in the query string
#: (``?gh_jid=``, ``?jk=``), and dropping those would collapse every job on a
#: board into one entry.
#:
#: ``embed`` is here for a measured reason: a live run returned an Ashby posting
#: as ``...?embed=js``, which is the widget variant of the page. Extraction
#: failed on it outright ("Failed to fetch url") while the same posting without
#: the parameter reads fine, so the job was demoted to a snippet score for a
#: reason that had nothing to do with the job.
_TRACKING_PARAMS = frozenset(
    """
    utm_source utm_medium utm_campaign utm_term utm_content utm_id
    ref refid ref_src source src trk trackingid tracking_id gh_src
    originalsubdomain position pagenum page_num refelement
    gclid fbclid msclkid mc_cid mc_eid _hsenc _hsmi embed
    """.split()
)

#: URL shapes that are a single posting, per host suffix. Matched against the
#: path, lowercased. First match wins, and a host that matches none of these
#: falls through to :func:`_generic_posting_shape`.
_POSTING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Greenhouse serves the same board from two hosts and both are in use.
    ("greenhouse.io", re.compile(r"^/[^/]+/jobs/\d+")),
    ("lever.co", re.compile(r"^/[^/]+/[0-9a-f-]{8,}")),
    ("ashbyhq.com", re.compile(r"^/[^/]+/[0-9a-f-]{8,}")),
    ("workable.com", re.compile(r"^/j/[0-9a-z]+")),
    ("linkedin.com", re.compile(r"^/jobs/view/")),
    ("naukri.com", re.compile(r"^/job-listings-|^/jobs?/[^/]+-\d+")),
    ("wellfound.com", re.compile(r"^/(jobs|l|company/[^/]+/jobs)/\d+")),
    ("indeed.com", re.compile(r"^/viewjob|^/rc/clk")),
    ("smartrecruiters.com", re.compile(r"^/[^/]+/\d{6,}")),
    ("recruitee.com", re.compile(r"^/o/[^/]+")),
    ("bamboohr.com", re.compile(r"^/careers/\d+")),
    ("workday(?:jobs)?\\.com", re.compile(r"/job/")),
)

#: Path segments that mark a listing index rather than a single job. Checked as
#: the *last* segment, so ``/jobs/12345`` survives and ``/jobs`` does not.
_INDEX_SEGMENTS = frozenset(
    """
    jobs job careers career openings opportunities vacancies search results
    board boards companies company about login signup blog
    """.split()
)

#: A last path segment that is a single job usually contains a number or a long
#: opaque token. Not sufficient alone, which is why it is only consulted for
#: hosts with no pattern of their own.
_HAS_ID = re.compile(r"\d{4,}|[0-9a-f]{8}-[0-9a-f]{4}|[a-z0-9]{10,}", re.IGNORECASE)

_DOMAIN_SHAPE = re.compile(r"^(?!-)[a-z0-9-]+(?:\.[a-z0-9-]+)+$")


def normalize_domain(raw: str) -> str | None:
    """Reduce whatever the user typed to a bare host, or reject it.

    Args:
        raw: A domain as typed: with or without a scheme, ``www.``, a path, or
            surrounding whitespace.

    Returns:
        The lowercased host, or ``None`` if it does not look like a domain at
        all. ``www.`` is stripped because search providers treat it as part of
        the host and no job board serves postings only from it.
    """
    value = (raw or "").strip().lower()
    if not value:
        return None

    if "//" in value:
        value = value.split("//", 1)[1]
    value = value.split("/", 1)[0].split("?", 1)[0].split("@")[-1]
    value = value.split(":", 1)[0]
    value = value.removeprefix("www.").strip(".")

    return value if _DOMAIN_SHAPE.match(value) else None


def normalize_sites(raw_sites: list[str]) -> tuple[list[str], list[str]]:
    """Normalise a whole site list, keeping order and reporting what was dropped.

    Args:
        raw_sites: Domains as the user left them.

    Returns:
        The accepted hosts, de-duplicated with first-seen order preserved, and
        the raw entries that could not be understood. The caller shows the
        second list rather than discarding it silently -- a typo'd domain is a
        search that quietly never happened.
    """
    accepted: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()

    for raw in raw_sites:
        domain = normalize_domain(raw)
        if domain is None:
            if raw.strip():
                rejected.append(raw.strip())
            continue
        if domain not in seen:
            seen.add(domain)
            accepted.append(domain)

    return accepted, rejected


def domain_of(url: str) -> str:
    """Return the lowercased host of a URL, without ``www.``."""
    host = urlsplit(url).netloc.lower().split("@")[-1].split(":", 1)[0]
    return host.removeprefix("www.")


def canonical_url(url: str) -> str:
    """Normalise a URL enough that two links to one posting compare equal.

    Lowercases the host (paths stay case-sensitive, because some boards' job
    tokens are), drops the fragment, removes tracking parameters, sorts what
    remains, and strips a trailing slash.

    Args:
        url: A URL as the search provider returned it.

    Returns:
        The canonical form, or the input unchanged if it cannot be parsed.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()

    if not parts.netloc:
        return url.strip()

    host = parts.netloc.lower().split("@")[-1]
    if host.startswith("www."):
        host = host[4:]

    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if key.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))

    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def is_probable_posting(url: str) -> bool:
    """Whether a URL looks like one job posting rather than a listing page.

    Args:
        url: A URL, canonical or not.

    Returns:
        ``True`` if the URL's shape matches a known board's posting pattern, or
        -- for hosts with no pattern -- if it has the structure of a detail page.
        A URL that cannot be parsed is rejected.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if not parts.netloc or not parts.path:
        return False

    host = domain_of(url)
    path = parts.path.rstrip("/") or "/"
    query = parts.query.lower()

    for suffix, pattern in _POSTING_PATTERNS:
        if re.search(rf"(^|\.){suffix}$", host) and (
            pattern.search(path.lower()) or pattern.search(f"{path.lower()}?{query}")
        ):
            return True
        if re.search(rf"(^|\.){suffix}$", host):
            # A recognised board that did not match its own posting shape is a
            # listing page on that board. Falling through to the generic check
            # would let ``/companies/acme`` through on the strength of a long
            # slug, which is exactly the page this function exists to reject.
            return _query_string_posting(query)

    return _generic_posting_shape(path, query)


def _query_string_posting(query: str) -> bool:
    """Handle boards that put the job id in the query string.

    Greenhouse's embedded boards are served from a company's own domain as
    ``?gh_jid=1234``, and Workday's from ``?jobId=``. These are real postings
    with a listing-shaped path, so they are checked explicitly rather than
    handed to a heuristic that would reject them.
    """
    params = dict(parse_qsl(query))
    return any(params.get(key) for key in ("gh_jid", "jobid", "job_id", "jk", "currentjobid"))


def _generic_posting_shape(path: str, query: str) -> bool:
    """Judge an unrecognised host by URL structure alone.

    Requires a path with at least two segments (a single-segment path is a
    section, not a document), a last segment that is not a plain index word, and
    something id-shaped somewhere in the path or query. Conservative by design:
    a missed posting costs one row, a category page scored as a job costs the
    user's trust in every number on the screen.
    """
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) < 2:
        return _query_string_posting(query)

    if segments[-1].lower() in _INDEX_SEGMENTS:
        return False

    return bool(_HAS_ID.search("/".join(segments)) or _query_string_posting(query))


def dedupe_key(url: str, title: str) -> tuple[str, str]:
    """The key two search results must share to be considered the same job.

    The canonical URL alone is not enough: the same role is genuinely posted at
    ``jobs.lever.co/acme/<id>`` and ``linkedin.com/jobs/view/<other-id>``, and
    those are two URLs for one job. The title is what catches that, normalised
    hard -- case, punctuation, and the location suffix boards append ("Senior
    Engineer - Bengaluru, India") all vary between listings of one role.

    Args:
        url: The result URL.
        title: The result title.

    Returns:
        A ``(canonical_url, normalised_title)`` pair. Callers treat a match on
        *either* element as a duplicate, which is why both are returned rather
        than one combined string.
    """
    normalised = re.sub(r"[^a-z0-9 ]+", " ", title.lower())
    normalised = re.sub(r"\s+", " ", normalised).strip()
    return canonical_url(url), normalised
