"""Deciding which company the user meant.

A wrong-company brief is this app's worst possible output: six sections of
confident prose about the wrong Apollo, with real sources backing every word.
Two things guard against it, and neither is a model call.

* **When the user supplies a domain, there is nothing to decide.** No search, no
  credit, no guesswork.
* **When they do not, the choice is made in code and then shown.** The resolved
  name and domain sit at the top of the report, so a wrong resolution is
  something the reader catches in two seconds rather than something the app hides
  (Q-1).

When the evidence does not support a choice, the app abstains (SC-5). Abstaining
is cheap; a confident profile of the wrong company is not.
"""

from collections import Counter

from app.core.exceptions import CompanyNotFound, SearchError
from app.core.logging import get_logger
from app.models.schemas import CompanyIdentity, SearchHit
from app.search.client import SearchClient
from app.search.queries import build_resolution_query
from app.services.normalize import registrable_host

logger = get_logger(__name__)

#: Hosts that aggregate companies rather than being one. Excellent evidence
#: sources — a G2 page is where the complaints in R-6 live — and terrible answers
#: to "what is this company's own site", which is the only question they are
#: excluded from (E-16).
AGGREGATOR_HOSTS = frozenset(
    {
        "wikipedia.org",
        "linkedin.com",
        "crunchbase.com",
        "g2.com",
        "capterra.com",
        "trustpilot.com",
        "glassdoor.com",
        "producthunt.com",
        "reddit.com",
        "youtube.com",
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "medium.com",
        "github.com",
        "bloomberg.com",
        "pitchbook.com",
        "owler.com",
        "zoominfo.com",
        "apollo.io",
        "getlatka.com",
        "tracxn.com",
        "forbes.com",
        "techcrunch.com",
    }
)


def _name_tokens(name: str) -> set[str]:
    """Significant lowercase tokens of a company name, for loose matching."""
    return {token for token in name.lower().split() if len(token) > 2}


def _plausibly_matches(hit: SearchHit, name: str) -> bool:
    """Whether a hit is plausibly about the company the user named.

    Deliberately loose: this decides whether a hit may *vote* for a domain, not
    what the report says. Being strict here would abstain on companies whose
    pages never spell their own name in a title.
    """
    haystack = f"{hit.title} {hit.url} {hit.content}".lower()
    tokens = _name_tokens(name)
    if not tokens:
        return name.lower() in haystack
    return any(token in haystack for token in tokens)


def _describe(candidates: list[SearchHit], host: str, name: str) -> str | None:
    """Pick a one-line description of the company from its own pages.

    The first snippet on the right domain is *not* safe to use: a homepage
    snippet is as likely to be site furniture as prose, and a live run put a
    language picker — "English (US) Bahasa Indonesia Dansk Deutsch…" — at the top
    of a report as though it described the company.

    So a candidate must look like a sentence about this company: it has to end a
    sentence, name the company, and not read as a list of short fragments.
    Nothing qualifying is a fine outcome; the name and domain are what the banner
    exists for.

    Args:
        candidates: Resolution hits that plausibly match.
        host: The winning registrable domain.
        name: The company name.

    Returns:
        A single sentence, or `None` when nothing on the page reads like one.
    """
    for hit in candidates:
        if registrable_host(hit.url) != host:
            continue

        sentence = hit.content.strip().split(". ")[0].strip()
        if not (40 <= len(sentence) <= 240):
            continue
        if name.split()[0].lower() not in sentence.lower():
            continue
        # Site furniture is a run of short fragments; prose is not.
        words = sentence.split()
        if len(words) < 8 or sum(len(word) for word in words) / len(words) < 4:
            continue
        return f"{sentence}."

    return None


def resolve(name: str, domain: str | None, client: SearchClient) -> CompanyIdentity:
    """Work out which company to profile.

    Args:
        name: The normalised company name.
        domain: The normalised domain, if the user gave one.
        client: The budget-aware search client.

    Returns:
        The identity to profile, with `supplied_by_user` set when no search was
        needed.

    Raises:
        CompanyNotFound: No candidate domain could be justified (E-11, E-13,
            E-14). Carries the query that was run, so the screen can name what was
            searched rather than shrugging.
    """
    if domain:
        return CompanyIdentity(name=name, domain=domain, supplied_by_user=True)

    query = build_resolution_query(name)
    try:
        hits = client.search(query, topic="general", max_results=5)
    except SearchError as exc:
        raise CompanyNotFound(f"could not search for “{name}”", query=query) from exc

    if not hits:
        raise CompanyNotFound(f"no results for “{name}”", query=query)

    candidates = [hit for hit in hits if _plausibly_matches(hit, name)]
    if not candidates:
        raise CompanyNotFound(f"nothing found that looks like “{name}”", query=query)

    votes: Counter[str] = Counter()
    for hit in candidates:
        host = registrable_host(hit.url)
        if host and host not in AGGREGATOR_HOSTS:
            # Tavily's score breaks ties, so a single strong hit can still win
            # over a single weak one without letting score dominate frequency.
            votes[host] += 1 + min(hit.score, 0.99)

    if not votes:
        # Only aggregator profiles came back. That is not evidence of a findable
        # company; it is evidence that other people list it (E-13).
        raise CompanyNotFound(
            f"only directory listings found for “{name}” — try adding their website",
            query=query,
        )

    best_host, best_score = votes.most_common(1)[0]

    # E-11: a single supporting page is not a company. One hit on a domain that
    # merely echoes the name is what a parked page, a personal blog, or a
    # coincidence looks like, and profiling it produces six confident sections
    # about nothing. Two independent pages is the cheapest signal that something
    # is actually there. A user who disagrees can supply the domain and skip this
    # decision entirely.
    supporting = sum(1 for hit in candidates if registrable_host(hit.url) == best_host)
    if supporting < 2:
        raise CompanyNotFound(
            f"only one page found for “{name}” — if that is right, enter their website to confirm",
            query=query,
        )
    description = _describe(candidates, best_host, name)

    logger.info("resolved '%s' to %s (score %.2f)", name, best_host, best_score)
    return CompanyIdentity(name=name, domain=best_host, description=description)
