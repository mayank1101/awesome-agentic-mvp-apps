"""Turning search hits into the fenced, budgeted evidence the model is given.

This is the one module that handles raw third-party text, which is deliberate:
retrieved page content is both the injection surface and the thing that has to
fit a token budget, and splitting those two responsibilities across modules is
how one of them ends up unenforced.

The order of operations matters:

1. **Filter for relevance** (E-21) — one hit about a different company with a
   similar name is enough to make a whole section read authoritatively wrong.
2. **Dedupe within the section**, never across sections (E-24) — a global rule
   starves later sections for any company whose only substantial page is its
   homepage.
3. **Rank, cap, trim** — top-k by score, each snippet cut on a word boundary.
4. **Fit the global budget** — proportional trimming, so no section can crowd out
   the others (E-28).
5. **Stamp ids and defang fence markers** — ids are deterministic, so a cached
   rerun produces identical evidence (E-33), and retrieved text cannot close the
   fence it sits in (E-44).

URLs are attached to :class:`EvidenceItem` but never rendered into the prompt.
The model cites ids; the renderer resolves those ids back to URLs from these
same records. That is what makes "zero invented links" structural (SC-1).
"""

import re

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.schemas import EvidenceItem, SearchHit, SectionEvidence, SectionKey
from app.services.guardrails import defang_fence_markers
from app.services.normalize import registrable_host

logger = get_logger(__name__)

#: Ranked hits kept per section. Four snippets at ~1200 characters is a section's
#: worth of substance without letting one section dominate the budget.
MAX_ITEMS_PER_SECTION = 4

#: A trim that lands mid-word reads as though the source was corrupted, so the
#: cut is moved back to the last space -- but only if that space is reasonably
#: close to the cap, or a snippet with no spaces would collapse to nothing.
_WORD_BOUNDARY_SEARCH = 120


def _tokens(name: str) -> set[str]:
    """Significant lowercase tokens of a company name."""
    return {token for token in name.lower().split() if len(token) > 2}


def _mentions(text: str, name: str) -> bool:
    """Whether `text` names the company, matching on word boundaries.

    Word boundaries are not fussiness: a substring test on a common-word name
    matches "linear TV" in an ESPN article and "linear park" in a city-planning
    one, both of which were pulled into a live report's recent-moves section
    before this existed.
    """
    haystack = text.lower()
    tokens = _tokens(name) or {name.lower()}
    return any(re.search(rf"\b{re.escape(token)}\b", haystack) for token in tokens)


def is_relevant(hit: SearchHit, name: str, domain: str | None, *, strict: bool = False) -> bool:
    """Whether a hit is plausibly about the company being profiled (E-21).

    A hit qualifies if it is on the company's own domain, or if the name appears
    as a whole word in its title, URL, or snippet.

    Args:
        hit: The search result.
        name: The resolved company name.
        domain: The resolved domain, when known.
        strict: Require the name in the **title or URL**, not merely in the body.
            Used for news (E-23): a genuine announcement about a company names it
            in the headline, while a passing mention in the body is almost always
            a different sense of the word.

    Returns:
        True when the hit may be used as evidence.
    """
    if domain and registrable_host(hit.url) == registrable_host(f"https://{domain}"):
        return True

    if strict:
        return _mentions(f"{hit.title} {hit.url}", name)
    return _mentions(f"{hit.title} {hit.url} {hit.content}", name)


def trim(text: str, cap: int) -> str:
    """Cut text to `cap` characters, preferring a word boundary."""
    if len(text) <= cap:
        return text

    cut = text[:cap]
    space = cut.rfind(" ")
    if space > cap - _WORD_BOUNDARY_SEARCH:
        cut = cut[:space]
    return cut.rstrip() + "…"


def select_hits(
    hits: list[SearchHit],
    *,
    name: str,
    domain: str | None,
    limit: int = MAX_ITEMS_PER_SECTION,
    strict: bool = False,
) -> list[SearchHit]:
    """Filter, dedupe, and rank one section's hits.

    Deduping is by URL and **within this section only**. Across sections a URL may
    appear more than once, listed under each section that used it (E-24).

    Args:
        hits: Everything the section's search returned.
        name: The resolved company name.
        domain: The resolved domain, when known.
        limit: How many hits to keep.
        strict: Passed to :func:`is_relevant`; requires the name in the title or
            URL rather than anywhere in the text.

    Returns:
        The kept hits, best first. Own-domain pages outrank third-party ones at
        equal score, since a company's own page is the primary source for what it
        sells and charges.
    """
    seen: set[str] = set()
    kept: list[SearchHit] = []

    for hit in hits:
        if hit.url in seen or not is_relevant(hit, name, domain, strict=strict):
            continue
        seen.add(hit.url)
        kept.append(hit)

    own_host = registrable_host(f"https://{domain}") if domain else None
    kept.sort(key=lambda hit: (hit.host == own_host, hit.score), reverse=True)
    return kept[:limit]


def pack_section(
    section: SectionKey,
    query: str,
    hits: list[SearchHit],
    *,
    name: str,
    domain: str | None,
    settings: Settings,
) -> SectionEvidence:
    """Build one section's evidence.

    Args:
        section: Which section this is.
        query: The query that produced the hits, kept so an empty section can say
            what was searched.
        hits: The raw search results.
        name: The resolved company name.
        domain: The resolved domain, when known.
        settings: Runtime configuration.

    Returns:
        The packed evidence, possibly empty.
    """
    # News is filtered strictly: a real announcement names the company in its
    # headline, and a body-only match is nearly always a different sense of the
    # word -- which is how "linear TV" reached a live report (E-21).
    selected = select_hits(
        hits,
        name=name,
        domain=domain,
        strict=section is SectionKey.RECENT_MOVES,
    )

    items = tuple(
        EvidenceItem(
            # Deterministic: section key plus position, so a cached rerun yields
            # byte-identical evidence and therefore comparable output (E-33).
            id=f"{section.value}-{index}",
            title=defang_fence_markers(hit.title),
            url=hit.url,
            content=defang_fence_markers(trim(hit.content, settings.snippet_char_cap)),
            published_date=hit.published_date,
        )
        for index, hit in enumerate(selected, start=1)
    )
    return SectionEvidence(section=section, query=query, items=items)


def fit_budget(
    sections: list[SectionEvidence], *, settings: Settings
) -> tuple[SectionEvidence, ...]:
    """Trim packed sections down to the global character budget (E-28).

    Trimming is proportional rather than first-come: a section whose snippets are
    long should not be able to consume the budget and leave the last two sections
    with nothing, which is exactly what a sequential cut does.

    Args:
        sections: The packed sections, in render order.
        settings: Runtime configuration.

    Returns:
        The sections, each within its share of the budget.
    """
    total = sum(len(item.content) for section in sections for item in section.items)
    if total <= settings.evidence_char_budget:
        return tuple(sections)

    ratio = settings.evidence_char_budget / total
    logger.info(
        "evidence is %d chars, trimming to %d (%.0f%%)",
        total,
        settings.evidence_char_budget,
        ratio * 100,
    )

    trimmed: list[SectionEvidence] = []
    for section in sections:
        items = tuple(
            item.model_copy(
                update={"content": trim(item.content, max(200, int(len(item.content) * ratio)))}
            )
            for item in section.items
        )
        trimmed.append(section.model_copy(update={"items": items}))
    return tuple(trimmed)


def total_chars(sections: tuple[SectionEvidence, ...] | list[SectionEvidence]) -> int:
    """Total characters of snippet text across all sections."""
    return sum(len(item.content) for section in sections for item in section.items)


def known_urls(sections: tuple[SectionEvidence, ...] | list[SectionEvidence]) -> set[str]:
    """Every URL the app retrieved, for the render-time allowlist (SC-1)."""
    return {item.url for section in sections for item in section.items}
