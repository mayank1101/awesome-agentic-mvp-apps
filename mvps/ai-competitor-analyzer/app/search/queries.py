"""The six section queries, and their per-section search parameters.

One search per section, always the same six, in the same order. That is what
makes the section-to-sources mapping trivial, the credit cost predictable, and
one brief comparable with the next (SC-8, `02` section 1).

Query wording is deliberately keyword-shaped rather than conversational: this
goes to a search API, not to a model, and "Acme pricing plans cost per user"
retrieves better than "what does Acme charge?".
"""

from typing import Any

from app.core.config import Settings
from app.models.schemas import SectionKey

#: Query templates. `{name}` is the resolved company name, `{domain}` its host or
#: an empty string when unknown.
_TEMPLATES: dict[SectionKey, str] = {
    SectionKey.SNAPSHOT: "{name} {domain} company overview what it does founded headquarters",
    SectionKey.PRODUCT: "{name} product features platform capabilities",
    SectionKey.PRICING: "{name} pricing plans cost per user",
    SectionKey.POSITIONING: '{name} for whom customers positioning "we help"',
    # Quoted: the news index is where a common-word name does the most damage,
    # and an unquoted "Linear" retrieves articles about linear TV and linear
    # parks. The evidence filter is strict here too -- belt and braces, because
    # a wrong item in this section is a dated, specific, wrong claim.
    SectionKey.RECENT_MOVES: '"{name}" company announcement launch funding acquisition',
    SectionKey.STRENGTHS_WEAKNESSES: "{name} reviews pros cons complaints alternatives",
}

#: The query used to work out which company the user meant, when they did not
#: supply a domain.
RESOLUTION_TEMPLATE = '"{name}" official website company'


def build_query(section: SectionKey, name: str, domain: str | None) -> str:
    """Render one section's query.

    Args:
        section: Which section the query is for.
        name: The company name.
        domain: Its host, or `None` — the snapshot query includes it when known,
            since a domain is the strongest disambiguator a search engine gets.

    Returns:
        The query string to send.
    """
    return " ".join(_TEMPLATES[section].format(name=name, domain=domain or "").split())


def build_resolution_query(name: str) -> str:
    """Render the identity-resolution query."""
    return RESOLUTION_TEMPLATE.format(name=name)


def search_params(section: SectionKey, settings: Settings) -> dict[str, Any]:
    """Return the per-section search parameters.

    Recent moves is the only section that differs, and it differs twice: it asks
    the news index rather than the general one, and it constrains recency at the
    API rather than asking the model to filter by date afterwards (E-23, Q-4). It
    also requests more results than the others because items without a date are
    dropped later, so some of what comes back is expected not to survive.

    Args:
        section: Which section is being searched.
        settings: Runtime configuration.

    Returns:
        Keyword arguments for the search client.
    """
    if section is SectionKey.RECENT_MOVES:
        return {
            "topic": "news",
            "time_range": "year",
            "max_results": settings.max_results_recent_moves,
        }
    return {"topic": "general", "max_results": settings.max_results_per_section}
