"""The input screen: a platform, a country, an app query, and one button.

A name search can be ambiguous ("Notes" matches a dozen apps), so this module
has two draw functions: :func:`render` for the initial query, and
:func:`render_picker` for the candidate list that appears when the query
resolved to more than one plausible app. A URL, numeric id, or package name
skips the picker entirely -- there is nothing to disambiguate.

Platform and country are explicit UI choices, not auto-detected from the
query: a free-text name is inherently platform-ambiguous ("Notion" is a name
on both stores), and guessing wrong would silently search the wrong catalog.
"""

from dataclasses import dataclass

import streamlit as st

from app.core.exceptions import AppNotFound, InvalidAppReference
from app.models.schemas import AppCandidate, AppIdentity, Platform
from app.services.guardrails import has_severity, scan_input
from app.services.normalize import is_meaningful_query, normalize_query
from ui import state as S

#: Confirmed by direct testing (docs/01-prd.md §7-addendum) to return real
#: Play Store review data, every one on the first attempt -- unlike the iOS
#: feed, Play's own review system is properly localized per storefront, so
#: this list is offered on that basis rather than one-by-one necessity.
#: Play supports more storefronts than this; the list is what has actually
#: been verified rather than an attempt to be exhaustive.
_PLAYSTORE_COUNTRIES = {
    "us": "United States",
    "gb": "United Kingdom",
    "in": "India",
    "ca": "Canada",
    "au": "Australia",
    "de": "Germany",
    "fr": "France",
    "br": "Brazil",
    "jp": "Japan",
    "mx": "Mexico",
    "id": "Indonesia",
    "ng": "Nigeria",
}

_PLATFORM_LABELS = {Platform.IOS: "iOS App Store", Platform.ANDROID: "Google Play"}


@dataclass
class Submission:
    """What the query form produced on this rerun."""

    submitted: bool = False
    identity: AppIdentity | None = None
    error: str | None = None
    warning: str | None = None


def render() -> Submission:
    """Draw the platform/country/query form and resolve whatever was submitted.

    Returns:
        The submission. When the query resolved to multiple candidates, this
        stores them via `ui.state.set_candidates` and returns an empty (not
        submitted) result -- the picker draws on the next rerun instead.
    """
    st.markdown(
        "Pulls a competitor's **most recent reviews** and clusters the critical ones into "
        "named feature gaps, each backed by real review excerpts — never invented quotes."
    )

    platform_choice = st.radio(
        "Store",
        options=[Platform.IOS, Platform.ANDROID],
        format_func=lambda p: _PLATFORM_LABELS[p],
        horizontal=True,
    )

    if platform_choice is Platform.IOS:
        st.caption(
            "Fixed to the US storefront — Apple's review feed does not reliably serve any "
            "other (see README)."
        )
        country = "us"
    else:
        country = st.selectbox(
            "Storefront",
            options=list(_PLAYSTORE_COUNTRIES),
            format_func=lambda c: _PLAYSTORE_COUNTRIES[c],
        )

    with st.form("resolve_app", clear_on_submit=False):
        placeholder, help_text = _query_hint(platform_choice)
        query = st.text_input("Competitor app", placeholder=placeholder, help=help_text)
        clicked = st.form_submit_button("Find app", type="primary", use_container_width=True)

    if not clicked:
        return Submission()

    return _resolve(query, platform=platform_choice, country=country)


def _query_hint(platform: Platform) -> tuple[str, str]:
    """Placeholder and help text for the query field, tailored to the platform."""
    if platform is Platform.IOS:
        return (
            "Notion, or https://apps.apple.com/us/app/notion/id1232780281, or 1232780281",
            "A name to search, a full App Store URL, or a numeric App Store id.",
        )
    return (
        "Notion, or https://play.google.com/store/apps/details?id=notion.id, or com.spotify.music",
        "A name to search, a full Play Store URL, or a package name (e.g. com.spotify.music).",
    )


def _resolve(raw_query: str, *, platform: Platform, country: str) -> Submission:
    """Normalise, scan, and resolve the submitted query."""
    settings = S.settings()
    query = normalize_query(raw_query)

    if not is_meaningful_query(query):
        return Submission(submitted=True, error="Enter an app name, store URL, or app id.")

    warning: str | None = None
    if settings.guardrails_enabled:
        findings = scan_input(query)
        if findings:
            worst = findings[0]
            if has_severity(findings, "high") and settings.block_flagged_input:
                return Submission(
                    submitted=True,
                    error=f"That input {worst.message}. Enter an app name on its own.",
                )
            warning = f"Heads up: the input {worst.message}. Searching anyway."

    try:
        result = S.cached_resolve(query, platform=platform, country=country)
    except (AppNotFound, InvalidAppReference) as exc:
        return Submission(submitted=True, error=str(exc))

    if isinstance(result, list):
        S.set_candidates(query, result, country=country)
        return Submission(submitted=True, warning=warning)

    return Submission(submitted=True, identity=result, warning=warning)


def render_picker(query: str, candidates: list[AppCandidate]) -> AppIdentity | None:
    """Draw the disambiguation list for a name search with multiple matches.

    Args:
        query: The search query that produced these candidates, shown for
            context.
        candidates: The store's own relevance-ordered matches. All share one
            platform -- a single query is only ever run against one store.

    Returns:
        The chosen app's identity, or `None` if nothing has been picked yet on
        this rerun.
    """
    st.markdown(f"**{len(candidates)} apps found for “{query}”** — pick the one to analyze:")

    labels = [f"{c.track_name} — {c.artist_name} ({c.primary_genre_name or 'App'})" for c in candidates]
    choice = st.radio("Matches", options=range(len(candidates)), format_func=lambda i: labels[i])

    col1, col2 = st.columns(2)
    with col1:
        confirmed = st.button("Use this app", type="primary", use_container_width=True)
    with col2:
        cancelled = st.button("Search again", use_container_width=True)

    if cancelled:
        S.clear_candidates()
        st.rerun()

    if confirmed:
        picked = candidates[choice]
        country = S.current_candidates_country()
        S.clear_candidates()
        return S.cached_lookup(picked, country=country)

    return None
