"""Session state and the cached store calls.

The only module that knows both Streamlit and the app/appstore + app/playstore
layer, for the same reason as this repo's other apps: Streamlit reruns the
whole script on every widget interaction, so anything that hits the network
needs either a cache or a run token guarding it, or a rerun mid-flow re-does
the work.

Two stages live in session state between the input form and the report:
`candidates` (a name search returned more than one plausible match, waiting on
a pick -- alongside the platform and country it was searched under, since a
picked candidate needs both to be looked up) and `report`/`error` (a run
finished). There is no paid budget to protect here -- every endpoint used is
free and keyless -- so the caching below is about avoiding redundant network
calls on a rerun, not about cost.
"""

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from app.core.config import Settings, get_settings
from app.models.schemas import AppCandidate, AppIdentity, Platform, ProgressEvent, Report, Review

_RUN_TOKEN = "run_token"
_CANDIDATES = "candidates"
_CANDIDATES_QUERY = "candidates_query"
_CANDIDATES_PLATFORM = "candidates_platform"
_CANDIDATES_COUNTRY = "candidates_country"
_REPORT = "report"
_ERROR = "error"
_WARNING = "warning"


@dataclass
class RunOutcome:
    """What a finished run left behind, for the screen to render."""

    report: Report | None = None
    error: str | None = None
    error_kind: str = "generic"
    events: list[ProgressEvent] = field(default_factory=list)


def init() -> None:
    """Seed session state. Safe to call on every rerun."""
    st.session_state.setdefault(_RUN_TOKEN, 0)
    st.session_state.setdefault(_CANDIDATES, None)
    st.session_state.setdefault(_CANDIDATES_QUERY, None)
    st.session_state.setdefault(_CANDIDATES_PLATFORM, None)
    st.session_state.setdefault(_CANDIDATES_COUNTRY, None)
    st.session_state.setdefault(_REPORT, None)
    st.session_state.setdefault(_ERROR, None)
    st.session_state.setdefault(_WARNING, None)


def settings() -> Settings:
    """The process settings."""
    return get_settings()


# --------------------------------------------------------------------------- #
# Candidate disambiguation
# --------------------------------------------------------------------------- #


def set_candidates(query: str, candidates: list[AppCandidate], *, country: str) -> None:
    """Hold a name search's results for the picker to draw."""
    st.session_state[_CANDIDATES_QUERY] = query
    st.session_state[_CANDIDATES] = candidates
    st.session_state[_CANDIDATES_PLATFORM] = candidates[0].platform if candidates else None
    st.session_state[_CANDIDATES_COUNTRY] = country


def current_candidates() -> tuple[str, list[AppCandidate]] | None:
    """The pending (query, candidates) pair, if a pick is outstanding."""
    candidates = st.session_state[_CANDIDATES]
    if candidates is None:
        return None
    return st.session_state[_CANDIDATES_QUERY], candidates


def current_candidates_country() -> str:
    """The country the pending candidates were searched under."""
    return st.session_state[_CANDIDATES_COUNTRY] or "us"


def clear_candidates() -> None:
    """Drop the pending picker, either because one was chosen or cancelled."""
    st.session_state[_CANDIDATES] = None
    st.session_state[_CANDIDATES_QUERY] = None
    st.session_state[_CANDIDATES_PLATFORM] = None
    st.session_state[_CANDIDATES_COUNTRY] = None


# --------------------------------------------------------------------------- #
# Run lifecycle
# --------------------------------------------------------------------------- #


def begin_run() -> int:
    """Claim a new run and return its token."""
    st.session_state[_RUN_TOKEN] += 1
    st.session_state[_REPORT] = None
    st.session_state[_ERROR] = None
    return st.session_state[_RUN_TOKEN]


def is_current(token: int) -> bool:
    """Whether `token` still identifies the newest run."""
    return token == st.session_state[_RUN_TOKEN]


def store_outcome(outcome: RunOutcome) -> None:
    """Persist a finished run for the next rerun to draw."""
    st.session_state[_REPORT] = outcome.report
    st.session_state[_ERROR] = (outcome.error, outcome.error_kind) if outcome.error else None


def current_report() -> Report | None:
    """The report on screen, if any."""
    return st.session_state[_REPORT]


def current_error() -> tuple[str, str] | None:
    """The error on screen as (message, kind), if any."""
    return st.session_state[_ERROR]


def clear() -> None:
    """Reset to the input screen."""
    st.session_state[_REPORT] = None
    st.session_state[_ERROR] = None
    st.session_state[_WARNING] = None
    clear_candidates()


def set_warning(message: str | None) -> None:
    """Hold a non-blocking guardrail warning across the rerun that follows."""
    st.session_state[_WARNING] = message


def take_warning() -> str | None:
    """Read and clear the pending warning."""
    warning = st.session_state[_WARNING]
    st.session_state[_WARNING] = None
    return warning


# --------------------------------------------------------------------------- #
# Cached store calls
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, ttl=600, max_entries=200)
def cached_resolve(
    raw: str, *, platform: Platform, country: str
) -> AppIdentity | list[AppCandidate]:
    """Resolve input to an identity or a candidate list, dispatched by platform.

    Memoised on `(raw, platform, country)` -- the same query means the same
    result within the TTL, and a rerun (Streamlit's default behaviour on any
    widget interaction) must not re-hit the network for a query already
    resolved this session.
    """
    if platform is Platform.IOS:
        from app.appstore.search import resolve

        return resolve(raw, settings=get_settings())

    from app.playstore.search import resolve

    return resolve(raw, country=country, settings=get_settings())


@st.cache_data(show_spinner=False, ttl=600, max_entries=200)
def cached_lookup(candidate: AppCandidate, *, country: str) -> AppIdentity:
    """Confirm a picked candidate into a full identity, dispatched by platform."""
    if candidate.platform is Platform.IOS:
        from app.appstore.search import lookup_app

        return lookup_app(candidate.track_id, settings=get_settings())

    from app.playstore.search import lookup_app

    return lookup_app(candidate.package_name, country=country, settings=get_settings())


@st.cache_data(show_spinner=False, ttl=300, max_entries=100)
def _cached_fetch_raw(
    platform: Platform, external_id: str, country: str, max_reviews: int
) -> list[dict[str, Any]]:
    """Fetch reviews and return them as plain dicts (cache-hashable, unlike Review)."""
    settings = get_settings()
    if platform is Platform.IOS:
        from app.appstore.reviews import fetch_reviews

        reviews = fetch_reviews(int(external_id), settings=settings)
    else:
        from app.playstore.reviews import fetch_reviews

        reviews = fetch_reviews(external_id, country=country, settings=settings)
    return [r.model_dump(mode="json") for r in reviews]


def cached_fetch_reviews(identity: AppIdentity, settings: Settings) -> list[Review]:
    """A `default_fetch_reviews`-shaped callable backed by Streamlit's cache.

    Matches :data:`app.services.pipeline.FetchReviews` so the pipeline stays
    unaware that caching happens at all.
    """
    raw = _cached_fetch_raw(identity.platform, identity.external_id, identity.country, settings.max_reviews)
    return [Review.model_validate(item) for item in raw]
