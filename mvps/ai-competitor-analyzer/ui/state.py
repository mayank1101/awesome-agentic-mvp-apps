"""Session state and the run driver.

The only module that knows both Streamlit and the pipeline, which is why the
Streamlit-specific concerns live here rather than in `app/`:

* **Search caching.** `st.cache_data` keyed on the query and its parameters, so
  the reruns Streamlit performs on every widget interaction never re-spend
  credits. A cache miss is never a correctness problem — only a cost one.
* **The run token.** Streamlit reruns the whole script constantly; without a
  token, a widget click mid-run would start a second run alongside the first
  (E-50).
* **The session cap.** A cost speed bump for a public deployment, honestly
  labelled: reloading the page resets it (E-52).
"""

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from app.core.config import Settings, get_settings
from app.models.schemas import ProgressEvent, Report

_RUN_TOKEN = "run_token"
_REPORT = "report"
_ERROR = "error"
_REPORTS_STARTED = "reports_started"
_WARNING = "guardrail_warning"


@dataclass
class RunOutcome:
    """What a finished run left behind, for the screen to render."""

    report: Report | None = None
    error: str | None = None
    error_kind: str = "generic"
    #: The query that failed to resolve, when there was one — the abstain screen
    #: names what was searched rather than shrugging (SC-5).
    error_detail: str | None = None
    events: list[ProgressEvent] = field(default_factory=list)


def init() -> None:
    """Seed session state. Safe to call on every rerun."""
    st.session_state.setdefault(_RUN_TOKEN, 0)
    st.session_state.setdefault(_REPORT, None)
    st.session_state.setdefault(_ERROR, None)
    st.session_state.setdefault(_REPORTS_STARTED, 0)
    st.session_state.setdefault(_WARNING, None)


def settings() -> Settings:
    """The process settings."""
    return get_settings()


# --------------------------------------------------------------------------- #
# Run lifecycle
# --------------------------------------------------------------------------- #


def begin_run() -> int:
    """Claim a new run and return its token (E-50)."""
    st.session_state[_RUN_TOKEN] += 1
    st.session_state[_REPORTS_STARTED] += 1
    st.session_state[_REPORT] = None
    st.session_state[_ERROR] = None
    return st.session_state[_RUN_TOKEN]


def is_current(token: int) -> bool:
    """Whether `token` still identifies the newest run."""
    return token == st.session_state[_RUN_TOKEN]


def can_start() -> bool:
    """Whether this session may start another report (E-52)."""
    cap = settings().max_reports_per_session
    return cap <= 0 or st.session_state[_REPORTS_STARTED] < cap


def reports_started() -> int:
    """How many reports this session has begun."""
    return st.session_state[_REPORTS_STARTED]


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
    """Reset to the input screen, keeping the session counter."""
    st.session_state[_REPORT] = None
    st.session_state[_ERROR] = None
    st.session_state[_WARNING] = None


def set_warning(message: str | None) -> None:
    """Hold a non-blocking guardrail warning across the rerun that follows."""
    st.session_state[_WARNING] = message


def take_warning() -> str | None:
    """Read and clear the pending warning."""
    warning = st.session_state[_WARNING]
    st.session_state[_WARNING] = None
    return warning


# --------------------------------------------------------------------------- #
# Cached search
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False, ttl=3600, max_entries=200)
def _cached_provider_search(_key_suffix: str, call_json: str) -> dict[str, Any]:
    """Call the search provider directly, memoised on the full call.

    Deliberately the *raw provider* rather than
    :class:`~app.search.client.SearchClient`: routing the cache through the
    client would build a second client with its own budget inside the memoised
    call, and the run's credit accounting would count each search twice.

    The leading underscore keeps `_key_suffix` out of Streamlit's hash while
    still busting the cache when the API key changes.
    """
    import json

    from tavily import TavilyClient

    call = json.loads(call_json)
    return TavilyClient(api_key=get_settings().tavily_api_key).search(**call)


class CachedTransport:
    """A search transport that answers repeats from Streamlit's cache.

    Satisfies the same protocol as the Tavily SDK, so `app/` stays unaware that
    caching happens at all — and the budget stays where it belongs, in the client
    above this. Streamlit reruns the script on every interaction, so without this
    a click during a run would re-spend the credits already spent (E-50).
    """

    def search(self, **kwargs: Any) -> dict[str, Any]:
        """Run one search, returning the provider's raw response."""
        import json

        key = (get_settings().tavily_api_key or "")[-6:]
        return _cached_provider_search(key, json.dumps(kwargs, sort_keys=True, default=str))
