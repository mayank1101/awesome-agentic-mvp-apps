"""Session state: the only module that touches `st.session_state` by key.

Streamlit reruns the whole script on every interaction, so anything that must
survive a click lives here. Nothing here is persisted anywhere -- a trip plan
lives in this process's memory for the life of one browser session and is
gone when it ends.
"""

from dataclasses import dataclass
from typing import Any

import streamlit as st

from app.core.config import Settings, get_settings
from app.models.schemas import TripPlan

_PLAN = "plan"
_ERROR = "error"
_BUSY = "busy"


@dataclass(frozen=True)
class ErrorState:
    """An error to show instead of a result.

    Attributes:
        title: The headline, in plain language.
        detail: What happened and what to do about it.
        items: Optional bullet list -- guardrail findings, for instance.
    """

    title: str
    detail: str
    items: list[str] | None = None


def init() -> None:
    """Create every key this app reads, once per session."""
    defaults: dict[str, Any] = {_PLAN: None, _ERROR: None, _BUSY: False}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def settings() -> Settings:
    """Return the process-wide settings."""
    return get_settings()


def plan() -> TripPlan | None:
    """The current trip plan, if one has finished."""
    return st.session_state[_PLAN]


def store_plan(result: TripPlan) -> None:
    """Store a finished plan and clear any previous error."""
    st.session_state[_PLAN] = result
    st.session_state[_ERROR] = None


def error() -> ErrorState | None:
    """The error to display, if any."""
    return st.session_state[_ERROR]


def store_error(error_state: ErrorState) -> None:
    """Store an error for the next rerun to display."""
    st.session_state[_ERROR] = error_state


def clear_error() -> None:
    """Drop the current error."""
    st.session_state[_ERROR] = None


def reset() -> None:
    """Clear everything and return to the input form."""
    st.session_state[_PLAN] = None
    st.session_state[_ERROR] = None


def busy() -> bool:
    """Whether a run is in progress.

    Streamlit queues interactions during a script run, so a second click on
    "Plan my trip" while the first is still going arrives as a second run
    against the same session. The flag makes the second one a no-op rather
    than a duplicate set of paid calls.
    """
    return bool(st.session_state[_BUSY])


def set_busy(value: bool) -> None:
    """Mark a run as started or finished."""
    st.session_state[_BUSY] = value
