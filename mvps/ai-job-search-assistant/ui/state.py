"""Session state: the only module that touches `st.session_state` by key.

Streamlit reruns the whole script on every interaction, so anything that must
survive a click lives here. Keeping the key strings in one file is what stops the
usual Streamlit failure -- two modules writing the same key with different
meanings, or a typo creating a third key that is always empty.

Nothing here is persisted anywhere. The resume exists in this process's memory
for the life of one browser session and is gone when it ends: no database, no
disk, no logging of its content. For an app whose input is a real person's
employment history, that is a design property worth being explicit about.
"""

from dataclasses import dataclass
from typing import Any

import streamlit as st

from app.core.config import DEFAULT_JOB_SITES, Settings, get_settings
from app.models.schemas import RunResult

_RESULT = "result"
_ERROR = "error"
_NOTICE = "notice"
_SITES = "sites"
_RUNS = "runs"


@dataclass(frozen=True)
class ErrorState:
    """An error to show instead of a result.

    Attributes:
        title: The headline, in plain language.
        detail: What happened and what to do about it.
        items: Optional bullet list -- guardrail findings, usually.
    """

    title: str
    detail: str
    items: list[str] | None = None


def init() -> None:
    """Create every key this app reads, once per session."""
    defaults: dict[str, Any] = {
        _RESULT: None,
        _ERROR: None,
        _NOTICE: None,
        _SITES: "\n".join(DEFAULT_JOB_SITES),
        _RUNS: 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def settings() -> Settings:
    """Return the process-wide settings."""
    return get_settings()


def result() -> RunResult | None:
    """The current run's result, if one has finished."""
    return st.session_state[_RESULT]


def store_result(value: RunResult) -> None:
    """Store a finished run and clear any error from a previous one."""
    st.session_state[_RESULT] = value
    st.session_state[_ERROR] = None
    st.session_state[_RUNS] += 1


def error() -> ErrorState | None:
    """The error to show instead of results, if there is one."""
    return st.session_state[_ERROR]


def store_error(value: ErrorState) -> None:
    """Store an error and drop the previous result.

    Dropping the result is the point: a shortlist left on screen under an error
    about a *different* search is the most misleading thing this app could show.
    """
    st.session_state[_ERROR] = value
    st.session_state[_RESULT] = None


def sites_text() -> str:
    """The site whitelist as the user has it, one domain per line."""
    return st.session_state[_SITES]


def store_sites(value: str) -> None:
    """Remember the edited whitelist across reruns."""
    st.session_state[_SITES] = value


def reset() -> None:
    """Clear the result and the error, returning to the search form."""
    st.session_state[_RESULT] = None
    st.session_state[_ERROR] = None


def set_notice(message: str) -> None:
    """Queue a one-off message to show above the form on the next rerun."""
    st.session_state[_NOTICE] = message


def take_notice() -> str | None:
    """Return the queued message and clear it, so it shows exactly once."""
    message = st.session_state[_NOTICE]
    st.session_state[_NOTICE] = None
    return message


def run_count() -> int:
    """How many searches have finished this session."""
    return st.session_state[_RUNS]
