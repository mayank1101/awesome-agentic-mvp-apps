"""Session state: the only module that touches `st.session_state` by key.

Streamlit reruns the whole script on every interaction, so anything that must
survive a click lives here. Keeping the key strings in one file is what stops the
usual Streamlit failure -- two modules writing the same key with different
meanings, or a typo creating a third key that is always empty.

Nothing here is persisted anywhere. The resume exists in this process's memory
for the life of one browser session and is gone when it ends: no database, no
disk, no logging of its content. For an app whose input is a real person's
contact details and employment history, that is a design property worth being
explicit about.
"""

from dataclasses import dataclass
from typing import Any, Literal

import streamlit as st

from app.core.config import Settings, get_settings
from app.models.schemas import TailoredResume
from app.services.analyzer import AnalysisResult

_ANALYSIS = "analysis"
_TAILORED = "tailored"
_NEXT_STEP = "next_step"
_ERROR = "error"
_NOTICE = "notice"
_RUNS = "runs"
_BUSY = "busy"

#: What the candidate chose to do about the gaps the report found.
NextStep = Literal["self", "ai"]


@dataclass(frozen=True)
class ErrorState:
    """An error to show instead of a result.

    Attributes:
        title: The headline, in plain language.
        detail: What happened and what to do about it.
        items: Optional bullet list -- guardrail findings, or the fragments a
            rewrite invented.
    """

    title: str
    detail: str
    items: list[str] | None = None


def init() -> None:
    """Create every key this app reads, once per session."""
    defaults: dict[str, Any] = {
        _ANALYSIS: None,
        _TAILORED: None,
        _NEXT_STEP: None,
        _ERROR: None,
        _NOTICE: None,
        _RUNS: 0,
        _BUSY: False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def settings() -> Settings:
    """Return the process-wide settings."""
    return get_settings()


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def analysis() -> AnalysisResult | None:
    """The current analysis, if one has finished."""
    return st.session_state[_ANALYSIS]


def store_analysis(result: AnalysisResult) -> None:
    """Store a finished analysis and drop any rewrite from a previous one.

    Dropping the rewrite is the point: a tailored resume left on screen beside a
    freshly analysed different job would be the most misleading thing this app
    could show.
    """
    st.session_state[_ANALYSIS] = result
    st.session_state[_TAILORED] = None
    st.session_state[_NEXT_STEP] = None
    st.session_state[_ERROR] = None
    st.session_state[_RUNS] += 1


def tailored() -> TailoredResume | None:
    """The current tailored resume, if one has been produced."""
    return st.session_state[_TAILORED]


def store_tailored(resume: TailoredResume) -> None:
    """Store a produced rewrite."""
    st.session_state[_TAILORED] = resume


def next_step() -> NextStep | None:
    """Which path the candidate chose after reading the report.

    ``None`` until they choose. The app does not pick for them: editing your own
    resume and taking a rewrite are different trades, and only the person whose
    resume it is knows which one they want.
    """
    return st.session_state[_NEXT_STEP]


def set_next_step(choice: NextStep) -> None:
    """Record the chosen path. Reversible -- both screens offer the other one."""
    st.session_state[_NEXT_STEP] = choice


def run_count() -> int:
    """How many analyses this session has run."""
    return st.session_state[_RUNS]


def reset() -> None:
    """Clear everything and return to the input form."""
    st.session_state[_ANALYSIS] = None
    st.session_state[_TAILORED] = None
    st.session_state[_NEXT_STEP] = None
    st.session_state[_ERROR] = None
    st.session_state[_NOTICE] = None


# --------------------------------------------------------------------------- #
# Errors and notices
# --------------------------------------------------------------------------- #


def error() -> ErrorState | None:
    """The error to display, if any."""
    return st.session_state[_ERROR]


def store_error(error_state: ErrorState) -> None:
    """Store an error for the next rerun to display."""
    st.session_state[_ERROR] = error_state


def clear_error() -> None:
    """Drop the current error."""
    st.session_state[_ERROR] = None


def set_notice(message: str) -> None:
    """Queue a one-shot informational message."""
    st.session_state[_NOTICE] = message


def take_notice() -> str | None:
    """Return the queued notice and clear it, so it shows exactly once."""
    notice = st.session_state[_NOTICE]
    st.session_state[_NOTICE] = None
    return notice


# --------------------------------------------------------------------------- #
# Re-entrancy
# --------------------------------------------------------------------------- #


def busy() -> bool:
    """Whether a run is in progress.

    Streamlit queues interactions during a script run, so a second click on
    "Analyse" while the first is still going arrives as a second run against the
    same session. The flag makes the second one a no-op rather than a duplicate
    set of paid model calls.
    """
    return bool(st.session_state[_BUSY])


def set_busy(value: bool) -> None:
    """Mark a run as started or finished."""
    st.session_state[_BUSY] = value
