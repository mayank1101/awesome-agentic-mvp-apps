"""Session state: the only module that touches `st.session_state` by key.

Streamlit reruns the whole script on every interaction, so anything that must
survive a click lives here. Nothing here is persisted anywhere -- the uploaded
dataset lives in this process's memory for the life of one browser session and
is gone when it ends: no database, no disk, no logging of its content.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from app.core.config import Settings, get_settings
from app.models.schemas import CsvProfile, QuestionAnswer

_DF = "dataframe"
_PROFILE = "profile"
_FILENAME = "filename"
_HISTORY = "history"
_BUSY = "busy"


@dataclass(frozen=True)
class HistoryItem:
    """One asked question and its outcome.

    Attributes:
        question: What the user asked.
        answer: The finished answer, when the run succeeded.
        error: A message to show instead, when it did not.
    """

    question: str
    answer: QuestionAnswer | None = None
    error: str | None = None


def init() -> None:
    """Create every key this app reads, once per session."""
    defaults: dict[str, Any] = {
        _DF: None,
        _PROFILE: None,
        _FILENAME: None,
        _HISTORY: [],
        _BUSY: False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def settings() -> Settings:
    """Return the process-wide settings."""
    return get_settings()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def dataset() -> tuple[pd.DataFrame, CsvProfile, str] | None:
    """The loaded dataset, if one has been uploaded this session."""
    df = st.session_state[_DF]
    if df is None:
        return None
    return df, st.session_state[_PROFILE], st.session_state[_FILENAME]


def store_dataset(df: pd.DataFrame, profile: CsvProfile, filename: str) -> None:
    """Store a newly loaded dataset and clear any previous question history.

    Clearing history is the point: an answer left on screen beside a freshly
    uploaded, unrelated file would be the most misleading thing this app could
    show.
    """
    st.session_state[_DF] = df
    st.session_state[_PROFILE] = profile
    st.session_state[_FILENAME] = filename
    st.session_state[_HISTORY] = []


def clear_dataset() -> None:
    """Drop the dataset and return to the upload screen."""
    st.session_state[_DF] = None
    st.session_state[_PROFILE] = None
    st.session_state[_FILENAME] = None
    st.session_state[_HISTORY] = []


# --------------------------------------------------------------------------- #
# Question history
# --------------------------------------------------------------------------- #


def history() -> list[HistoryItem]:
    """Every question asked this session, oldest first."""
    return st.session_state[_HISTORY]


def append_history(item: HistoryItem) -> None:
    """Record one question's outcome."""
    st.session_state[_HISTORY].append(item)


# --------------------------------------------------------------------------- #
# Re-entrancy
# --------------------------------------------------------------------------- #


def busy() -> bool:
    """Whether a question is currently being answered.

    Streamlit queues interactions during a script run, so a second click while
    the first is still going arrives as a second run against the same session.
    The flag makes the second one a no-op rather than a duplicate paid call.
    """
    return bool(st.session_state[_BUSY])


def set_busy(value: bool) -> None:
    """Mark a run as started or finished."""
    st.session_state[_BUSY] = value
