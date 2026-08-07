"""Session state: the keys the app keeps between Streamlit reruns.

Streamlit re-executes the whole script on every interaction, so anything that
must survive a click lives in ``st.session_state``. Centralising the keys here
keeps their names and lifecycles in one place instead of scattered string
literals.

The state machine is small:

===================  =========================================================
``pending_backlog``  A backlog accepted this run, not yet estimated
``backlog``          The backlog the current ranking was built from
``estimate``         The estimator's reply -- the *only* thing a model produced
``overrides``        Per-feature factor edits, keyed by feature id
``run``              Increments per estimation, to reset the editor widget
``estimations``      How many estimation calls this session made, for the cost guard
``guardrail_note``   A warning to show once, on the run after it was raised
``estimation_error`` A failed call's message, shown once alongside the form
===================  =========================================================

The important property: ``estimate`` and ``overrides`` are what persist, not the
ranking. The ranking is recomputed from them on every rerun, which costs
microseconds and means there is no cached score that can drift out of step with
the factors on screen.
"""

import streamlit as st

from app.core.config import get_settings
from app.models.schemas import BacklogEstimate, BacklogInput

PENDING_BACKLOG = "pending_backlog"
BACKLOG = "backlog"
ESTIMATE = "estimate"
OVERRIDES = "overrides"
RUN = "run"
ESTIMATIONS = "estimations"
GUARDRAIL_NOTE = "guardrail_note"
ESTIMATION_ERROR = "estimation_error"


def init_session_state() -> None:
    """Seed every key this app reads. Safe to call on each rerun."""
    st.session_state.setdefault(PENDING_BACKLOG, None)
    st.session_state.setdefault(BACKLOG, None)
    st.session_state.setdefault(ESTIMATE, None)
    st.session_state.setdefault(OVERRIDES, {})
    st.session_state.setdefault(RUN, 0)
    st.session_state.setdefault(ESTIMATIONS, 0)
    st.session_state.setdefault(GUARDRAIL_NOTE, None)
    st.session_state.setdefault(ESTIMATION_ERROR, None)


def get_pending_backlog() -> BacklogInput | None:
    """The backlog awaiting estimation, if any."""
    return st.session_state[PENDING_BACKLOG]


def get_backlog() -> BacklogInput | None:
    """The backlog the current ranking was built from."""
    return st.session_state[BACKLOG]


def get_estimate() -> BacklogEstimate | None:
    """The estimator's reply, or None before the first run."""
    return st.session_state[ESTIMATE]


def get_overrides() -> dict[str, dict[str, float]]:
    """User factor edits, keyed by feature id then factor name."""
    return st.session_state[OVERRIDES]


def set_override(feature_id: str, factor: str, value: float) -> None:
    """Record one factor edit."""
    st.session_state[OVERRIDES].setdefault(feature_id, {})[factor] = value


def clear_overrides() -> None:
    """Drop every edit, returning the table to the estimator's own numbers."""
    st.session_state[OVERRIDES] = {}
    # Bump the run counter so the editor widget is rebuilt from scratch; without
    # this its own cached edits would immediately re-apply what was just cleared.
    st.session_state[RUN] += 1


def editor_key() -> str:
    """Widget key for the factor editor, unique per estimation run."""
    return f"factor_editor_{st.session_state[RUN]}"


def queue_estimation(backlog: BacklogInput) -> None:
    """Accept a backlog and clear the previous ranking.

    The model is called on the *next* run rather than this one, so the spinner
    is on screen before the request starts.
    """
    st.session_state[PENDING_BACKLOG] = backlog
    st.session_state[BACKLOG] = None
    st.session_state[ESTIMATE] = None
    st.session_state[OVERRIDES] = {}


def clear_pending_backlog() -> None:
    """Drop the queued backlog, whether its estimation succeeded or failed."""
    st.session_state[PENDING_BACKLOG] = None


def start_ranking(backlog: BacklogInput, estimate: BacklogEstimate) -> None:
    """Adopt a fresh estimate as the current ranking's source."""
    st.session_state[BACKLOG] = backlog
    st.session_state[ESTIMATE] = estimate
    st.session_state[OVERRIDES] = {}
    st.session_state[RUN] += 1


def reset() -> None:
    """Return to the empty state, keeping the session's estimation count."""
    st.session_state[PENDING_BACKLOG] = None
    st.session_state[BACKLOG] = None
    st.session_state[ESTIMATE] = None
    st.session_state[OVERRIDES] = {}
    st.session_state[RUN] += 1


def can_start_estimation() -> bool:
    """Whether this session is still under the per-session estimation cap.

    A cost guard rather than a security control -- reloading the page starts a
    fresh session. Disabled when `MAX_ESTIMATIONS_PER_SESSION` is 0. Editing
    factors is never checked against this, because editing calls no model.
    """
    limit = get_settings().max_estimations_per_session
    return limit <= 0 or st.session_state[ESTIMATIONS] < limit


def record_estimation() -> None:
    """Count one estimation against the session cap.

    Called once the reply lands, not on submit: a call that failed to parse cost
    the user a ranking already, and charging for it would punish a rate-limited
    free-tier retry.
    """
    st.session_state[ESTIMATIONS] += 1


def set_estimation_error(message: str) -> None:
    """Stash a failed call's message for the next run to display.

    The failure and the form the user needs to retry from are drawn on different
    runs: the run that fails is the run that was showing a spinner instead of the
    form. Without this the error would be rendered onto a page with nothing to
    act on, and then vanish.
    """
    st.session_state[ESTIMATION_ERROR] = message


def take_estimation_error() -> str | None:
    """Return the pending failure message, clearing it so it shows only once."""
    message = st.session_state[ESTIMATION_ERROR]
    st.session_state[ESTIMATION_ERROR] = None
    return message


def set_guardrail_warning(message: str) -> None:
    """Stash a guardrail note for the next run to display."""
    st.session_state[GUARDRAIL_NOTE] = message


def take_guardrail_warning() -> str | None:
    """Return the pending guardrail note, clearing it so it shows only once."""
    message = st.session_state[GUARDRAIL_NOTE]
    st.session_state[GUARDRAIL_NOTE] = None
    return message
