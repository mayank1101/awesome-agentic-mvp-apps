"""Entry point: `streamlit run streamlit_app.py`.

Kept thin on purpose -- this module owns the run order and nothing else. The
widgets live in :mod:`ui`, the single model call in :mod:`app.agents`, and every
number in :mod:`app.services.scoring`.

One rerun proceeds as: bridge secrets, accept a submitted backlog, estimate it if
one is queued, then render either the empty state or the ranking. Scoring happens
on every rerun regardless -- it is pure arithmetic, and recomputing it is how the
table can never disagree with the factors beneath it.
"""

import os

import streamlit as st


def bridge_secrets() -> None:
    """Copy `st.secrets` into the environment, before any settings are read.

    Streamlit Community Cloud has no `.env`: secrets arrive through
    `st.secrets`. Bridging them here keeps `app/core/config.py` a plain
    pydantic-settings class with no Streamlit import, which is what lets the
    whole `app` package be tested without a browser.

    `setdefault` rather than assignment, so a real environment variable wins --
    a container passing `--env-file` should not be overridden by a stale entry
    in a committed secrets file.

    The whole iteration sits inside the `try`, not just the attribute access:
    `st.secrets` is lazy, so touching it succeeds and `.items()` is what raises
    `StreamlitSecretNotFoundError`. Guarding only the attribute lookup crashes
    the app on every machine that has no secrets file -- which is every
    developer's, since local runs use `.env`.
    """
    try:
        items = list(st.secrets.items())
    except Exception:  # noqa: BLE001 - no secrets file locally is the normal case
        return
    for key, value in items:
        if isinstance(value, str):
            os.environ.setdefault(key, value)


bridge_secrets()

from app.agents import estimate_backlog  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.services.guardrails import redact_secrets  # noqa: E402
from ui import state  # noqa: E402
from ui.input_form import Submission, render_empty_state, render_input_form  # noqa: E402
from ui.results import build_ranking, collect_editor_overrides, render_results  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402

logger = get_logger(__name__)


def configure_page() -> None:
    """Apply page-level chrome. Must run before any other Streamlit call."""
    st.set_page_config(
        page_title="Feature Prioritisation Assistant",
        page_icon=":material/leaderboard:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def handle_submission(submission: Submission) -> None:
    """Act on the input form.

    A valid backlog is queued and the script rerun immediately, so the form
    disappears and the spinner appears before the model call starts.
    """
    if not submission.submitted:
        return

    if submission.error is not None:
        st.error(submission.error, icon=":material/error:")
        return

    if not state.can_start_estimation():
        st.error(
            f"Session limit reached: {get_settings().max_estimations_per_session} estimates per "
            "session. Reload the page to start a new one. Editing factors is never limited — it "
            "calls no model.",
            icon=":material/hourglass_disabled:",
        )
        return

    assert submission.backlog is not None  # guaranteed when error is None
    if submission.warning is not None:
        # Kept in session state: this run ends in a rerun, so a warning drawn
        # now would vanish before anyone read it.
        state.set_guardrail_warning(submission.warning)
    state.queue_estimation(submission.backlog)
    st.rerun()


def estimate_pending_backlog() -> None:
    """Estimate the queued backlog, if there is one waiting.

    The backlog is dequeued either way: on failure the error is shown and the
    user can resubmit, rather than the app retrying the same call on every rerun.
    """
    backlog = state.get_pending_backlog()
    if backlog is None or state.get_estimate() is not None:
        return

    with st.spinner(f"Estimating {len(backlog.features)} features in one pass..."):
        try:
            estimate = estimate_backlog(backlog)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            logger.exception("Estimation failed")
            # Provider errors can echo request context; never render one raw.
            state.set_estimation_error(redact_secrets(str(exc)))
            estimate = None

    state.clear_pending_backlog()
    if estimate is not None:
        state.record_estimation()
        state.start_ranking(backlog, estimate)
    # Rerun either way: this run drew a spinner and nothing else, so on failure
    # it has no form for the user to retry from.
    st.rerun()


def main() -> None:
    """Render one Streamlit run."""
    configure_page()
    configure_logging()
    state.init_session_state()

    backlog = state.get_backlog()
    estimate = state.get_estimate()

    ranked = None
    if backlog is not None and estimate is not None:
        ranked = build_ranking(backlog, estimate)
        # Editor deltas are read before anything is drawn, so an edit and the
        # re-ranked table land in the same interaction. Scoring twice is free.
        if collect_editor_overrides(backlog, ranked):
            ranked = build_ranking(backlog, estimate)

    render_sidebar(backlog, ranked)

    failure = state.take_estimation_error()
    if failure is not None:
        st.error(f"Estimation failed: {failure}", icon=":material/error:")

    warning = state.take_guardrail_warning()
    if warning is not None:
        st.warning(
            f"Guardrails flagged this backlog but let it through — {warning}. The notes were sent "
            "as data, not as instructions.",
            icon=":material/gpp_maybe:",
        )

    if ranked is None:
        # While a backlog is queued, the form is deliberately *not* drawn: the
        # run ends in a spinner and the model call, and leaving an editable form
        # on screen invites edits that the in-flight request cannot see.
        if state.get_pending_backlog() is not None:
            estimate_pending_backlog()
        else:
            handle_submission(render_input_form())
            render_empty_state()
    else:
        render_results(backlog, ranked)


main()
