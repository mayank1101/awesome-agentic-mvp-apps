"""Entry point: `streamlit run streamlit_app.py`.

Kept thin on purpose -- this module owns the run order and nothing else. The
widgets live in :mod:`ui`, the model calls in :mod:`app.agents`, and the
conversation in ``st.session_state`` -- see :mod:`ui.state`.

One rerun proceeds as: bridge secrets into the environment, draw the sidebar, act
on a start request, then render whichever screen the phase selects. There is no
background work and nothing to clean up -- the conversation lives and dies with
the browser session.
"""

import os

import streamlit as st


def configure_page() -> None:
    """Apply page-level chrome. Must run before any other Streamlit call."""
    st.set_page_config(
        page_title="AI PM Interview Coach",
        page_icon=":material/record_voice_over:",
        layout="centered",
        initial_sidebar_state="expanded",
    )


def load_secrets_into_env() -> None:
    """Bridge st.secrets into the environment before settings are first read.

    Streamlit Community Cloud injects secrets through ``st.secrets`` and provides
    no ``.env`` file, while ``pydantic-settings`` reads the environment and
    ``app/`` must not import Streamlit. This module already depends on both, so
    the bridge belongs here and nowhere else.

    Uses ``setdefault``, so a real environment variable always wins over a secrets
    entry -- which is what makes local runs behave predictably.

    Tolerates the absence of a secrets store: reading ``st.secrets`` raises when
    no secrets file exists, which is the normal local case, so the failure is
    caught and the environment is left as it was.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file is the normal local case
        return

    for key, value in secrets.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(str(key), str(value))


configure_page()
load_secrets_into_env()

# Imported after the bridge, so the first get_settings() call sees the secrets.
from app.core.logging import configure_logging, get_logger  # noqa: E402
from ui import interview, report, sidebar, state  # noqa: E402

logger = get_logger(__name__)


def handle_start(request: sidebar.StartRequest) -> None:
    """Act on the sidebar's start button."""
    if not request.submitted:
        return

    if request.error is not None:
        st.error(request.error, icon=":material/error:")
        return

    assert request.config is not None  # guaranteed when error is None
    state.begin_interview(request.config)
    st.rerun()


def render_current_phase() -> None:
    """Draw the one screen this phase calls for.

    Every phase has a defined screen, including the failure ones, so there is no
    state in which two are half-drawn or none is.
    """
    phase = state.phase()

    if phase == "idle":
        sidebar.render_brief()
        return

    if phase == "interviewing":
        # A restart clears session state, so a browser can arrive here with the
        # phase set and no conversation behind it. That resolves to the ended
        # screen, never a stack trace.
        if not state.has_conversation():
            state.expire("gone")
            st.rerun()
            return
        interview.render_interview()
        return

    if phase == "ending":
        if state.turn_error() is not None:
            report.render_grading_failure()
        else:
            report.render_grading()
        return

    if phase == "reported":
        current = state.report()
        config = state.config()
        if current is None or config is None:
            state.reset_to_idle()
            st.rerun()
            return
        report.render_report(current, config)
        return

    report.render_expired()


def main() -> None:
    """Render one Streamlit run."""
    configure_logging()
    state.init_session_state()

    request = sidebar.render_sidebar()
    handle_start(request)

    note = state.take_guardrail_note()
    if note is not None:
        st.warning(f"Guardrails: {note}", icon=":material/gpp_maybe:")

    render_current_phase()


main()
