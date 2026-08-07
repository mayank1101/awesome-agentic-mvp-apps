"""Entry point: `streamlit run streamlit_app.py`.

Kept thin on purpose -- this module owns the run order and nothing else. The
widgets live in :mod:`ui`, the model calls in :mod:`app.agents`.

One rerun proceeds as: draw the sidebar, accept a submitted brief, plan its
outline if one is queued, then render either the empty state or the document.
"""

import streamlit as st

from app.agents import generate_outline
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.guardrails import redact_secrets
from ui import state
from ui.document import render_document, render_empty_state
from ui.sidebar import BriefSubmission, render_sidebar

logger = get_logger(__name__)


def configure_page() -> None:
    """Apply page-level chrome. Must run before any other Streamlit call."""
    st.set_page_config(
        page_title="PRD Generator",
        page_icon=":material/description:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def handle_submission(submission: BriefSubmission) -> None:
    """Act on the sidebar form.

    Errors are shown in the main pane, where there is room for them. A valid
    brief is queued and the script rerun immediately, so the sidebar collapses
    and the spinner appears before the first model call.
    """
    if not submission.submitted:
        return

    if submission.error is not None:
        st.error(submission.error, icon=":material/error:")
        return

    if not state.can_start_generation():
        st.error(
            f"Session limit reached: {get_settings().max_generations_per_session} PRDs "
            "per session. Reload the page to start a new one.",
            icon=":material/hourglass_disabled:",
        )
        return

    assert submission.prd_input is not None  # guaranteed when error is None
    if submission.warning is not None:
        # Kept in session state: this run ends in a rerun, so a warning drawn
        # now would vanish before anyone read it.
        state.set_guardrail_warning(submission.warning)
    state.queue_generation(submission.prd_input)
    st.rerun()


def plan_pending_outline() -> None:
    """Plan the queued brief's outline, if there is one waiting.

    The brief is dequeued either way: on failure the error is shown and the user
    can resubmit, rather than the app retrying the same call on every rerun.
    """
    prd_input = state.get_pending_input()
    if prd_input is None or state.get_outline() is not None:
        return

    with st.spinner("Planning the outline..."):
        try:
            outline = generate_outline(prd_input)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            logger.exception("Outline generation failed")
            # Provider errors can echo request context; never render one raw.
            st.error(
                f"Outline generation failed: {redact_secrets(str(exc))}",
                icon=":material/error:",
            )
            outline = None

    state.clear_pending_input()
    if outline is not None:
        state.record_generation()
        state.start_document(prd_input, outline)
        st.rerun()


def main() -> None:
    """Render one Streamlit run."""
    configure_page()
    configure_logging()
    state.init_session_state()

    submission, slots = render_sidebar()
    handle_submission(submission)
    plan_pending_outline()

    warning = state.take_guardrail_warning()
    if warning is not None:
        st.warning(
            f"Guardrails flagged this brief but let it through: {warning}.",
            icon=":material/gpp_maybe:",
        )

    outline = state.get_outline()
    prd_input = state.get_prd_input()
    if outline is None or prd_input is None:
        render_empty_state()
    else:
        render_document(prd_input, outline, slots)


main()
