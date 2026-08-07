"""Session state: the keys the app keeps between Streamlit reruns.

Streamlit re-executes the whole script on every interaction, so anything that
must survive a click lives in ``st.session_state``. Centralising the keys here
keeps their names and lifecycles in one place instead of scattered string
literals.

The state machine is small:

===================  =========================================================
``pending_input``    A brief accepted this run, whose outline is not planned yet
``prd_input``        The brief the current document was generated from
``outline``          The plan; its presence is what switches the main pane from
                     the empty state to the document
``sections``         One slot per outline entry: the written section, or None
``brief_expanded``   Whether the sidebar form starts open
``generations``      How many PRDs this session has started, for the cost guard
``guardrail_note``   A warning to show once, on the run after it was raised
===================  =========================================================
"""

import streamlit as st

from app.core.config import get_settings
from app.models.schemas import PRDInput, PRDOutline, PRDSection

PENDING_INPUT = "pending_input"
PRD_INPUT = "prd_input"
OUTLINE = "outline"
SECTIONS = "sections"
BRIEF_EXPANDED = "brief_expanded"
GENERATIONS = "generations"
GUARDRAIL_NOTE = "guardrail_note"


def init_session_state() -> None:
    """Seed every key this app reads. Safe to call on each rerun."""
    st.session_state.setdefault(PRD_INPUT, None)
    st.session_state.setdefault(OUTLINE, None)
    st.session_state.setdefault(SECTIONS, None)
    st.session_state.setdefault(PENDING_INPUT, None)
    st.session_state.setdefault(BRIEF_EXPANDED, True)
    st.session_state.setdefault(GENERATIONS, 0)
    st.session_state.setdefault(GUARDRAIL_NOTE, None)


def get_pending_input() -> PRDInput | None:
    """The brief awaiting outline planning, if any."""
    return st.session_state[PENDING_INPUT]


def get_prd_input() -> PRDInput | None:
    """The brief the current document was generated from."""
    return st.session_state[PRD_INPUT]


def get_outline() -> PRDOutline | None:
    """The current outline, or None when no PRD has been planned yet."""
    return st.session_state[OUTLINE]


def get_sections() -> list[PRDSection | None]:
    """Written sections, positionally aligned with the outline's sections."""
    return st.session_state[SECTIONS]


def set_section(index: int, section: PRDSection) -> None:
    """Store one freshly written section."""
    st.session_state[SECTIONS][index] = section


def is_brief_expanded() -> bool:
    """Whether the sidebar's brief form should render expanded."""
    return st.session_state[BRIEF_EXPANDED]


def queue_generation(prd_input: PRDInput) -> None:
    """Accept a brief and clear the previous document.

    The outline is planned on the *next* run rather than this one, so the
    sidebar can collapse and the spinner can appear before the model is called.
    """
    st.session_state[PENDING_INPUT] = prd_input
    st.session_state[OUTLINE] = None
    st.session_state[SECTIONS] = None
    st.session_state[BRIEF_EXPANDED] = False


def start_document(prd_input: PRDInput, outline: PRDOutline) -> None:
    """Adopt a planned outline and reserve an empty slot per section."""
    st.session_state[PRD_INPUT] = prd_input
    st.session_state[OUTLINE] = outline
    st.session_state[SECTIONS] = [None] * len(outline.sections)


def clear_pending_input() -> None:
    """Drop the queued brief, whether its outline succeeded or failed."""
    st.session_state[PENDING_INPUT] = None


def can_start_generation() -> bool:
    """Whether this session is still under the per-session generation cap.

    A cost guard rather than a security control -- reloading the page starts a
    fresh session. Disabled when `MAX_GENERATIONS_PER_SESSION` is 0.
    """
    limit = get_settings().max_generations_per_session
    return limit <= 0 or st.session_state[GENERATIONS] < limit


def record_generation() -> None:
    """Count one PRD against the session cap.

    Called once the outline lands, not on submit: a brief that failed to plan
    cost nothing worth counting, and charging for it would punish a rate-limited
    free-tier retry.
    """
    st.session_state[GENERATIONS] += 1


def set_guardrail_warning(message: str) -> None:
    """Stash a guardrail note for the next run to display."""
    st.session_state[GUARDRAIL_NOTE] = message


def take_guardrail_warning() -> str | None:
    """Return the pending guardrail note, clearing it so it shows only once."""
    message = st.session_state[GUARDRAIL_NOTE]
    st.session_state[GUARDRAIL_NOTE] = None
    return message
