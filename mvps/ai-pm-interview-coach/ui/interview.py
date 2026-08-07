"""Screen 2: the interview loop.

One rerun draws the conversation from the projected transcript, plus the answer
submitted this run that has not been sent yet. That last part is the optimistic
render, and it is required rather than a nicety: an answer only enters the stored
conversation when the follow-up call's ``after_run`` fires, so between submitting
and the next question arriving it exists nowhere but ``st.session_state``.

There is no ticking timer and no deadline on this screen. An interview ends because
the candidate finished it or asked to stop; the conversation lives in
``st.session_state`` and goes away with the browser session.
"""

import streamlit as st

from app.agents.interview_agents import ask_followup, open_interview
from app.agents.prompts import format_answer_message
from app.core.config import get_settings
from app.models.schemas import CandidateAnswer, Transcript
from app.services.guardrails import has_severity, redact_secrets, scan_answer
from ui import state
from ui.state import LiveInterview


def render_interview() -> None:
    """Draw the interview screen and advance the conversation by at most one turn."""
    live = state.live_interview()
    transcript = live.transcript()

    _render_header(live, transcript)
    _render_conversation(transcript)
    _advance(live, transcript)


def _render_header(live: LiveInterview, transcript: Transcript) -> None:
    """Configuration restated compactly, plus where we are in the interview."""
    from app.agents import presets

    config = live.config
    interview = presets.INTERVIEW_TYPE_PRESETS[config.interview_type]
    seniority = presets.SENIORITY_PRESETS[config.seniority]
    archetype = presets.ARCHETYPE_PRESETS[config.archetype]

    asked = max(len(transcript.turns), 1)
    position = min(asked, config.total_questions)

    with st.container(
        horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"
    ):
        st.markdown(
            f":violet-badge[:material/psychology: {interview.label}] "
            f":blue-badge[:material/badge: {seniority.label}] "
            f":gray-badge[:material/domain: {archetype.label}]"
        )
        # A candidate needs to know how much runway is left. An interview with no
        # visible end is a different, worse experience.
        st.markdown(f"**Question {position} of {config.total_questions}**")

    st.divider()


def _render_conversation(transcript: Transcript) -> None:
    """Draw every turn, then the answer that has not been sent yet."""
    for turn in transcript.turns:
        with st.chat_message("assistant", avatar=":material/mic:"):
            st.markdown(turn.question)
        if turn.is_answered:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(turn.answer)

    pending = state.pending_answer()
    if pending is not None:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(pending)


def _advance(live: LiveInterview, transcript: Transcript) -> None:
    """Fetch the next question if one is due, otherwise take input or finish.

    The order of these checks is load-bearing, and getting it wrong is not subtle:
    fetching the follow-up before testing for completion asks one question past the
    budget -- a wasted model call, and a question the candidate can never answer,
    since the interview ends the moment the run finishes.

    So completion is judged against the answer count *including* the one still
    pending, because an answer submitted this run has not reached the stored
    conversation yet.
    """
    error = state.turn_error()
    if error is not None:
        st.error(error, icon=":material/error:")
        if st.button("Retry", icon=":material/refresh:", type="primary"):
            state.set_turn_error(None)
            st.rerun()
        _render_end_button(disabled=False)
        return

    pending = state.pending_answer()

    if not transcript.turns:
        _stream_next_question(live, None)
        return

    answered = transcript.answered_turns + (1 if pending is not None else 0)
    if answered >= live.config.total_questions:
        # The pending answer has no follow-up to ride along with, so it is written
        # to history directly or it would never be graded.
        if pending is not None:
            state.record_final_answer(format_answer_message(pending))
            state.clear_pending_answer()
        state.begin_ending("complete")
        st.rerun()
        return

    if pending is not None:
        _stream_next_question(live, pending)
        return

    _render_input()


def _stream_next_question(live: LiveInterview, pending: str | None) -> None:
    """Stream one question into the chat, then rerun to redraw from the transcript."""
    with st.chat_message("assistant", avatar=":material/mic:"):
        try:
            if pending is None:
                st.write_stream(open_interview(live))
            else:
                st.write_stream(ask_followup(live, pending))
        except Exception as exc:  # noqa: BLE001 - surfaced to the candidate, not swallowed
            # The answer stays in pending_answer, so Retry resends the same text
            # and nothing is lost or double-stored.
            state.set_turn_error(
                f"That turn failed: {redact_secrets(str(exc))}. Your answer is still here."
            )
            st.rerun()
            return

    state.clear_pending_answer()
    st.rerun()


def _render_input() -> None:
    """Draw the answer box and the end button."""
    settings = get_settings()

    answer = st.chat_input(
        "Type your answer…",
        max_chars=settings.max_answer_chars,
    )
    _render_end_button(disabled=False)
    st.caption(
        f"Short answers score low — aim for a paragraph. Limit "
        f"{settings.max_answer_chars:,} characters."
    )

    if not answer:
        return

    # A stray keystroke must not consume one of four follow-ups.
    if not answer.strip():
        return

    try:
        validated = CandidateAnswer(text=answer)
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
        return

    if settings.guardrails_enabled:
        findings = scan_answer(validated.stripped)
        if findings and settings.block_flagged_input and has_severity(findings, "high"):
            state.set_guardrail_note(
                f"{findings[0].message} — the turn was not sent. Answer the question instead."
            )
            st.rerun()
            return
        if findings:
            state.set_guardrail_note(f"{findings[0].message} — sent anyway.")

    state.set_pending_answer(validated.stripped)
    st.rerun()


def _render_end_button(*, disabled: bool) -> None:
    """The end button, reachable for the whole interview.

    Next to the input rather than in the sidebar, which is collapsed during the
    interview. Ending early is a first-class path, not an escape hatch.
    """
    if st.button(
        "End session and grade",
        icon=":material/stop_circle:",
        disabled=disabled,
        help="Grades what you have answered so far.",
    ):
        # The phase moves before any grading call, which is what makes a
        # double-click idempotent.
        state.begin_ending("manual")
        st.rerun()
