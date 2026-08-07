"""Session state: everything the app keeps between Streamlit reruns.

Streamlit re-executes the whole script on every interaction, so anything that must
survive a click lives in ``st.session_state``. Centralising the keys here keeps
their names and lifecycles in one place instead of scattered string literals.

**The conversation lives here too**, in the framework's ``AgentSession`` -- whose
``state`` holds the messages as a plain list -- under one session-state key. There
is no server-side store, and that is the whole design:

* ``st.session_state`` is already per-browser-session, so one candidate cannot
  reach another's conversation. No shared dict, no unguessable handles, no
  capacity limit to tune, no storage service to run.
* Starting an interview builds a fresh ``AgentSession``, so the message list begins
  empty every time. Restarting the app has the same effect for the same reason --
  nothing was written anywhere that survives the process, so there is nothing to
  clean up and no deletion step to get wrong.

The cost is the honest one: a conversation is process-local and lasts as long as
the browser session. A restart or a closed tab ends an interview in progress, which
for a practice tool is what the candidate already expects.

The phase machine is small and one-directional:

===============  =========================================================
``idle``         No live interview. Screen 1, the brief.
``interviewing`` A conversation exists and the loop is running. Screen 2.
``ending``       Grading, or grading failed and a retry is still allowed.
``reported``     A validated report is on screen. Screen 3.
``expired``      The interview ended with nothing to grade.
===============  =========================================================
"""

from dataclasses import dataclass
from typing import Any

import streamlit as st

from app.core.config import get_settings
from app.models.schemas import FeedbackReport, InterviewConfig, SessionPhase, Transcript
from app.services.transcript import (
    append_answer_to_state,
    messages_from_state,
    to_transcript,
)

PHASE = "phase"
CONFIG = "config"
AGENT_SESSION = "agent_session"
PENDING_ANSWER = "pending_answer"
REPORT = "report"
INTERVIEWS_STARTED = "interviews_started"
GUARDRAIL_NOTE = "guardrail_note"
TURN_ERROR = "turn_error"
ENDED_REASON = "ended_reason"


def init_session_state() -> None:
    """Seed every key this app reads. Safe to call on each rerun."""
    st.session_state.setdefault(PHASE, "idle")
    st.session_state.setdefault(CONFIG, None)
    st.session_state.setdefault(AGENT_SESSION, None)
    st.session_state.setdefault(PENDING_ANSWER, None)
    st.session_state.setdefault(REPORT, None)
    st.session_state.setdefault(INTERVIEWS_STARTED, 0)
    st.session_state.setdefault(GUARDRAIL_NOTE, None)
    st.session_state.setdefault(TURN_ERROR, None)
    st.session_state.setdefault(ENDED_REASON, None)


# --- phase -------------------------------------------------------------------


def phase() -> SessionPhase:
    """Which screen this browser session is on."""
    return st.session_state[PHASE]


def begin_interview(config: InterviewConfig) -> None:
    """Start a conversation and move to the interview screen.

    The fresh ``AgentSession`` is what resets the message list -- there is no
    separate clearing step, because a new session has nothing in it.
    """
    from app.agents.interview_agents import new_agent_session

    st.session_state[CONFIG] = config
    st.session_state[AGENT_SESSION] = new_agent_session()
    st.session_state[PHASE] = "interviewing"
    st.session_state[PENDING_ANSWER] = None
    st.session_state[REPORT] = None
    st.session_state[TURN_ERROR] = None
    st.session_state[ENDED_REASON] = None
    st.session_state[INTERVIEWS_STARTED] += 1


def begin_ending(reason: str) -> None:
    """Move to the grading screen.

    The transition happens *before* the grading call, which is what makes a
    double-clicked End button idempotent: the second click arrives with the phase
    already ``ending`` and is ignored rather than starting a second report.
    """
    st.session_state[PHASE] = "ending"
    st.session_state[ENDED_REASON] = reason


def finish_with_report(report: FeedbackReport) -> None:
    """Adopt a validated report.

    The conversation stays reachable until the next interview replaces it, which is
    what lets the report quote it. It never outlives the browser session.
    """
    st.session_state[REPORT] = report
    st.session_state[PHASE] = "reported"
    st.session_state[PENDING_ANSWER] = None


def expire(reason: str) -> None:
    """Explain an interview that ended with nothing to grade.

    Never a crash: arriving here with no conversation is an expected state, not an
    error.
    """
    st.session_state[PHASE] = "expired"
    st.session_state[AGENT_SESSION] = None
    st.session_state[PENDING_ANSWER] = None
    st.session_state[ENDED_REASON] = reason


def reset_to_idle() -> None:
    """Return to the brief, discarding the report.

    The one place the report is thrown away on purpose -- it is the only artifact
    that outlives the conversation, so nothing else may drop it silently.
    """
    st.session_state[PHASE] = "idle"
    st.session_state[CONFIG] = None
    st.session_state[AGENT_SESSION] = None
    st.session_state[REPORT] = None
    st.session_state[PENDING_ANSWER] = None
    st.session_state[TURN_ERROR] = None
    st.session_state[ENDED_REASON] = None


# --- the conversation --------------------------------------------------------


@dataclass
class LiveInterview:
    """The configuration and conversation of the interview in progress.

    Exists because :mod:`app.agents.interview_agents` needs ``config``,
    ``agent_session`` and ``transcript()`` from one object, and reads nothing else.
    Passing this rather than reaching into ``st.session_state`` from ``app/`` is what
    keeps the dependency one-way: ``ui`` imports ``app``, never the reverse.
    """

    config: InterviewConfig
    agent_session: Any

    @property
    def messages(self) -> list[Any]:
        """The stored conversation, as the framework holds it."""
        return messages_from_state(getattr(self.agent_session, "state", None))

    def transcript(self) -> Transcript:
        """The conversation paired into question-and-answer turns."""
        return to_transcript(self.config, self.messages)


def has_conversation() -> bool:
    """Whether a live conversation exists to render or grade."""
    return st.session_state[AGENT_SESSION] is not None and st.session_state[CONFIG] is not None


def live_interview() -> LiveInterview:
    """The interview in progress.

    Callers must check :func:`has_conversation` first -- this is a plain read and
    does not invent a conversation to hand back.
    """
    return LiveInterview(
        config=st.session_state[CONFIG],
        agent_session=st.session_state[AGENT_SESSION],
    )


def record_final_answer(fenced_answer: str) -> bool:
    """Store the last answer of an interview, which has no follow-up to carry it.

    Every other answer is persisted by the framework as the input of the next
    follow-up call. The final one has no next call, so it is written directly --
    otherwise finishing the budget would either lose the last answer or force a
    wasted question to be asked purely to store it.

    Args:
        fenced_answer: The answer, already fenced, so the stored form matches what
            the framework would have written.

    Returns:
        Whether it was stored.
    """
    return append_answer_to_state(
        getattr(st.session_state[AGENT_SESSION], "state", None), fenced_answer
    )


# --- accessors ---------------------------------------------------------------


def config() -> InterviewConfig | None:
    """The configuration the current or most recent interview ran under."""
    return st.session_state[CONFIG]


def report() -> FeedbackReport | None:
    """The graded report, once one exists."""
    return st.session_state[REPORT]


def ended_reason() -> str | None:
    """Why the interview ended, for the wording on screen 3 and the expiry state."""
    return st.session_state[ENDED_REASON]


# --- the pending answer ------------------------------------------------------


def set_pending_answer(answer: str) -> None:
    """Hold an answer submitted this run, whose follow-up is fetched on the next.

    This is what makes the optimistic render possible. An answer only enters the
    stored conversation when the follow-up call's ``after_run`` fires, so between
    submit and the next question it exists nowhere else -- and if that call fails,
    this is what the retry resends.
    """
    st.session_state[PENDING_ANSWER] = answer


def pending_answer() -> str | None:
    """The answer awaiting a follow-up, if any."""
    return st.session_state[PENDING_ANSWER]


def clear_pending_answer() -> None:
    """Drop the queued answer once its follow-up has landed."""
    st.session_state[PENDING_ANSWER] = None


# --- transient notices -------------------------------------------------------


def set_guardrail_note(message: str) -> None:
    """Stash a guardrail warning for the next run to display."""
    st.session_state[GUARDRAIL_NOTE] = message


def take_guardrail_note() -> str | None:
    """Return the pending guardrail note, clearing it so it shows only once."""
    message = st.session_state[GUARDRAIL_NOTE]
    st.session_state[GUARDRAIL_NOTE] = None
    return message


def set_turn_error(message: str | None) -> None:
    """Record a turn-level failure, so the retry affordance can be drawn."""
    st.session_state[TURN_ERROR] = message


def turn_error() -> str | None:
    """The current turn-level failure, if the last call failed."""
    return st.session_state[TURN_ERROR]


# --- the cost guard ----------------------------------------------------------


def can_start_interview() -> bool:
    """Whether this browser session is still under the per-session interview cap.

    A cost guard rather than a security control -- reloading the page starts a
    fresh session, and a restart resets the counter. Disabled when
    ``MAX_INTERVIEWS_PER_SESSION`` is 0.
    """
    limit = get_settings().max_interviews_per_session
    return limit <= 0 or st.session_state[INTERVIEWS_STARTED] < limit


def interviews_started() -> int:
    """How many interviews this browser session has started."""
    return st.session_state[INTERVIEWS_STARTED]
