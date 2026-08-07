"""Screen 1: the configuration form and the brief.

The sidebar holds the four configuration axes and the start button; the main pane
explains what the session is and what it grades.

Two things in the main pane are there deliberately rather than as filler:

* **The rubric, before you start.** Rendered from `rubric.py`, so it cannot drift
  from what the evaluator actually applies. A candidate who knows they are scored
  on counter-metrics behaves differently, and that is coaching rather than
  cheating -- it is the point of a practice tool.
* **The deletion behaviour, stated up front.** Someone who loses a transcript they
  assumed was saved has a legitimate complaint. Saying it before the first click
  removes that entirely.
"""

from dataclasses import dataclass

import streamlit as st

from app.agents import presets, rubric
from app.agents.client import preflight
from app.core.config import get_settings
from app.core.exceptions import PreflightError
from app.models.schemas import (
    CompanyArchetype,
    InterviewConfig,
    InterviewType,
    Seniority,
)
from ui import state


@dataclass(frozen=True)
class StartRequest:
    """What the sidebar reported this run.

    Attributes:
        submitted: Whether the start button was pressed.
        config: The validated configuration, when there is one.
        error: A message to show instead of starting.
    """

    submitted: bool = False
    config: InterviewConfig | None = None
    error: str | None = None


def _interview_type_labels() -> dict[str, InterviewType]:
    return {
        presets.INTERVIEW_TYPE_PRESETS[key].label: key
        for key in InterviewType.__args__  # type: ignore[attr-defined]
    }


def _seniority_labels() -> dict[str, Seniority]:
    return {
        presets.SENIORITY_PRESETS[key].label: key
        for key in Seniority.__args__  # type: ignore[attr-defined]
    }


def _archetype_labels() -> dict[str, CompanyArchetype]:
    return {
        presets.ARCHETYPE_PRESETS[key].label: key
        for key in CompanyArchetype.__args__  # type: ignore[attr-defined]
    }


def render_sidebar() -> StartRequest:
    """Draw the sidebar and report whether an interview should start."""
    settings = get_settings()

    with st.sidebar:
        st.markdown("### AI PM Interview Coach")
        st.caption("A mock interview that probes, then grades against a fixed rubric.")

        live = state.phase() == "interviewing"

        types = _interview_type_labels()
        seniorities = _seniority_labels()
        archetypes = _archetype_labels()

        type_label = st.selectbox(
            "Interview type",
            list(types),
            disabled=live,
            help="Drives the shape of the question and which rubric dimensions it can test.",
        )
        seniority_label = st.selectbox(
            "Seniority",
            list(seniorities),
            index=1,
            disabled=live,
            help="The bar your answer is held to.",
        )
        archetype_label = st.selectbox(
            "Company archetype",
            list(archetypes),
            disabled=live,
            help="Decides what a good answer optimises for -- the same answer is strong at one and weak at another.",
        )
        focus_area = st.text_input(
            "Focus area (optional)",
            max_chars=120,
            placeholder="payments, developer tools, …",
            disabled=live,
            help="Leave empty and the interviewer picks a domain itself.",
        )
        budget = st.radio(
            "Follow-ups",
            options=[2, 4, 6],
            index=1,
            horizontal=True,
            disabled=live,
            help="Questions after the opener. Total is this plus one.",
        )

        st.divider()

        if live:
            st.info("Interview in progress.", icon=":material/mic:")
            submitted = False
        else:
            submitted = st.button(
                "Start session",
                type="primary",
                width="stretch",
                icon=":material/play_arrow:",
                disabled=not state.can_start_interview(),
            )

        st.caption(f"Model: `{settings.model_provider}` · `{settings.model_name}`")
        if settings.max_interviews_per_session > 0:
            st.caption(
                f"Sessions used: {state.interviews_started()} / "
                f"{settings.max_interviews_per_session}"
            )

    if not submitted:
        return StartRequest()

    if not state.can_start_interview():
        return StartRequest(
            submitted=True,
            error=(
                f"Session limit reached: {settings.max_interviews_per_session} interviews "
                "per browser session. Reload the page to start over."
            ),
        )

    # Preflight before minting a session. Failing on the opening question instead
    # would already have counted a session and put the UI into the interview
    # phase, showing a broken interview rather than a fixable configuration error.
    try:
        preflight()
    except PreflightError as exc:
        return StartRequest(submitted=True, error=str(exc))

    config = InterviewConfig(
        interview_type=types[type_label],
        seniority=seniorities[seniority_label],
        archetype=archetypes[archetype_label],
        focus_area=focus_area.strip() or None,
        followup_budget=budget,
    )
    return StartRequest(submitted=True, config=config)


def render_brief() -> None:
    """Draw the main pane for the idle phase."""
    st.title("Practise the interview, not the framework")
    st.markdown(
        "A PM interviewer asks you one question, then probes what you actually said "
        "— three to seven turns. At the end you get scored against a five-dimension "
        "rubric, with the moment that earned each score quoted back to you."
    )

    with st.expander("What you'll be graded on", icon=":material/checklist:"):
        st.caption(
            "Four points, no middle: 1–2 is below the bar, 3–4 is at or above it. "
            "Every dimension is a decision, not a hedge."
        )
        for dimension in rubric.RUBRIC:
            st.markdown(f"**{dimension.label}** — {dimension.question}")
            st.markdown(
                "\n".join(
                    f"- `{score}` {text}" for score, text in sorted(dimension.anchors.items())
                )
            )

    with st.expander("How the session ends", icon=":material/timer:"):
        st.markdown(
            """
End it yourself with **End session**, or run out the follow-ups. Either way the
interview is graded.

Nothing is written to a database or a disk. The conversation lives in this browser
session only, so closing the tab or restarting the app ends an interview in
progress — and leaves nothing behind.

The report stays on this page until you start another session or close the tab.
It quotes what you said, because that is what makes the scores arguable. Nothing
is kept between sessions.
"""
        )
