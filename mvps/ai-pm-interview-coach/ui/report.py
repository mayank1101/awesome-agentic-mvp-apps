"""Screen 3: grading, the scored report, and the expiry state.

Three phases share this module because they are three states of one screen, and
rendering them together is what stops the transition from flashing a blank page.

The score row leads, because it is what a candidate came for. Each row is a
dimension, a four-segment meter, the number, and the one-line justification — four
segments because the scale has four points, where a percentage bar would imply a
precision the rubric does not have.
"""

import streamlit as st

from app.agents import presets, rubric
from app.core.exceptions import EmptyInterviewError
from app.core.logging import get_logger
from app.models.schemas import FeedbackReport, InterviewConfig, RubricDimension
from app.services.guardrails import redact_secrets, sanitize_markdown
from app.services.markdown_renderer import download_filename, render_markdown, score_meter
from ui import state

logger = get_logger(__name__)

#: Colour per score, in Streamlit's inline-markdown palette. Below the bar reads
#: warm, at or above it reads cool -- the split the rubric already makes.
_SCORE_COLOUR = {1: "red", 2: "orange", 3: "blue", 4: "green"}


def render_grading() -> None:
    """Grade the conversation.

    The conversation is left in place rather than deleted afterwards: the report
    quotes it, and it goes away on its own when the next interview starts or the
    browser session ends. A grading call that fails therefore stays retryable for
    as long as the tab is open, with no window to run out.
    """
    from app.agents.interview_agents import generate_report

    reason = state.ended_reason()
    heading = {
        "manual": "Grading your interview…",
        "complete": "That's the last question. Grading…",
    }.get(reason or "", "Grading your interview…")

    st.title("Interview report")

    if not state.has_conversation():
        state.expire("gone")
        st.rerun()
        return

    transcript = state.live_interview().transcript()

    with st.spinner(heading):
        try:
            report = generate_report(transcript)
        except EmptyInterviewError:
            # E-39: the opening question alone is not an interview, and grading it
            # would have the evaluator invent five scores from no evidence.
            state.expire("empty")
            st.rerun()
            return
        except Exception as exc:  # noqa: BLE001 - shown to the candidate, with a retry
            logger.exception("Grading failed")
            state.set_turn_error(f"Grading failed: {redact_secrets(str(exc))}")
            st.rerun()
            return

    state.finish_with_report(report)
    st.rerun()


def render_grading_failure() -> None:
    """The failure state: an error and a retry."""
    st.title("Interview report")
    st.error(state.turn_error() or "Grading failed.", icon=":material/error:")

    if not state.has_conversation():
        state.expire("gone")
        st.rerun()
        return

    st.caption("Your answers are still here, so a retry sends the same interview.")

    with st.container(horizontal=True):
        if st.button("Retry grading", type="primary", icon=":material/refresh:"):
            state.set_turn_error(None)
            st.rerun()
        if st.button("Discard and start over", icon=":material/delete:"):
            state.reset_to_idle()
            st.rerun()


def render_report(report: FeedbackReport, config: InterviewConfig) -> None:
    """Draw the finished report."""
    if st.button("New session", icon=":material/arrow_back:", type="tertiary"):
        state.reset_to_idle()
        st.rerun()

    interview = presets.INTERVIEW_TYPE_PRESETS[config.interview_type]
    seniority = presets.SENIORITY_PRESETS[config.seniority]
    archetype = presets.ARCHETYPE_PRESETS[config.archetype]
    primary = presets.primary_dimensions(config.interview_type)

    st.title("Interview report")
    st.markdown(
        f":violet-badge[:material/psychology: {interview.label}] "
        f":blue-badge[:material/badge: {seniority.label}] "
        f":gray-badge[:material/domain: {archetype.label}]"
    )
    st.subheader(sanitize_markdown(report.headline))

    _render_scores(report, primary)

    st.divider()
    _render_narrative(report)

    st.divider()
    _render_download(report, config)

    st.caption(
        "Nothing is stored anywhere. This report and the conversation behind it go "
        "when you start another session or close the tab — download it if you want "
        "to keep it."
    )


def _render_scores(report: FeedbackReport, primary: tuple[RubricDimension, ...]) -> None:
    """The score row: meter, number, justification, and the quote that earned it."""
    total = sum(score.score for score in report.scores)
    st.markdown(f"**{total} / 20** across five dimensions")

    for score in report.ordered_scores(primary):
        dimension = rubric.dimension(score.dimension)
        colour = _SCORE_COLOUR[score.score]
        is_primary = score.dimension in primary

        with st.container(border=True):
            with st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
            ):
                label = f"**{dimension.label}**" if is_primary else dimension.label
                st.markdown(f"{label}" + (" :violet-badge[primary]" if is_primary else ""))
                # No backticks around the meter: Streamlit's inline colour syntax
                # does not reach inside inline code, so `:red[`...`]` renders a
                # monochrome block and loses the one signal the meter carries.
                st.markdown(f":{colour}[{score_meter(score.score)}] **{score.score}/4**")

            st.caption(sanitize_markdown(score.justification))
            # Evidence is shown, not just the score: a score with a quote behind it
            # is arguable, and arguable is what makes it useful.
            st.markdown(f"> {sanitize_markdown(score.evidence)}")


def _render_narrative(report: FeedbackReport) -> None:
    """What worked, what cost points, the rewrite, and the next drill."""
    if report.what_worked:
        st.markdown("#### What worked")
        for item in report.what_worked:
            st.markdown(f"- {sanitize_markdown(item)}")

    if report.what_cost_points:
        st.markdown("#### What cost you points")
        for deduction in report.what_cost_points:
            st.markdown(f"- {sanitize_markdown(deduction.moment)}")
            st.markdown(
                f"  &nbsp;&nbsp;↳ **Stronger move:** {sanitize_markdown(deduction.stronger_move)}"
            )

    rewrite = report.rewritten_answer
    with st.expander("One answer, rewritten at the bar", icon=":material/auto_fix_high:"):
        st.caption(sanitize_markdown(rewrite.question))
        st.markdown(sanitize_markdown(rewrite.rewrite))
        st.info(sanitize_markdown(rewrite.why_better), icon=":material/lightbulb:")

    st.markdown("#### Practise next")
    st.success(sanitize_markdown(report.next_focus), icon=":material/target:")


def _render_download(report: FeedbackReport, config: InterviewConfig) -> None:
    """The one artifact a candidate can keep."""
    st.download_button(
        "Download report",
        data=render_markdown(report, config),
        file_name=download_filename(config),
        mime="text/markdown",
        icon=":material/download:",
        type="primary",
    )


def render_expired() -> None:
    """The session is gone. Explain which way, without apologising for it."""
    reason = state.ended_reason()

    st.title("Session ended")
    if reason == "empty":
        st.info(
            "The interview ended before you answered anything, so there was nothing "
            "to grade. The opening question on its own is not an interview.",
            icon=":material/info:",
        )
    else:
        st.info(
            "This session ended and its conversation is gone. Nothing is kept between "
            "sessions, so restarting the app or closing the tab ends an interview.",
            icon=":material/timer_off:",
        )

    if st.button("Start a new session", type="primary", icon=":material/play_arrow:"):
        state.reset_to_idle()
        st.rerun()
