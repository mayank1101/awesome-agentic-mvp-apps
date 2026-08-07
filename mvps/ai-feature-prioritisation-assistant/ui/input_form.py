"""The input screen: product context, the backlog paste box, and the button.

A paste box rather than a grid of numeric fields is the whole product decision.
Every tool being replaced here already offers four columns per row and no help
filling them; the reason people do not use RICE is not that multiplication is
hard.

Validation runs in one place, :func:`render_input_form`, and returns a
:class:`Submission` rather than acting on it. That keeps the decision about what
to do with an error (show it, block, warn and continue) with the entry point,
where the rerun order is visible.
"""

from dataclasses import dataclass

import streamlit as st

from app.core.config import get_settings
from app.core.exceptions import BacklogParseError
from app.models.schemas import BacklogInput
from app.services.backlog import parse_backlog
from app.services.guardrails import Finding, has_severity, scan_backlog

BACKLOG_TEXT = "backlog_text"
CONTEXT_TEXT = "context_text"

_SAMPLE_CONTEXT = (
    "B2B SaaS invoicing tool. 4,000 paying accounts, ~12,000 seats. "
    "8 engineers, 1 designer. This quarter is about enterprise readiness."
)

#: Deliberately messy: bullets, em dashes, a feature with no evidence at all,
#: and one with a number in the wrong unit. A sample that parses perfectly and
#: estimates confidently would demo the app rather than show it working.
_SAMPLE_BACKLOG = """\
- Bulk CSV export — sales asks for this every single week, blocked two renewals last quarter. Maybe a sprint.
- Dark mode — everyone asks in the feedback widget, nobody has ever churned over it. Easy, mostly CSS.
- SSO / SAML — only 3 enterprise deals blocked on it, but they're our biggest. Big lift, needs a security review.
- Invoice reminder emails — support gets ~40 tickets a month asking why customers weren't reminded
- Mobile app — no data, an exec keeps mentioning it. Huge.
- Custom invoice templates — 12 accounts have asked, all on the top plan. Design-heavy, a month or two?
- Audit log — enterprise checklist item, comes up in every security questionnaire
- Multi-currency — 400ish EU accounts hit this, we lose them at signup. Tricky, touches billing core.
- Keyboard shortcuts — power users on the forum. Two days.
"""


@dataclass(frozen=True)
class Submission:
    """What one press of the estimate button produced.

    Attributes:
        submitted: Whether the button was pressed this run.
        backlog: The parsed backlog, when it validated.
        error: A message to show instead of estimating.
        warning: A message to show *alongside* an estimate that went ahead.
    """

    submitted: bool = False
    backlog: BacklogInput | None = None
    error: str | None = None
    warning: str | None = None


def _describe(findings: list[Finding]) -> str:
    """Turn findings into one line a user can act on."""
    return "; ".join(f"{finding.field}: {finding.message}" for finding in findings[:3])


def _validate(raw_text: str, context: str) -> Submission:
    """Parse and scan one submission, deciding whether it may proceed."""
    settings = get_settings()

    try:
        backlog = parse_backlog(raw_text, product_context=context)
    except BacklogParseError as exc:
        return Submission(submitted=True, error=str(exc))

    if not settings.guardrails_enabled:
        return Submission(submitted=True, backlog=backlog)

    findings = scan_backlog(backlog)
    if not findings:
        return Submission(submitted=True, backlog=backlog)

    detail = _describe(findings)
    if has_severity(findings, "high") and settings.block_flagged_input:
        return Submission(
            submitted=True,
            error=(
                f"Blocked before sending: {detail}. Backlog notes are treated as data, never as "
                "instructions to the estimator — rewrite the flagged item to describe the feature "
                "instead of the score you want it to get."
            ),
        )
    return Submission(submitted=True, backlog=backlog, warning=detail)


def _load_sample() -> None:
    """Fill both boxes with the sample backlog."""
    st.session_state[BACKLOG_TEXT] = _SAMPLE_BACKLOG
    st.session_state[CONTEXT_TEXT] = _SAMPLE_CONTEXT


def render_input_form() -> Submission:
    """Draw the input screen and report what the user submitted.

    Returns:
        A :class:`Submission`. ``submitted`` is False on every run where the
        button was not pressed, which is most of them.
    """
    settings = get_settings()

    st.markdown("#### Your backlog")
    st.caption(
        "One feature per line, or one per paragraph for longer notes. Paste them as you already "
        "wrote them — rough effort and impact hints are exactly what the estimator reads."
    )

    st.text_area(
        "Product context",
        key=CONTEXT_TEXT,
        height=80,
        max_chars=settings.max_context_chars,
        placeholder="B2B SaaS, 4,000 accounts, 8 engineers. This quarter is about enterprise readiness.",
        help=(
            "Optional, but it is what anchors Reach in real numbers and Effort in your team's "
            "months. Without it both become assumptions, and the app will say so."
        ),
    )

    st.text_area(
        f"Feature ideas (up to {settings.max_features})",
        key=BACKLOG_TEXT,
        height=260,
        placeholder="Bulk CSV export — sales asks weekly, blocked two renewals. Maybe a sprint.\nDark mode — everyone asks, nobody churns over it. Easy.\nSSO — 3 enterprise deals blocked, big lift.",
    )

    left, right = st.columns([1, 1])
    with left:
        submitted = st.button(
            "Score and rank",
            type="primary",
            width="stretch",
            icon=":material/leaderboard:",
        )
    with right:
        st.button(
            "Load a sample backlog",
            width="stretch",
            on_click=_load_sample,
            icon=":material/description:",
        )

    if not submitted:
        return Submission()

    return _validate(st.session_state.get(BACKLOG_TEXT, ""), st.session_state.get(CONTEXT_TEXT, ""))


def render_empty_state() -> None:
    """Explain what the app does, before there is anything to show."""
    st.markdown(
        """
##### How this works

1. **You paste rough notes.** No numeric form to fill in — that is the thing being replaced.
2. **One model call estimates four factors per feature** — Reach, Impact, Confidence, Effort —
   reading the whole list at once so the features are calibrated against each other. It produces
   factors and reasoning. It never produces a score.
3. **Both scores are computed in Python** from those factors, and shown next to the numbers that
   made them.
4. **You disagree and edit.** Any factor, any row. Re-ranking is instant and calls no model.

RICE and ICE share one factor set here, so where they disagree it is because of what each formula
**ignores** — ICE has no Reach term at all. The app names the features that move and why.
"""
    )
