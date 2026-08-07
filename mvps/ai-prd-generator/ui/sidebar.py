"""Sidebar: the brief form, the progress checklist, and the export slot.

The form is rendered here, but the *outcome* is handed back to the caller rather
than acted on, so validation errors surface in the main pane where there is room
for them.

Two of the widgets this module creates are empty placeholders
(:class:`SidebarSlots`) that :mod:`ui.document` fills in later, as sections
land. They have to be created here because Streamlit renders widgets in call
order, and these belong under the form.
"""

from dataclasses import dataclass

import streamlit as st
from pydantic import ValidationError
from streamlit.delta_generator import DeltaGenerator

from app.agents import LENGTH_PRESETS, SCOPE_PRESETS
from app.core.config import get_settings
from app.models.schemas import PRDInput
from app.services.guardrails import has_severity, scan_brief
from ui import state

DEFAULT_SCOPE = "product"
DEFAULT_LENGTH = "medium"
DEFAULT_AUDIENCE = "general"

_TEXT_AREA_HEIGHT = 68


@dataclass(frozen=True)
class SidebarSlots:
    """Placeholders the document pane writes into as generation progresses.

    Attributes:
        checklist: One placeholder per outline section, for its status line.
            Empty until an outline exists.
        export: Placeholder for the download button and size estimate.
    """

    checklist: list[DeltaGenerator]
    export: DeltaGenerator


@dataclass(frozen=True)
class BriefSubmission:
    """What the form produced this run.

    Exactly one of `prd_input` and `error` is set when `submitted` is True; both
    are None on runs where the user did not press the button.

    Attributes:
        submitted: Whether the form was submitted on this run.
        prd_input: The validated brief, when it passed validation.
        error: A message to show the user, when it did not.
        warning: A guardrail note to show alongside a brief that passed -- set
            when the scanner flagged something that did not warrant blocking.
    """

    submitted: bool
    prd_input: PRDInput | None = None
    error: str | None = None
    warning: str | None = None


def render_sidebar() -> tuple[BriefSubmission, SidebarSlots]:
    """Draw the whole sidebar.

    Returns:
        The form's outcome for this run, and the placeholders the document pane
        will fill in.
    """
    with st.sidebar:
        st.markdown("### PRD generator")
        scope = _render_scope_switch()
        submission = _render_brief_form(scope)
        slots = _render_progress_slots()
    return submission, slots


def _render_scope_switch() -> str:
    """Product vs. feature.

    Deliberately outside the form: switching scope has to rerun immediately so
    the form can swap its labels and show or hide the parent-product field.
    """
    scope = st.segmented_control(
        "Scope",
        options=list(SCOPE_PRESETS.keys()),
        default=DEFAULT_SCOPE,
        format_func=lambda key: SCOPE_PRESETS[key]["label"],
        label_visibility="collapsed",
        width="stretch",
    )
    return scope or DEFAULT_SCOPE


def _render_brief_form(scope: str) -> BriefSubmission:
    """Draw the brief form and, on submit, validate it into a `PRDInput`."""
    is_feature = scope == "feature"

    with (
        st.expander(
            f"{SCOPE_PRESETS[scope]['label']} brief",
            icon=":material/edit_note:",
            expanded=state.is_brief_expanded(),
        ),
        st.form("prd_form", border=False),
    ):
        product_name = st.text_input(
            "Feature name" if is_feature else "Product name",
            placeholder="SAML single sign-on" if is_feature else "Inbox Triage Assistant",
            max_chars=80,
        )

        parent_product = None
        if is_feature:
            parent_product = st.text_area(
                "Parent product *",
                placeholder="Multi-tenant analytics SaaS on Postgres, ~400 business "
                "customers, email/password auth today",
                height=_TEXT_AREA_HEIGHT,
                max_chars=600,
                help="What the feature plugs into: what it does, who uses it, the stack.",
            )

        one_liner = st.text_input(
            "One-liner *",
            placeholder=(
                "Add SAML single sign-on to our analytics product"
                if is_feature
                else "An AI agent that pre-sorts support tickets"
            ),
            max_chars=150,
        )
        problem_statement = st.text_area(
            "Problem statement *",
            placeholder=(
                "Why this feature is needed now"
                if is_feature
                else "What problem this solves, and for whom"
            ),
            height=_TEXT_AREA_HEIGHT,
            max_chars=1000,
        )
        target_users = st.text_area(
            "Target users *",
            placeholder=(
                "IT admins at enterprise customers"
                if is_feature
                else "Support leads at mid-size SaaS companies"
            ),
            height=_TEXT_AREA_HEIGHT,
            max_chars=500,
        )
        goals_raw = st.text_area(
            "Goals (one per line, optional)",
            placeholder="Cut manual triage time by 50%",
            height=_TEXT_AREA_HEIGHT,
            max_chars=500,
        )
        context_notes = st.text_area(
            "Context notes (optional)",
            placeholder="Research, constraints, tech stack",
            height=_TEXT_AREA_HEIGHT,
            max_chars=1500,
        )

        meta_left, meta_right = st.columns(2, gap="small")
        with meta_left:
            audience = st.text_input(
                "Audience",
                value=DEFAULT_AUDIENCE,
                placeholder="e.g. eng team",
                max_chars=100,
            )
        with meta_right:
            length = st.selectbox(
                "Length",
                options=list(LENGTH_PRESETS.keys()),
                index=list(LENGTH_PRESETS.keys()).index(DEFAULT_LENGTH),
                format_func=lambda key: LENGTH_PRESETS[key]["label"],
            )

        st.space("xsmall")
        submitted = st.form_submit_button(
            "Generate PRD",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

    if not submitted:
        return BriefSubmission(submitted=False)

    return _validate(
        scope=scope,
        is_feature=is_feature,
        product_name=product_name,
        parent_product=parent_product,
        one_liner=one_liner,
        problem_statement=problem_statement,
        target_users=target_users,
        goals_raw=goals_raw,
        context_notes=context_notes,
        audience=audience,
        length=length,
    )


def _validate(
    *,
    scope: str,
    is_feature: bool,
    product_name: str,
    parent_product: str | None,
    one_liner: str,
    problem_statement: str,
    target_users: str,
    goals_raw: str,
    context_notes: str,
    audience: str,
    length: str,
) -> BriefSubmission:
    """Turn raw widget values into a validated brief, or a message to show.

    Required fields are checked here so the user gets a plain sentence rather
    than a Pydantic error; `PRDInput` is still the authority on limits, and
    backstops anything the widgets' `max_chars` did not catch.
    """
    if not (one_liner and problem_statement and target_users):
        return BriefSubmission(
            submitted=True,
            error="One-liner, problem statement, and target users are required.",
        )
    if is_feature and not parent_product:
        return BriefSubmission(
            submitted=True,
            error=(
                "Parent product is required for a feature PRD -- it's what the feature plugs into."
            ),
        )

    goals = [line.strip() for line in goals_raw.splitlines() if line.strip()] or None
    try:
        prd_input = PRDInput(
            scope=scope,
            product_name=product_name or None,
            parent_product=parent_product or None,
            one_liner=one_liner,
            problem_statement=problem_statement,
            target_users=target_users,
            goals=goals,
            context_notes=context_notes or None,
            audience=audience or DEFAULT_AUDIENCE,
            length=length,
        )
    except ValidationError as exc:
        first = exc.errors()[0]
        return BriefSubmission(submitted=True, error=first.get("msg", str(exc)))

    return _apply_guardrails(prd_input)


def _apply_guardrails(prd_input: PRDInput) -> BriefSubmission:
    """Scan a validated brief for injection attempts.

    High-severity findings block when `BLOCK_FLAGGED_INPUT` is on; medium ones
    always pass through as a warning, because the phrasing they match ("act as
    a PM would") has honest uses in a product brief. Either way the brief is
    fenced before it reaches a model, so a warning is not a hole.
    """
    settings = get_settings()
    if not settings.guardrails_enabled:
        return BriefSubmission(submitted=True, prd_input=prd_input)

    findings = scan_brief(prd_input)
    if not findings:
        return BriefSubmission(submitted=True, prd_input=prd_input)

    detail = "; ".join(f"**{f.field}** {f.message}" for f in findings[:3])

    if has_severity(findings, "high") and settings.block_flagged_input:
        return BriefSubmission(
            submitted=True,
            error=(
                f"This brief was blocked by the input guardrails: {detail}. "
                "Rewrite it to describe the product rather than instruct the assistant."
            ),
        )

    return BriefSubmission(submitted=True, prd_input=prd_input, warning=detail)


def _render_progress_slots() -> SidebarSlots:
    """Reserve the checklist and export placeholders under the form."""
    checklist: list[DeltaGenerator] = []
    outline = state.get_outline()

    if outline is not None:
        st.markdown("**Sections**")
        with st.container(gap="xsmall"):
            checklist = [st.empty() for _ in outline.sections]

    return SidebarSlots(checklist=checklist, export=st.empty())
