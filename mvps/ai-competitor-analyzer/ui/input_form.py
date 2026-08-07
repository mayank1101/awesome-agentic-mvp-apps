"""The input screen: a company name, an optional domain, and one button."""

from dataclasses import dataclass

import streamlit as st

from app.models.schemas import AnalysisRequest
from app.services.guardrails import has_severity, scan_input
from app.services.normalize import (
    InvalidDomainError,
    is_meaningful_name,
    normalize_domain,
    normalize_name,
)
from ui import state as S


@dataclass
class Submission:
    """What the form produced on this rerun."""

    submitted: bool = False
    request: AnalysisRequest | None = None
    error: str | None = None
    warning: str | None = None


def render() -> Submission:
    """Draw the form and validate whatever was submitted.

    Returns:
        The submission. A returned `error` is drawn by the caller in the main
        pane, where there is room for it.
    """
    st.markdown(
        "Profiles **one competitor** from public web sources: what they sell, how "
        "they price it, who they target, and what they have shipped in the last "
        "year — with every source listed."
    )

    with st.form("analyze", clear_on_submit=False):
        name = st.text_input(
            "Company name",
            placeholder="Notion",
            max_chars=S.settings().max_company_name_chars * 2,
            help="The competitor to profile.",
        )
        domain = st.text_input(
            "Website (optional)",
            placeholder="notion.so",
            help=(
                "Skips the identification step and removes any ambiguity — worth "
                "filling in for a common name like “Apollo” or “Linear”."
            ),
        )
        clicked = st.form_submit_button(
            "Analyze",
            type="primary",
            use_container_width=True,
            disabled=not S.can_start(),
        )

    if not S.can_start():
        cap = S.settings().max_reports_per_session
        st.info(
            f"Session limit reached ({cap} reports). Reload the page to start a new "
            "session — this is a cost guard on a shared demo key, not an account limit.",
            icon=":material/hourglass_disabled:",
        )

    if not clicked:
        return Submission()

    return _validate(name, domain)


def _validate(raw_name: str, raw_domain: str) -> Submission:
    """Normalise and check the submitted fields (E-01 to E-05, E-41)."""
    settings = S.settings()
    name = normalize_name(raw_name, max_chars=settings.max_company_name_chars)

    if not is_meaningful_name(name):
        return Submission(submitted=True, error="Enter a company name to analyze.")

    try:
        domain = normalize_domain(raw_domain)
    except InvalidDomainError as exc:
        return Submission(submitted=True, error=str(exc))

    warning: str | None = None
    if settings.guardrails_enabled:
        findings = scan_input(name, domain)
        if findings:
            worst = findings[0]
            if has_severity(findings, "high") and settings.block_flagged_input:
                return Submission(
                    submitted=True,
                    error=f"That input {worst.message}. Enter a company name on its own.",
                )
            warning = f"Heads up: the input {worst.message}. Analyzing it anyway."

    truncated = len(normalize_name(raw_name, max_chars=10_000)) > len(name)
    if truncated:
        warning = f"Name was shortened to {settings.max_company_name_chars} characters."

    return Submission(
        submitted=True,
        request=AnalysisRequest(name=name, domain=domain),
        warning=warning,
    )
