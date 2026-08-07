"""The result screen: the score, the evidence behind it, and the rewrite.

Two rules shape everything here.

**A number is never shown alone.** The score sits next to its band, its four
weighted dimensions, and a per-requirement table with the quoted resume line
behind every verdict. A bare "68" tells a candidate nothing they can act on and
invites them to treat a language model's opinion as a measurement.

**The rewrite is shown beside the original, never instead of it.** The app's
guarantee is mechanical -- no invented numbers, names, or contact details -- and
it is honest about what it cannot check, which is tone and emphasis. The person
whose resume it is has to read the diff before sending it, so the diff is the
default view rather than something behind a toggle.
"""

import streamlit as st

from app.core.exceptions import (
    FabricationDetected,
    InputBlocked,
    JobMatchError,
    ModelQuotaExhausted,
    ModelRateLimited,
    RunDeadlineExceeded,
)
from app.models.schemas import FitReport
from app.services.analyzer import AnalysisResult, analyze
from app.services.resume_pdf import render_pdf
from app.services.tailor import tailor_resume
from ui import state as S
from ui.input_form import Submission

_STATUS_ICON = {"covered": "✅", "partial": "🟡", "missing": "❌"}

_CATEGORY_LABEL = {
    "surface": "Move it up",
    "reword": "Reword it",
    "quantify": "Put a number on it",
    "restructure": "Restructure",
    "gap": "Gap",
}

_BAND_COLOR = {
    "Strong match": "🟢",
    "Good match": "🟢",
    "Partial match": "🟡",
    "Weak match": "🟠",
    "Poor match": "🔴",
}


# --------------------------------------------------------------------------- #
# Running the work
# --------------------------------------------------------------------------- #


def run_analysis(submission: Submission) -> None:
    """Run one analysis and store the outcome in session state.

    Every failure lands as an :class:`~ui.state.ErrorState` rather than a
    traceback: the exception hierarchy already distinguishes the cases that need
    different actions from the reader, and this is where that pays off.
    """
    if S.busy():
        return

    S.set_busy(True)
    status = st.status("Analysing…", expanded=True)
    try:
        result = analyze(
            submission.resume_text,
            submission.job_description,
            truncated_resume=submission.truncated_resume,
            truncated_jd=submission.truncated_jd,
            progress=lambda message: status.update(label=message),
        )
    except InputBlocked as exc:
        status.update(label="Blocked", state="error")
        S.store_error(
            S.ErrorState(
                title="That document tries to instruct the assistant",
                detail=(
                    "One of the two documents contains text aimed at the model rather "
                    "than at a human reader, so the analysis was stopped. If this is a "
                    "genuine posting, remove the offending lines and try again."
                ),
                items=[f"{f.field}: {f.message}" for f in exc.findings],
            )
        )
    except (ModelRateLimited, ModelQuotaExhausted) as exc:
        status.update(label="Provider limit reached", state="error")
        S.store_error(S.ErrorState(title="The model provider is throttling us", detail=str(exc)))
    except RunDeadlineExceeded as exc:
        status.update(label="Timed out", state="error")
        S.store_error(
            S.ErrorState(
                title="That took too long",
                detail=f"{exc} Try a shorter job description.",
            )
        )
    except JobMatchError as exc:
        status.update(label="Failed", state="error")
        S.store_error(S.ErrorState(title="The analysis could not finish", detail=str(exc)))
    else:
        status.update(label="Done", state="complete")
        S.store_analysis(result)
    finally:
        S.set_busy(False)


def run_tailoring(analysis: AnalysisResult) -> None:
    """Produce a tailored resume and store it, or store the refusal."""
    if S.busy():
        return

    S.set_busy(True)
    status = st.status("Rewriting…", expanded=True)
    try:
        resume = tailor_resume(analysis, progress=lambda message: status.update(label=message))
    except FabricationDetected as exc:
        status.update(label="Rewrite refused", state="error")
        S.store_error(
            S.ErrorState(
                title="The rewrite was refused",
                detail=str(exc),
                items=exc.offenders,
            )
        )
    except JobMatchError as exc:
        status.update(label="Failed", state="error")
        S.store_error(S.ErrorState(title="The rewrite could not finish", detail=str(exc)))
    else:
        status.update(label="Done", state="complete")
        S.store_tailored(resume)
    finally:
        S.set_busy(False)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def render_report(analysis: AnalysisResult) -> None:
    """Draw the score, the dimensions, the evidence, and the rewrite controls."""
    report = analysis.report

    _render_headline(report, analysis)
    _render_dimensions(report)
    _render_advice(report)
    _render_requirements(report, analysis)
    _render_actions(report, analysis)
    _render_next_step(analysis)

    st.divider()
    if st.button("Analyse another job", use_container_width=True):
        S.reset()
        st.rerun()


def _render_headline(report: FitReport, analysis: AnalysisResult) -> None:
    """The score, the band, and the caveats that belong beside them."""
    posting = analysis.posting
    heading = posting.title or "This role"
    if posting.company:
        heading += f" · {posting.company}"
    st.subheader(heading)

    left, right = st.columns([1, 2])
    with left:
        st.metric("Fit score", f"{report.overall_score}/100")
        st.write(f"{_BAND_COLOR.get(report.band, '⚪')} **{report.band}**")
    with right:
        st.caption(
            "This score is computed from the requirement verdicts below, not asked for "
            "from a model. It describes how this resume reads against this posting — it "
            "is not a hiring decision, and no recruiter sees it."
        )
        if report.matching_mode == "lexical":
            st.caption(
                "⚠️ Running without an embedding key, so requirements were matched on "
                "word overlap. Equivalent wording (“Golang” vs “Go”) may be scored as a "
                "miss. Set MISTRAL_API_KEY for semantic matching."
            )
        if report.truncated_resume or report.truncated_jd:
            trimmed = " and ".join(
                part
                for part, flag in (
                    ("resume", report.truncated_resume),
                    ("posting", report.truncated_jd),
                )
                if flag
            )
            st.caption(f"⚠️ The {trimmed} was longer than the cap and was truncated.")


def _render_dimensions(report: FitReport) -> None:
    """The weighted components, so the total is reconstructible by hand."""
    st.write("**How the score breaks down**")
    for dimension in report.dimensions:
        if dimension.weight == 0:
            continue
        st.progress(
            min(1.0, dimension.earned / 100),
            text=(
                f"{dimension.name} — {dimension.earned:.0f}/100 "
                f"(weight {dimension.weight:.0%}) · {dimension.detail}"
            ),
        )


def _render_advice(report: FitReport) -> None:
    """Strengths and gaps, side by side."""
    columns = st.columns(2)
    with columns[0]:
        if report.strengths:
            st.write("**Lead with these**")
            for item in report.strengths:
                st.markdown(f"- {item}")
    with columns[1]:
        if report.gaps:
            st.write("**Gaps against the posting**")
            for item in report.gaps:
                st.markdown(f"- {item}")


def _render_actions(report: FitReport, analysis: AnalysisResult) -> None:
    """The action plan: what to change, in priority order.

    Split into two lists on purpose. The first is work that can genuinely move
    the score, because the evidence is already on the resume. The second is
    honest handling of what is not there -- kept visually separate so nobody
    reads "you don't have Kubernetes" as "add Kubernetes".
    """
    if not report.actions and not report.keyword_actions:
        return

    st.divider()
    st.subheader("What to change")

    fixable = [action for action in report.actions if not action.is_gap]
    gaps = [action for action in report.actions if action.is_gap]
    requirements = {r.id: r for r in analysis.posting.requirements}

    if fixable:
        st.write("**Do these — the evidence is already in your resume**")
        for index, action in enumerate(fixable, start=1):
            st.markdown(
                f"**{index}. {_CATEGORY_LABEL.get(action.category, action.category)}"
                f"{f' · {action.section}' if action.section else ''}**"
            )
            st.markdown(action.change)
            if action.rationale:
                st.caption(action.rationale)
            served = [
                requirements[rid].text for rid in action.requirement_ids if rid in requirements
            ]
            if served:
                st.caption("Addresses: " + " · ".join(served))

    if gaps:
        st.write("**These are real gaps — do not write around them**")
        for action in gaps:
            st.markdown(f"- {action.change}")
            if action.rationale:
                st.caption(action.rationale)

    _render_keyword_actions(report)


def _render_keyword_actions(report: FitReport) -> None:
    """Which missing posting keywords the resume could honestly carry."""
    if not report.keyword_actions:
        return

    supported = [action for action in report.keyword_actions if action.supported]
    unsupported = [action for action in report.keyword_actions if not action.supported]

    with st.expander(
        f"Keywords the posting uses that your resume does not ({len(report.keyword_actions)})"
    ):
        st.caption(
            "Computed by matching each keyword against your own resume lines — not asked "
            "of a model."
        )
        if supported:
            st.write("**You can use these honestly — the work is already described**")
            for action in supported:
                st.markdown(f"- **{action.keyword}** — you wrote: “{action.evidence}”")
        if unsupported:
            st.write("**No supporting line found — leave these out**")
            st.markdown(", ".join(f"`{action.keyword}`" for action in unsupported))


def _render_requirements(report: FitReport, analysis: AnalysisResult) -> None:
    """Every requirement, its verdict, and the resume line behind it."""
    requirements = {r.id: r for r in analysis.posting.requirements}
    with st.expander(f"Requirement-by-requirement ({len(report.assessments)})", expanded=False):
        for assessment in report.assessments:
            requirement = requirements.get(assessment.requirement_id)
            if requirement is None:
                continue
            label = "must-have" if requirement.must_have else "preferred"
            st.markdown(
                f"{_STATUS_ICON[assessment.status]} **{requirement.text}** "
                f"<span style='opacity:0.6'>({label}, similarity "
                f"{assessment.similarity:.2f})</span>",
                unsafe_allow_html=True,
            )
            if assessment.evidence:
                st.caption(f"From your resume: “{assessment.evidence}”")
            if assessment.note:
                st.caption(assessment.note)
            st.write("")


# --------------------------------------------------------------------------- #
# Choosing what happens next
# --------------------------------------------------------------------------- #


def _render_next_step(analysis: AnalysisResult) -> None:
    """Let the candidate choose between editing it themselves and a rewrite.

    Two genuinely different things, and which one is right is not the app's call.
    A candidate who edits their own resume keeps their voice and knows every line
    they will be asked about; one who takes the rewrite saves twenty minutes. The
    app owes them the choice and the same evidence either way, not a funnel into
    the button that costs a model call.
    """
    st.divider()
    choice = S.next_step()

    if choice is None:
        st.subheader("How do you want to fix it?")
        left, right = st.columns(2)
        with left:
            st.markdown("**I'll edit it myself**")
            st.caption(
                "Take the checklist above into your own resume file. Keeps your wording, "
                "and you will know every line you are asked about in the interview."
            )
            if st.button("Give me the checklist", use_container_width=True):
                S.set_next_step("self")
                st.rerun()
        with right:
            st.markdown("**Let AI rewrite it**")
            st.caption(
                "Reorders and rewords what is already there, in this posting's vocabulary. "
                "Every number, employer, tool, and contact detail in the output is checked "
                "against your original; a rewrite that invents one is rejected, not shown."
            )
            if st.button("Rewrite it for me", type="primary", use_container_width=True):
                S.set_next_step("ai")
                run_tailoring(analysis)
                st.rerun()
        return

    if choice == "self":
        _render_checklist(analysis)
        if st.button("Actually, let AI rewrite it instead", use_container_width=True):
            S.set_next_step("ai")
            run_tailoring(analysis)
            st.rerun()
        return

    _render_rewrite_section(analysis)


def _render_checklist(analysis: AnalysisResult) -> None:
    """The self-edit path: the action plan as a document to work from."""
    report = analysis.report
    st.subheader("Your edit checklist")
    st.caption(
        "Work top to bottom. Everything here is supported by something already in your "
        "resume — nothing on this list asks you to claim experience you do not have."
    )

    markdown = _checklist_markdown(analysis)
    st.markdown(markdown)

    st.download_button(
        "Download checklist (Markdown)",
        data=markdown.encode("utf-8"),
        file_name=f"{_file_stem(analysis)}-checklist.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if report.actions:
        st.caption(
            "Re-run the analysis after editing to see the score move — the app keeps "
            "nothing, so upload the new version as a fresh run."
        )


def _checklist_markdown(analysis: AnalysisResult) -> str:
    """Render the action plan as a standalone Markdown document."""
    report = analysis.report
    posting = analysis.posting
    title = posting.title or "this role"

    lines = [
        f"# Resume edit checklist — {title}",
        "",
        f"Current fit score: **{report.overall_score}/100** ({report.band})",
        "",
    ]

    fixable = [action for action in report.actions if not action.is_gap]
    gaps = [action for action in report.actions if action.is_gap]

    if fixable:
        lines += ["## Do these first", ""]
        for index, action in enumerate(fixable, start=1):
            where = f" — *{action.section}*" if action.section else ""
            lines.append(
                f"{index}. **{_CATEGORY_LABEL.get(action.category, action.category)}**{where}"
            )
            lines.append(f"   {action.change}")
            if action.rationale:
                lines.append(f"   _{action.rationale}_")
            lines.append("")

    supported = [a for a in report.keyword_actions if a.supported]
    if supported:
        lines += [
            "## Wording the posting uses that yours does not",
            "",
            "Each of these is backed by a line you already wrote — swapping in the posting's "
            "term is honest, and it is what a keyword filter looks for.",
            "",
        ]
        lines += [f"- **{a.keyword}** — your line: “{a.evidence}”" for a in supported]
        lines.append("")

    if gaps:
        lines += [
            "## Real gaps — handle, do not hide",
            "",
        ]
        lines += [f"- {action.change}" for action in gaps]
        lines.append("")

    unsupported = [a.keyword for a in report.keyword_actions if not a.supported]
    if unsupported:
        lines += [
            "## Do not add these",
            "",
            "Nothing in your resume supports them. Adding them is the thing that fails in "
            "an interview:",
            "",
            ", ".join(f"`{keyword}`" for keyword in unsupported),
            "",
        ]

    return "\n".join(lines)


def _render_rewrite_section(analysis: AnalysisResult) -> None:
    """The rewrite, once it exists."""
    resume = S.tailored()

    if resume is None:
        st.caption("The rewrite was not produced. Use the checklist option instead.")
        if st.button("Show me the checklist", use_container_width=True):
            S.set_next_step("self")
            st.rerun()
        return

    st.subheader("Tailored resume")
    if resume.flagged:
        st.warning(
            "These fragments are not in your original resume. Strict mode is off, so "
            "they were kept — check each one before sending this.",
            icon="⚠️",
        )
        for item in resume.flagged:
            st.markdown(f"- {item}")

    tabs = st.tabs(["Tailored", "Original", "What changed"])
    with tabs[0]:
        st.markdown(resume.markdown)
    with tabs[1]:
        st.text(analysis.resume_text)
    with tabs[2]:
        if resume.changes:
            for change in resume.changes:
                st.markdown(f"**{change.section}** — {change.change}")
                if change.reason:
                    st.caption(change.reason)
        else:
            st.caption("The rewrite reported no changes.")

    _render_downloads(resume.markdown, analysis)


def _render_downloads(markdown: str, analysis: AnalysisResult) -> None:
    """Markdown and PDF downloads, side by side."""
    stem = _file_stem(analysis)
    left, right = st.columns(2)

    with left:
        st.download_button(
            "Download Markdown",
            data=markdown.encode("utf-8"),
            file_name=f"{stem}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with right:
        try:
            pdf_bytes = render_pdf(markdown, name_hint=analysis.profile.name or "Resume")
        except Exception as exc:  # noqa: BLE001 - a render failure must not lose the Markdown
            st.caption(
                f"PDF rendering failed ({type(exc).__name__}). The Markdown above is intact."
            )
        else:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"{stem}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )


def _file_stem(analysis: AnalysisResult) -> str:
    """Build a filename stem from the candidate and the role."""
    parts = [analysis.profile.name or "resume", analysis.posting.title or "tailored"]
    slug = "-".join(
        "-".join("".join(c if c.isalnum() else " " for c in part).split()) for part in parts
    ).lower()
    return slug[:80] or "tailored-resume"


# --------------------------------------------------------------------------- #
# Error screens
# --------------------------------------------------------------------------- #


def render_error(error: S.ErrorState) -> None:
    """Draw a stored error, with a way back to the form."""
    st.error(f"**{error.title}**\n\n{error.detail}")
    if error.items:
        for item in error.items:
            st.markdown(f"- {item}")

    if st.button("Back", use_container_width=True):
        S.clear_error()
        st.rerun()


def render_setup_error(missing: list[str]) -> None:
    """Draw the startup screen for missing credentials."""
    st.error(
        "**This app is not configured yet.**\n\n"
        "Set the following before running it:\n\n" + "\n".join(f"- `{name}`" for name in missing)
    )
    st.caption(
        "Locally, put them in a `.env` beside `streamlit_app.py`. On Streamlit "
        "Community Cloud, put them in the app's Secrets. `MISTRAL_API_KEY` is optional: "
        "without it, requirement matching falls back to word overlap."
    )
