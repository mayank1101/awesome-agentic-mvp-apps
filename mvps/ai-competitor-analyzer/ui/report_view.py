"""Progress, the finished report, and the error screens."""

import streamlit as st

from app.core.exceptions import (
    CompanyNotFound,
    ModelUnavailable,
    ProviderQuotaExhausted,
    SearchAuthError,
    SearchQuotaError,
    SynthesisError,
)
from app.models.schemas import AnalysisRequest, Report
from app.services.pipeline import InputRejected, run
from app.services.renderer import download_filename
from ui import state as S


def run_and_render(request: AnalysisRequest) -> S.RunOutcome:
    """Drive one run, painting progress as each event arrives.

    Consumes the pipeline generator on Streamlit's own script thread — the whole
    reason the pipeline is a plain generator rather than something threaded.

    Args:
        request: The validated request.

    Returns:
        The outcome to store for the next rerun.
    """
    token = S.begin_run()
    outcome = S.RunOutcome()
    generator = run(request, transport=S.CachedTransport())

    with st.status(f"Analyzing {request.name}…", expanded=True) as status:
        try:
            while True:
                try:
                    event = next(generator)
                except StopIteration as stop:
                    outcome.report = stop.value
                    break

                if not S.is_current(token):
                    # A newer run started; abandon this one rather than paint over it.
                    status.update(label="Superseded by a newer run.", state="error")
                    return outcome

                outcome.events.append(event)
                st.write(event.message)

            status.update(label=_done_label(outcome.report), state="complete", expanded=False)

        except CompanyNotFound as exc:
            outcome.error, outcome.error_kind = str(exc), "not_found"
            outcome.error_detail = getattr(exc, "query", None)
            status.update(label="Could not identify that company.", state="error")
        except InputRejected as exc:
            outcome.error, outcome.error_kind = str(exc), "rejected"
            status.update(label="Request refused.", state="error")
        except (SearchAuthError, SearchQuotaError) as exc:
            outcome.error, outcome.error_kind = str(exc), "search"
            status.update(label="Search unavailable.", state="error")
        except (ProviderQuotaExhausted, ModelUnavailable, SynthesisError) as exc:
            outcome.error, outcome.error_kind = str(exc), "model"
            status.update(label="The model provider failed.", state="error")

    return outcome


def escape_dollars(markdown: str) -> str:
    """Escape ``$`` so Streamlit does not read prices as LaTeX.

    Streamlit's Markdown renderer treats ``$...$`` as inline maths, so a pricing
    section reading "Basic $10, Business $16" renders as "Basic 10, Business 16"
    — the two dollar signs are consumed as delimiters and the text between them
    is silently reformatted. In a section whose entire job is quoting prices,
    that is a wrong answer rather than a cosmetic problem.

    Applied at display time only. The document itself is untouched, so the
    downloaded `.md` stays exactly what the renderer produced (SC-9): the escape
    is how this one viewer must be spoken to, not a change to the content.

    Args:
        markdown: The finished report.

    Returns:
        The same text with every ``$`` escaped for Streamlit.
    """
    return markdown.replace("$", r"\$")


def _done_label(report: Report | None) -> str:
    """The one-line summary the collapsed status block keeps."""
    if report is None:
        return "Finished."
    return f"Done — {report.source_count} sources, {report.credits_spent} search credits."


def render_report(report: Report) -> None:
    """Draw a finished report, with its download."""
    if report.synthesis_failed:
        st.warning(
            "The model did not return a usable brief. The sources below are real "
            "and were retrieved for this company — the sections are not.",
            icon=":material/warning:",
        )

    left, right = st.columns([3, 1])
    with left:
        st.caption(
            f"{report.source_count} sources · {report.credits_spent} search credits · "
            f"generated {report.generated_on.isoformat()}"
        )
    with right:
        st.download_button(
            "Download .md",
            data=report.markdown,
            file_name=download_filename(report.identity, report.generated_on),
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown(escape_dollars(report.markdown))

    if st.button("Analyze another company", use_container_width=True):
        S.clear()
        st.rerun()


def render_error(message: str, kind: str) -> None:
    """Draw a terminal error, with the action that actually fixes it."""
    if kind == "not_found":
        st.error(message, icon=":material/search_off:")
        st.markdown(
            "Nothing found that looks like a company by that name. Try the "
            "**Website** field — a domain removes the ambiguity entirely."
        )
    elif kind == "rejected":
        st.error(f"That input {message}.", icon=":material/block:")
        st.markdown("Enter a company name on its own, with no instructions attached.")
    elif kind == "search":
        st.error(message, icon=":material/cloud_off:")
        st.markdown(
            "Search is what this app is built on, so there is no degraded mode: "
            "without it a brief would be the model's memory, which is exactly what "
            "this app exists to avoid."
        )
    else:
        st.error(message, icon=":material/error:")

    if st.button("Start over", use_container_width=True):
        S.clear()
        st.rerun()


def render_setup_error(missing: list[str]) -> None:
    """Draw the startup gate for missing credentials (E-61)."""
    st.error(
        f"Missing configuration: {', '.join(missing)}",
        icon=":material/key_off:",
    )
    st.markdown(
        "Set these in `.env` locally, or in **Settings → Secrets** on Streamlit "
        "Community Cloud, then restart the app.\n\n"
        "```\nTAVILY_API_KEY=tvly-…\nGROQ_API_KEY=gsk_…\n```\n\n"
        "Both are required. There is no partial mode: search without synthesis is "
        "a list of links, and synthesis without search is an unsourced chatbot."
    )
