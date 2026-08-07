"""Progress, the finished report, and the error screens."""

import streamlit as st

from app.core.exceptions import (
    AnalysisError,
    ModelUnavailable,
    ProviderQuotaExhausted,
    ReviewFetchError,
)
from app.models.schemas import AppIdentity, Report
from app.services.pipeline import run
from app.services.renderer import download_filename
from ui import state as S


def run_and_render(identity: AppIdentity) -> S.RunOutcome:
    """Drive one run, painting progress as each event arrives.

    Consumes the pipeline generator on Streamlit's own script thread -- the
    whole reason the pipeline is a plain generator rather than something
    threaded.
    """
    token = S.begin_run()
    outcome = S.RunOutcome()
    generator = run(identity, fetch_reviews=S.cached_fetch_reviews)

    with st.status(f"Analyzing {identity.track_name}…", expanded=True) as status:
        try:
            while True:
                try:
                    event = next(generator)
                except StopIteration as stop:
                    outcome.report = stop.value
                    break

                if not S.is_current(token):
                    status.update(label="Superseded by a newer run.", state="error")
                    return outcome

                outcome.events.append(event)
                st.write(event.message)

            status.update(label=_done_label(outcome.report), state="complete", expanded=False)

        except ReviewFetchError as exc:
            outcome.error, outcome.error_kind = str(exc), "reviews"
            status.update(label="Could not fetch reviews.", state="error")
        except (ProviderQuotaExhausted, ModelUnavailable, AnalysisError) as exc:
            outcome.error, outcome.error_kind = str(exc), "model"
            status.update(label="The model provider failed.", state="error")

    return outcome


def escape_dollars(markdown: str) -> str:
    """Escape ``$`` so Streamlit does not read prices/currency as LaTeX.

    Applied at display time only -- the downloaded `.md` stays exactly what
    the renderer produced.
    """
    return markdown.replace("$", r"\$")


def _done_label(report: Report | None) -> str:
    """The one-line summary the collapsed status block keeps."""
    if report is None:
        return "Finished."
    return f"Done — {report.stats.fetched_count} reviews, {len(report.gaps)} gap(s) found."


def render_report(report: Report) -> None:
    """Draw a finished report, with its download."""
    if report.analysis_failed:
        st.warning(
            "The model did not return a usable gap analysis. The stats below are "
            "real; the gap list is not.",
            icon=":material/warning:",
        )
    elif report.insufficient_signal:
        st.info(
            f"Only {report.stats.critical_count} critical review(s) in this sample — "
            "not enough for a reliable gap analysis. Showing the rating snapshot instead.",
            icon=":material/info:",
        )

    left, right = st.columns([3, 1])
    with left:
        st.caption(
            f"{report.stats.fetched_count} reviews fetched · {len(report.gaps)} gap(s) · "
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

    st.markdown(escape_dollars(report.markdown), unsafe_allow_html=True)

    if st.button("Analyze another app", use_container_width=True):
        S.clear()
        st.rerun()


def render_error(message: str, kind: str) -> None:
    """Draw a terminal error, with the action that actually fixes it."""
    if kind == "reviews":
        st.error(message, icon=":material/cloud_off:")
        st.markdown(
            "Apple's public review feed is unofficial and occasionally rejects a "
            "request outright. Try again in a moment, or try a different app."
        )
    elif kind == "model":
        st.error(message, icon=":material/smart_toy:")
        st.markdown(
            "The rating snapshot would still be available — this failure is "
            "specific to the gap-analysis model call."
        )
    else:
        st.error(message, icon=":material/error:")

    if st.button("Start over", use_container_width=True):
        S.clear()
        st.rerun()


def render_setup_error(missing: list[str]) -> None:
    """Draw the startup gate for missing credentials."""
    st.error(f"Missing configuration: {', '.join(missing)}", icon=":material/key_off:")
    st.markdown(
        "Set this in `.env` locally, or in **Settings → Secrets** on Streamlit "
        "Community Cloud, then restart the app.\n\n"
        "```\nGROQ_API_KEY=gsk_…\n```\n\n"
        "This is the only required credential — app lookup and review fetching, on both "
        "stores, are free, keyless, public endpoints."
    )
