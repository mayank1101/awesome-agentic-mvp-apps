"""Everything the user reads: progress, the shortlist, and the failure screens.

Three rules run through this module.

**A number never appears without its basis.** Every row carries its tier, and a
snippet-ranked row says so in words, not in a colour. The app's whole claim is
that its scores mean something; a score whose provenance is hidden quietly
withdraws that claim.

**Partial runs are shown, not swallowed.** A job whose page would not load, one
whose scoring call failed, one that fell below the deep-scoring cut -- all of
them appear, each saying which happened. The user came for links; the scoring is
what orders them.

**Progress is consumed on the script's own thread.** The pipeline is a generator
and this module iterates it directly. Painting from a worker thread raises
`NoSessionContext` in Streamlit, and that is why `app/` yields events instead of
taking a callback.
"""

import streamlit as st

from app.core.exceptions import InputBlocked, JobSearchError
from app.models.schemas import CandidateProfile, RunResult, ScoredJob
from app.services import pipeline
from ui import state as S
from ui.input_form import Submission

#: Score bands used for the coloured badge. Deep and shallow rows use the same
#: bands because the bands describe the *number*; what differs is the label
#: beside it, which says how the number was arrived at.
_BANDS: tuple[tuple[float, str, str], ...] = (
    (75.0, "Strong match", "🟢"),
    (55.0, "Worth a look", "🟡"),
    (0.0, "Weak match", "⚪"),
)


def run_search(submission: Submission) -> None:
    """Run one search, painting progress as the pipeline yields it."""
    for notice in submission.notices:
        st.info(notice, icon=":material/info:")

    progress = st.progress(0.0, text="Starting…")
    profile_slot = st.empty()

    try:
        with st.spinner("Working…"):
            for event in pipeline.run_search(submission.resume_text, submission.filters):
                if isinstance(event, pipeline.Progress):
                    progress.progress(min(event.fraction, 1.0), text=event.message)
                elif isinstance(event, pipeline.ProfileReady):
                    with profile_slot.container():
                        _render_profile(event.profile, expanded=False)
                elif isinstance(event, pipeline.Finished):
                    S.store_result(event.result)
    except InputBlocked as exc:
        S.store_error(
            S.ErrorState(
                title="That resume was not used for a search",
                detail=str(exc),
                items=[f"{finding.field}: {finding.message}" for finding in exc.findings],
            )
        )
    except JobSearchError as exc:
        S.store_error(S.ErrorState(title="The search did not finish", detail=str(exc)))
    finally:
        progress.empty()
        profile_slot.empty()


def render_report(result: RunResult) -> None:
    """Draw a finished run: what it did, how the resume was read, and the jobs."""
    _render_summary(result)
    _render_profile(result.profile, expanded=False)

    if not result.jobs:
        st.warning(
            "No postings matched. Widen the role, extend the date window, or add sites.",
            icon=":material/search_off:",
        )
    else:
        st.subheader(f"{len(result.jobs)} jobs, best fit first")
        for job in result.jobs:
            _render_job(job)

    st.divider()
    if st.button("New search", type="primary"):
        S.reset()
        st.rerun()


def render_error(error: S.ErrorState) -> None:
    """Draw a failure screen with a way back."""
    st.error(f"**{error.title}**\n\n{error.detail}", icon=":material/error:")
    if error.items:
        for item in error.items:
            st.write(f"- {item}")

    if st.button("Back to the search", type="primary"):
        S.reset()
        st.rerun()


def render_setup_error(missing: list[str]) -> None:
    """Draw the screen shown when required credentials are absent."""
    st.error(
        "This app needs two keys before it can do anything: one to search the web and one "
        "to read resumes and postings.",
        icon=":material/key_off:",
    )
    st.write("Missing: " + ", ".join(f"`{name}`" for name in missing))
    st.markdown(
        "- `TAVILY_API_KEY` — free tier at [tavily.com](https://tavily.com). Every web request "
        "this app makes goes through it.\n"
        "- `GROQ_API_KEY` — free tier at [console.groq.com](https://console.groq.com). Reads the "
        "resume and judges each posting.\n"
        "- `MISTRAL_API_KEY` — optional, at [console.mistral.ai](https://console.mistral.ai). "
        "Without it, matching falls back to word overlap and the app says so on screen.\n\n"
        "Put them in a local `.env` (see `.env.example`), or in the app's Secrets on "
        "Streamlit Community Cloud."
    )


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #


def _render_summary(result: RunResult) -> None:
    """State plainly what the run did, including what it did not do."""
    summary = result.summary

    left, middle, right = st.columns(3)
    left.metric("Postings found", summary.results_kept)
    middle.metric("Read in full", summary.deep_scored)
    right.metric("Sites searched", len(summary.sites))

    detail = (
        f"{summary.results_found} search results, {summary.results_kept} of them job postings. "
        f"{summary.deep_scored} had their posting fetched and scored requirement by requirement; "
        "the rest are ranked on their title and search snippet."
    )
    if summary.postings_unreadable:
        detail += (
            f" {summary.postings_unreadable} posting(s) could not be read -- usually a page "
            "rendered by JavaScript, behind a login, or already closed."
        )
    st.caption(detail)

    for notice in summary.notices:
        st.info(notice, icon=":material/info:")

    with st.expander("What was searched"):
        st.write("**Queries**")
        for query in summary.queries:
            st.write(f"- `{query}`")
        st.write("**Sites**")
        st.write(", ".join(f"`{site}`" for site in summary.sites))
        st.caption(
            f"Matching mode: {summary.matching_mode}. "
            + (
                "Semantic matching compares meaning, so 'shipped services in Go' matches "
                "'Golang experience'."
                if summary.matching_mode == "semantic"
                else "Lexical matching compares words, so wording differences read as gaps."
            )
        )


def _render_profile(profile: CandidateProfile, *, expanded: bool) -> None:
    """Show how the resume was read.

    Worth its place on screen: a resume read as the wrong role produces a page of
    plausible, wrong results, and this is the only place that mistake is visible
    in two seconds rather than inferred from thirty links.
    """
    with st.expander("How the resume was read", expanded=expanded):
        if profile.summary:
            st.write(profile.summary)

        left, right = st.columns(2)
        with left:
            st.write(f"**Level** {profile.seniority}")
            if profile.years_experience is not None:
                st.write(f"**Experience** {profile.years_experience:g} years")
            if profile.locations:
                st.write("**Locations** " + ", ".join(profile.locations))
        with right:
            if profile.titles:
                st.write("**Titles** " + ", ".join(profile.titles))
            if profile.domains:
                st.write("**Domains** " + ", ".join(profile.domains))

        if profile.skills:
            st.write("**Skills** " + ", ".join(profile.skills))
        st.caption(
            "If this is wrong, the results will be too. Adjust the target role and search again."
        )


def _render_job(job: ScoredJob) -> None:
    """Draw one row: the link, the score, its basis, and the reasoning."""
    label, icon = _band(job.score)

    with st.container(border=True):
        header, score_column = st.columns([4, 1])

        with header:
            st.markdown(f"#### [{job.display_title}]({job.hit.url})")
            facts = [fact for fact in (job.company, job.location, job.hit.domain) if fact]
            if job.remote:
                facts.append("remote")
            st.caption(" · ".join(facts))

        with score_column:
            st.metric("Match", f"{job.score:.0f}", label_visibility="collapsed")
            st.caption(f"{icon} {label}")

        if job.tier == "deep":
            st.write(job.reason)
        else:
            st.caption(f":grey[Snippet-ranked — the posting was not read.] {job.reason}")

        if job.error:
            st.caption(f":orange[Scoring failed: {job.error}]")

        if job.assessment and job.breakdown:
            _render_breakdown(job)


def _render_breakdown(job: ScoredJob) -> None:
    """Show the requirement-by-requirement arithmetic behind a deep score."""
    assessment = job.assessment
    breakdown = job.breakdown
    if assessment is None or breakdown is None:
        return

    verdicts = {verdict.requirement_id: verdict for verdict in assessment.assessments}
    marks = {"covered": "✅", "partial": "🟨", "missing": "❌"}

    with st.expander(
        f"Requirements: {breakdown.must_have_covered}/{breakdown.must_have_total} must-haves covered"
    ):
        for requirement in assessment.requirements:
            verdict = verdicts.get(requirement.id)
            if verdict is None:
                continue

            tag = "must have" if requirement.must_have else "preferred"
            st.markdown(f"{marks.get(verdict.status, '❔')} **{requirement.text}** :grey[({tag})]")
            if verdict.evidence:
                st.caption(f"Resume: “{verdict.evidence}”")
            if verdict.note:
                st.caption(verdict.note)

        st.divider()
        st.caption(
            f"Must-haves {breakdown.must_have_score:.0f}/100"
            + (
                f" · preferred {breakdown.nice_to_have_score:.0f}/100"
                if breakdown.nice_to_have_score is not None
                else " · this posting stated no preferred requirements"
            )
            + f" · matching mode: {job.matching_mode}"
        )
        if breakdown.demoted:
            st.caption(
                ":orange[Claimed coverage was reduced for "
                f"{', '.join(breakdown.demoted)} — the resume did not support it.]"
            )


def _band(score: float) -> tuple[str, str]:
    """Return the label and icon for a score."""
    for floor, label, icon in _BANDS:
        if score >= floor:
            return label, icon
    return "Weak match", "⚪"
