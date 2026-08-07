"""The search form: a resume, a few optional narrowings, and the site list.

Everything except the resume is optional, which is the point -- the zero-effort
path is upload and run. The filters exist for the user whose resume points at
their past while their intent points somewhere else.

The site list is a text area rather than a multiselect, for a boring reason: a
multiselect can only offer options it already knows, and the whole promise here
is that the user chooses which sites are searched. One domain per line is
editable, pasteable, and obvious.

This module returns a :class:`Submission` and paints nothing about results. It
also does no network work: PDF extraction happens here because it is local, fast,
and its failures are about the *form* -- a password-protected file is a thing to
fix in the uploader, not an error screen to land on.
"""

from dataclasses import dataclass, field

import streamlit as st

from app.core.config import DEFAULT_JOB_SITES
from app.core.exceptions import ResumeExtractionError
from app.models.schemas import SearchFilters
from app.services import sites as site_rules
from app.services.pdf_extract import extract_resume
from ui import state as S

_SENIORITY_LABELS: dict[str, str] = {
    "Infer from resume": "",
    "Junior": "junior",
    "Mid": "mid",
    "Senior": "senior",
    "Lead / Staff": "lead",
}

_RECENCY_LABELS: dict[str, int | None] = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Any time": None,
}


@dataclass(frozen=True)
class Submission:
    """What the form produced on this rerun.

    Attributes:
        submitted: Whether the button was pressed at all.
        resume_text: Extracted resume text, empty when extraction failed.
        filters: What to search for.
        error: A message to show above the form instead of running.
        notices: Non-fatal things worth saying -- a truncated resume, a domain
            that could not be parsed.
    """

    submitted: bool = False
    resume_text: str = ""
    filters: SearchFilters = field(default_factory=SearchFilters)
    error: str = ""
    notices: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """Whether this submission can start a run."""
        return self.submitted and not self.error and bool(self.resume_text)


def render() -> Submission:
    """Draw the form and return what the user submitted."""
    st.write(
        "Upload a resume. The app reads it, searches the job sites you choose, and scores "
        "each posting against what the resume actually says."
    )

    with st.form("search", border=False):
        upload = st.file_uploader(
            "Resume (PDF)",
            type=["pdf"],
            help="Text-based PDF. Scans are rejected -- there is no OCR in this app.",
        )

        left, right = st.columns(2)
        with left:
            role = st.text_input(
                "Target role",
                placeholder="Backend Engineer",
                help="Leave blank to use the titles the resume evidences. Fill it in if you are "
                "changing direction -- the search follows this, the scoring stays honest.",
            )
            location = st.text_input("Location", placeholder="Bengaluru, or Remote")
        with right:
            seniority_label = st.selectbox("Seniority", list(_SENIORITY_LABELS))
            recency_label = st.selectbox("Posted", list(_RECENCY_LABELS), index=1)

        remote_only = st.checkbox("Remote only")

        with st.expander("Sites to search", expanded=False):
            st.caption(
                "One domain per line. Only these are searched -- a posting anywhere else is "
                "never fetched. Applicant-tracking domains (Greenhouse, Lever, Ashby, Workable) "
                "give one job per page and the fullest requirements; the big boards find "
                "postings that exist nowhere else."
            )
            sites_text = st.text_area(
                "Domains",
                value=S.sites_text(),
                height=180,
                label_visibility="collapsed",
            )
            if st.form_submit_button("Reset to defaults", type="secondary"):
                S.store_sites("\n".join(DEFAULT_JOB_SITES))
                st.rerun()

        submitted = st.form_submit_button("Search jobs", type="primary")

    if not submitted:
        return Submission()

    S.store_sites(sites_text)
    return _build(upload, role, location, remote_only, seniority_label, recency_label, sites_text)


def _build(
    upload: object,
    role: str,
    location: str,
    remote_only: bool,
    seniority_label: str,
    recency_label: str,
    sites_text: str,
) -> Submission:
    """Validate the form and turn it into a submission."""
    notices: list[str] = []

    site_list, rejected = site_rules.normalize_sites(sites_text.splitlines())
    if rejected:
        notices.append("Not searched, because these are not domains: " + ", ".join(rejected))
    if not site_list:
        return Submission(
            submitted=True,
            error="Add at least one job site to search, for example boards.greenhouse.io.",
        )

    if upload is None:
        return Submission(submitted=True, error="Upload a resume PDF to search against.")

    try:
        extracted = extract_resume(upload.getvalue())  # type: ignore[attr-defined]
    except ResumeExtractionError as exc:
        return Submission(submitted=True, error=str(exc))

    if extracted.truncated:
        notices.append(
            "The resume was longer than this app reads, so it was cut at a line boundary. "
            "Scores reflect the part that was read."
        )

    filters = SearchFilters(
        role=role.strip(),
        location=location.strip(),
        remote_only=remote_only,
        seniority=_SENIORITY_LABELS[seniority_label] or None,  # type: ignore[arg-type]
        recency_days=_RECENCY_LABELS[recency_label],
        sites=site_list,
    )

    return Submission(
        submitted=True,
        resume_text=extracted.text,
        filters=filters,
        notices=notices,
    )
