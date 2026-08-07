"""The input screen: a PDF upload and a pasted job description.

The form validates *before* spending a model call -- an empty box, a job
description of three words, or a PDF that turns out to be a scan all fail here,
in milliseconds, with a sentence saying what to do. Every one of those checks
exists because the alternative is a 40-second wait ending in a generic error.

The extracted resume text is shown in a collapsed expander. That is not a debug
affordance: PDF text extraction interleaves columns on some templates, and a
candidate who can see what the app actually read can tell in two seconds whether
the analysis is worth reading.
"""

from dataclasses import dataclass

import streamlit as st

from app.core.exceptions import ResumeExtractionError
from app.services.pdf_extract import extract_resume, prepare_job_description

#: Below this many characters, a "job description" is a job title. The model
#: would happily invent requirements from it, which is the failure this app is
#: built to avoid.
_MIN_JD_CHARS = 120


@dataclass(frozen=True)
class Submission:
    """What the form produced on this rerun.

    Attributes:
        submitted: Whether the button was pressed.
        resume_text: Extracted resume text, when extraction succeeded.
        job_description: Normalised posting text.
        truncated_resume: Whether the resume was capped.
        truncated_jd: Whether the posting was capped.
        error: A message to show instead of running, when validation failed.
    """

    submitted: bool = False
    resume_text: str = ""
    job_description: str = ""
    truncated_resume: bool = False
    truncated_jd: bool = False
    error: str = ""

    @property
    def ready(self) -> bool:
        """Whether this submission should start an analysis."""
        return self.submitted and not self.error and bool(self.resume_text)


def render() -> Submission:
    """Draw the form and return what it produced.

    Returns:
        A :class:`Submission`. Check ``ready`` before running anything.
    """
    st.write(
        "Upload your resume, paste the job description, and get a scored breakdown of "
        "how the two line up — then a rewritten resume that uses only what is already "
        "yours."
    )

    upload = st.file_uploader(
        "Your resume (PDF)",
        type=["pdf"],
        help="Exported from a word processor, not scanned. Nothing is stored.",
    )

    posting = st.text_area(
        "The job description",
        height=240,
        placeholder="Paste the full posting: responsibilities, requirements, nice-to-haves.",
    )

    submitted = st.button("Analyse fit", type="primary", use_container_width=True)

    if not submitted:
        return Submission()

    if upload is None:
        return Submission(submitted=True, error="Upload your resume as a PDF first.")

    if not posting or not posting.strip():
        return Submission(submitted=True, error="Paste the job description.")

    try:
        extracted = extract_resume(upload.getvalue())
    except ResumeExtractionError as exc:
        return Submission(submitted=True, error=str(exc))

    jd_text, jd_truncated = prepare_job_description(posting)
    if len(jd_text) < _MIN_JD_CHARS:
        return Submission(
            submitted=True,
            error=(
                "That job description is too short to analyse. Paste the full posting, "
                "including the requirements section."
            ),
        )

    with st.expander("What the app read from your PDF", expanded=False):
        st.caption(
            f"{extracted.page_count} page(s), {len(extracted.text):,} characters. "
            "If this looks scrambled, your PDF uses a multi-column layout — the analysis "
            "will still run, but a single-column export gives better results."
        )
        st.text(extracted.text[:4000])

    return Submission(
        submitted=True,
        resume_text=extracted.text,
        job_description=jd_text,
        truncated_resume=extracted.truncated,
        truncated_jd=jd_truncated,
    )
