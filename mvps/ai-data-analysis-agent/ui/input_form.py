"""The two input widgets: the CSV upload, and the question box.

Split into two functions because they belong to two different screens of the
same session -- upload happens once, questions happen repeatedly against the
dataset it produced. Validation happens before any model call is made in
either case: an empty question or an unreadable CSV fails here, in
milliseconds, rather than after a spinner.
"""

from dataclasses import dataclass

import streamlit as st

from app.core.config import get_settings
from app.core.exceptions import CsvError
from app.services.csv_loader import load_csv
from ui import state as S


@dataclass(frozen=True)
class UploadSubmission:
    """What the upload widget produced on this rerun."""

    submitted: bool = False
    error: str = ""


def render_upload() -> UploadSubmission:
    """Draw the CSV uploader and load the file if one was submitted.

    On success, stores the dataset directly via :mod:`ui.state` and returns a
    submission with no error, so the caller knows to rerun.
    """
    st.write(
        "Upload a clean CSV file, then ask questions about it in plain language. "
        "Every answer is computed by running real pandas code against your data — "
        "the model never states a number on its own."
    )

    settings = get_settings()
    upload = st.file_uploader(
        "Your CSV file",
        type=["csv"],
        help=f"Up to {settings.max_upload_bytes / 1_048_576:.0f} MB, comma-separated, with a header row.",
    )

    if upload is None:
        return UploadSubmission()

    try:
        df, profile = load_csv(upload.getvalue())
    except CsvError as exc:
        return UploadSubmission(submitted=True, error=str(exc))

    S.store_dataset(df, profile, upload.name)
    return UploadSubmission(submitted=True)


def render_question(*, disabled: bool) -> str:
    """Draw the question box and return the submitted question, or an empty string.

    Args:
        disabled: Whether to disable the input while a previous question is
            still being answered.
    """
    settings = get_settings()
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about your data",
            placeholder="e.g. What are the top 5 categories by total revenue?",
            max_chars=settings.max_question_chars,
            disabled=disabled,
        )
        submitted = st.form_submit_button("Ask", type="primary", disabled=disabled)

    if not submitted or not question.strip():
        return ""
    return question.strip()
