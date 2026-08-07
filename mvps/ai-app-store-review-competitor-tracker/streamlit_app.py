"""Entry point.

    streamlit run streamlit_app.py

Thin on purpose: this module owns the secrets bridge, page setup, and the run
order. The widgets live in :mod:`ui`, everything else in :mod:`app`.

The secrets bridge runs before the first settings read and uses `setdefault`
so a real environment variable always wins over a `st.secrets` entry. It lives
here rather than in `app/core/config.py` so that module stays a plain
pydantic-settings class with no Streamlit import.
"""

import os

import streamlit as st


def _bridge_secrets() -> None:
    """Copy `st.secrets` into the environment.

    Streamlit Community Cloud has no `.env`; locally there is no
    `secrets.toml`. Neither is required to exist, and reading a missing
    secrets file raises, so the whole thing is best-effort by design.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file is the normal local case
        return

    for key, value in secrets.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(key, str(value))


_bridge_secrets()

st.set_page_config(
    page_title="App Review Competitor Tracker",
    page_icon=":material/reviews:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from app.core.logging import configure_logging  # noqa: E402
from ui import input_form, report_view  # noqa: E402
from ui import state as S  # noqa: E402

configure_logging()
S.init()

st.title("App Review Competitor Tracker")

missing = S.settings().missing_credentials()
if missing:
    report_view.render_setup_error(missing)
    st.stop()

report = S.current_report()
error = S.current_error()

if report is not None:
    report_view.render_report(report)
elif error is not None:
    report_view.render_error(*error)
else:
    warning = S.take_warning()
    if warning:
        st.warning(warning, icon=":material/info:")

    pending = S.current_candidates()
    if pending is not None:
        query, candidates = pending
        identity = input_form.render_picker(query, candidates)
        if identity is not None:
            outcome = report_view.run_and_render(identity)
            S.store_outcome(outcome)
            st.rerun()
    else:
        submission = input_form.render()

        if submission.submitted and submission.error:
            st.error(submission.error, icon=":material/error:")
        elif submission.submitted and submission.identity is not None:
            if submission.warning:
                S.set_warning(submission.warning)
            outcome = report_view.run_and_render(submission.identity)
            S.store_outcome(outcome)
            st.rerun()
        elif submission.submitted and S.current_candidates() is not None:
            # Ambiguous query: candidates were stored by input_form, and the
            # picker draws on this rerun instead of a report.
            st.rerun()
