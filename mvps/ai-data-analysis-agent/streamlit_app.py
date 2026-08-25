"""Entry point.

    streamlit run streamlit_app.py

Thin on purpose: this module owns the secrets bridge, page setup, and the run
order. The widgets live in :mod:`ui`, everything else in :mod:`app`.
"""

import os

import streamlit as st


def _bridge_secrets() -> None:
    """Copy `st.secrets` into the environment.

    Streamlit Community Cloud has no `.env`; locally there is usually no
    `secrets.toml`. Neither is required to exist, and reading a missing secrets
    file raises, so the whole thing is best-effort by design.
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
    page_title="AI Data Analysis Agent",
    page_icon=":material/monitoring:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from app.core.logging import configure_logging  # noqa: E402
from ui import input_form, result_view  # noqa: E402
from ui import state as S  # noqa: E402

configure_logging()
S.init()

st.title("AI Data Analysis Agent")

missing = S.settings().missing_credentials()
if missing:
    result_view.render_setup_error(missing)
    st.stop()

dataset = S.dataset()

if dataset is None:
    submission = input_form.render_upload()
    if submission.submitted and submission.error:
        st.error(submission.error, icon=":material/error:")
    elif submission.submitted:
        st.rerun()
else:
    _, profile, filename = dataset
    result_view.render_dataset_summary(profile, filename)
    st.divider()
    result_view.render_history(S.history())
    question = input_form.render_question(disabled=S.busy())
    if question:
        result_view.run_question(question)
        st.rerun()
