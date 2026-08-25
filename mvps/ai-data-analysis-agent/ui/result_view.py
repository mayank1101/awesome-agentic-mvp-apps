"""The dataset summary, the chat-style question history, and error screens.

The result of every question is shown two ways at once: the model's prose
answer, and the actual computed table or value underneath it, in an expander
alongside the code that produced it. The prose is commentary; the table is the
evidence -- the same "a number is never shown alone" rule this repo's other
apps apply to their own scores.
"""

import pandas as pd
import streamlit as st

from app.core.exceptions import (
    CodeError,
    DataAnalysisError,
    ModelQuotaExhausted,
    ModelRateLimited,
    QuestionBlocked,
    RunDeadlineExceeded,
)
from app.models.schemas import CsvProfile, QuestionAnswer
from app.services.analyzer import analyze
from ui import state as S

# --------------------------------------------------------------------------- #
# Running a question
# --------------------------------------------------------------------------- #


def run_question(question: str) -> None:
    """Answer one question and append the outcome to history.

    Every failure becomes a :class:`~ui.state.HistoryItem` with an error
    message rather than a traceback, so a failed question does not lose the
    dataset or the questions answered before it.
    """
    if S.busy():
        return

    dataset = S.dataset()
    if dataset is None:
        return
    df, profile, _ = dataset

    S.set_busy(True)
    status = st.status("Answering…", expanded=True)
    try:
        result = analyze(df, profile, question, progress=lambda m: status.update(label=m))
    except QuestionBlocked as exc:
        status.update(label="Blocked", state="error")
        detail = "; ".join(f"{f.field}: {f.pattern}" for f in exc.findings)
        S.append_history(
            S.HistoryItem(
                question=question,
                error=f"{exc} ({detail})" if detail else str(exc),
            )
        )
    except (ModelRateLimited, ModelQuotaExhausted) as exc:
        status.update(label="Provider limit reached", state="error")
        S.append_history(S.HistoryItem(question=question, error=str(exc)))
    except RunDeadlineExceeded as exc:
        status.update(label="Timed out", state="error")
        S.append_history(S.HistoryItem(question=question, error=str(exc)))
    except CodeError as exc:
        status.update(label="Could not run", state="error")
        S.append_history(
            S.HistoryItem(
                question=question,
                error=f"The analysis code could not run: {exc}",
            )
        )
    except DataAnalysisError as exc:
        status.update(label="Failed", state="error")
        S.append_history(S.HistoryItem(question=question, error=str(exc)))
    else:
        status.update(label="Done", state="complete")
        S.append_history(S.HistoryItem(question=question, answer=result))
    finally:
        S.set_busy(False)


# --------------------------------------------------------------------------- #
# Dataset summary
# --------------------------------------------------------------------------- #


def render_dataset_summary(profile: CsvProfile, filename: str) -> None:
    """Row/column counts and per-column stats, collapsed by default."""
    left, right = st.columns([3, 1])
    with left:
        st.caption(
            f"📄 **{filename}** — {profile.row_count:,} rows × {profile.column_count} columns"
        )
    with right:
        if st.button("Upload a different file", use_container_width=True):
            S.clear_dataset()
            st.rerun()

    if profile.truncated_rows or profile.truncated_columns:
        parts = " and ".join(
            p
            for p, flag in (
                ("rows", profile.truncated_rows),
                ("columns", profile.truncated_columns),
            )
            if flag
        )
        st.caption(f"⚠️ This file's {parts} were truncated to fit this app's limits.")

    with st.expander("Columns", expanded=False):
        table = pd.DataFrame(
            [
                {
                    "Column": c.name,
                    "Type": c.dtype,
                    "Nulls": c.null_count,
                    "Unique": c.unique_count,
                    "Sample values": ", ".join(c.sample_values[:3]),
                }
                for c in profile.columns
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Question history
# --------------------------------------------------------------------------- #


def render_history(items: list[S.HistoryItem]) -> None:
    """Draw every asked question as a chat turn, most recent last."""
    for item in items:
        with st.chat_message("user"):
            st.write(item.question)
        with st.chat_message("assistant"):
            if item.error:
                st.error(item.error)
            elif item.answer:
                _render_answer(item.answer)


def _render_answer(answer: QuestionAnswer) -> None:
    """One answer: the prose, then the evidence behind it."""
    st.write(answer.answer)
    if answer.caveats:
        st.caption(f"⚠️ {answer.caveats}")

    with st.expander("How this was computed"):
        if answer.summary:
            st.caption(answer.summary)
        st.code(answer.code, language="python")

        if answer.result_kind in ("dataframe", "series") and answer.result_rows:
            st.dataframe(
                pd.DataFrame(answer.result_rows), use_container_width=True, hide_index=True
            )
            if answer.truncated_result:
                st.caption(
                    f"Showing {len(answer.result_rows)} of {answer.result_row_count:,} result rows."
                )
        elif answer.result_kind == "scalar":
            st.write(f"Result: `{answer.result_scalar}`")
        elif answer.result_kind == "other":
            st.text(answer.result_scalar)
        else:
            st.caption("The computation produced no rows.")


# --------------------------------------------------------------------------- #
# Setup screen
# --------------------------------------------------------------------------- #


def render_setup_error(missing: list[str]) -> None:
    """Draw the startup screen for missing credentials."""
    st.error(
        "**This app is not configured yet.**\n\n"
        "Set the following before running it:\n\n" + "\n".join(f"- `{name}`" for name in missing)
    )
    st.caption(
        "Locally, put them in a `.env` beside `streamlit_app.py`. On Streamlit "
        "Community Cloud, put them in the app's Secrets."
    )
