"""Orchestrates one question: generate code, run it, explain the result.

The pipeline is deliberately linear and each step's failure mode is distinct:

1. Guard the question and a sample of the dataset's own text against obvious
   prompt-injection phrasing (:mod:`app.services.guardrails`).
2. Ask the model for pandas code (:mod:`app.prompts`, :mod:`app.services.llm`).
3. Run that code in the sandbox (:mod:`app.services.sandbox`). A validation or
   runtime failure here gets exactly one repair attempt, shown the error.
4. Format the real result in code -- never asked of the model.
5. Ask the model to explain that real result in a sentence or two.

Step 4 is the app's actual guarantee: whatever the answer says, the numbers
under it came from pandas running against the real dataframe, not from the
model's own arithmetic.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import CodeError, RunDeadlineExceeded
from app.core.logging import get_logger
from app.models.schemas import AnalysisAnswer, CsvProfile, GeneratedCode, QuestionAnswer, ResultKind
from app.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    CODE_REPAIR_TEMPLATE,
    CODE_SYSTEM_PROMPT,
    CODE_USER_TEMPLATE,
)
from app.services import guardrails, llm
from app.services.sandbox import run_code

logger = get_logger(__name__)

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    return None


def _sample_texts(profile: CsvProfile) -> dict[str, str]:
    """Column name to a joined string of its sample values, for guardrail scanning."""
    return {col.name: " ".join(col.sample_values) for col in profile.columns if col.sample_values}


def _check_deadline(started_at: float, deadline: float) -> None:
    if time.monotonic() - started_at > deadline:
        raise RunDeadlineExceeded(
            f"This question took longer than the {deadline:.0f}s limit for one run."
        )


def _format_result(
    result: Any, *, max_rows: int
) -> tuple[ResultKind, list[dict[str, str]], str, int, bool]:
    """Turn whatever `result` came back as into a renderable, bounded shape.

    Returns:
        A tuple of (kind, rows, scalar_text, true_row_count, truncated).
    """
    if isinstance(result, pd.DataFrame):
        true_count = len(result)
        shown = result.head(max_rows)
        rows = [
            {str(c): "" if pd.isna(v) else str(v) for c, v in row.items()}
            for _, row in shown.iterrows()
        ]
        return "dataframe", rows, "", true_count, true_count > max_rows

    if isinstance(result, pd.Series):
        true_count = len(result)
        shown = result.head(max_rows)
        rows = [
            {"index": str(idx), "value": "" if pd.isna(v) else str(v)} for idx, v in shown.items()
        ]
        return "series", rows, "", true_count, true_count > max_rows

    if result is None:
        return "empty", [], "None", 0, False

    if isinstance(result, np.generic):
        result = result.item()

    if isinstance(result, (int, float, str, bool)):
        return "scalar", [], str(result), 1, False

    text = str(result)
    return "other", [], text[:2000], 1, len(text) > 2000


def _result_text_for_prompt(kind: ResultKind, rows: list[dict[str, str]], scalar: str) -> str:
    if kind in ("dataframe", "series"):
        if not rows:
            return "(no rows)"
        headers = list(rows[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines += ["| " + " | ".join(r.get(h, "") for h in headers) + " |" for r in rows]
        return "\n".join(lines)
    return scalar


def analyze(
    df: pd.DataFrame,
    profile: CsvProfile,
    question: str,
    *,
    progress: ProgressFn | None = None,
) -> QuestionAnswer:
    """Answer one question against a loaded dataset.

    Args:
        df: The loaded dataframe.
        profile: Its schema profile, used to build the code-generation prompt.
        question: The user's question, already length-checked by the caller.
        progress: Optional callback for UI status updates.

    Returns:
        The finished :class:`QuestionAnswer`.

    Raises:
        QuestionBlocked: Guardrail scanning found a high-severity match.
        CodeError: Generated code failed validation or execution, twice.
        ModelError: Any provider failure the retries could not absorb.
        RunDeadlineExceeded: The run exceeded its time budget.
    """
    report = progress or _noop
    settings = get_settings()
    started_at = time.monotonic()

    report("Checking the question…")
    guardrails.guard(question, _sample_texts(profile))
    _check_deadline(started_at, settings.run_deadline_seconds)

    report("Writing analysis code…")
    schema_text = profile.to_prompt_text()
    generated = llm.complete_model(
        system=CODE_SYSTEM_PROMPT,
        user=CODE_USER_TEMPLATE.format(schema=schema_text, question=question),
        schema=GeneratedCode,
        max_tokens=settings.max_tokens_code,
    )
    _check_deadline(started_at, settings.run_deadline_seconds)

    report("Running the analysis…")
    try:
        result = run_code(generated.code, df, timeout_seconds=settings.code_timeout_seconds)
    except CodeError as first_error:
        logger.warning("First code attempt failed: %s", type(first_error).__name__)
        report("First attempt failed, retrying…")
        repair_user = CODE_REPAIR_TEMPLATE.format(
            previous=CODE_USER_TEMPLATE.format(schema=schema_text, question=question),
            code=generated.code,
            error=str(first_error),
        )
        generated = llm.complete_model(
            system=CODE_SYSTEM_PROMPT,
            user=repair_user,
            schema=GeneratedCode,
            max_tokens=settings.max_tokens_code,
            temperature=0.0,
        )
        _check_deadline(started_at, settings.run_deadline_seconds)
        result = run_code(generated.code, df, timeout_seconds=settings.code_timeout_seconds)

    kind, rows, scalar, true_count, truncated = _format_result(
        result, max_rows=settings.max_output_rows
    )
    _check_deadline(started_at, settings.run_deadline_seconds)

    report("Writing the answer…")
    row_note = (
        f"showing {len(rows)} of {true_count} rows" if kind in ("dataframe", "series") else "scalar"
    )
    answer = llm.complete_model(
        system=ANSWER_SYSTEM_PROMPT,
        user=ANSWER_USER_TEMPLATE.format(
            question=question,
            summary=generated.summary or "(no summary given)",
            result_kind=kind,
            row_note=row_note,
            result_text=_result_text_for_prompt(kind, rows, scalar),
        ),
        schema=AnalysisAnswer,
        max_tokens=settings.max_tokens_answer,
    )

    return QuestionAnswer(
        question=question,
        code=generated.code,
        summary=generated.summary,
        result_kind=kind,
        result_rows=rows,
        result_scalar=scalar,
        result_row_count=true_count,
        truncated_result=truncated,
        answer=answer.answer,
        caveats=answer.caveats,
    )
