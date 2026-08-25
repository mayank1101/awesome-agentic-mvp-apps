"""Pydantic models for every boundary in the app.

:class:`GeneratedCode` and :class:`AnalysisAnswer` are parsed straight from
model output and are therefore the app's real validation layer. Everything else
describes a dataset or a result the app computed itself.

The load-bearing design choice is that the model never states a number
directly. It writes pandas code; the app executes that code against the actual
dataframe in a restricted sandbox and keeps the real result; the model's prose
answer is synthesised *from* that result and shown next to it, never instead of
it. See :mod:`app.services.sandbox` and :mod:`app.services.analyzer`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ResultKind = Literal["dataframe", "series", "scalar", "other", "empty"]


class _Strict(BaseModel):
    """Base for parsed model output: unknown keys are dropped, not fatal."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# The dataset, as structure
# --------------------------------------------------------------------------- #


class ColumnProfile(BaseModel):
    """One column's shape, computed by the app -- never asked of a model.

    Attributes:
        name: Column name, verbatim.
        dtype: Pandas dtype as a string (`"int64"`, `"object"`).
        non_null_count: Non-missing values in this column.
        null_count: Missing values in this column.
        unique_count: Distinct non-null values.
        sample_values: A few representative values, stringified and truncated,
            so the model can see real data rather than only a dtype label.
    """

    name: str
    dtype: str
    non_null_count: int
    null_count: int
    unique_count: int
    sample_values: list[str] = Field(default_factory=list)


class CsvProfile(BaseModel):
    """A loaded CSV, summarised.

    Attributes:
        row_count: Rows after any cap was applied.
        column_count: Columns after any cap was applied.
        columns: Per-column stats, in file order.
        preview_rows: The first few rows, every cell stringified and truncated.
        truncated_rows: Whether rows were dropped to fit `max_rows`.
        truncated_columns: Whether columns were dropped to fit `max_columns`.
    """

    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    preview_rows: list[dict[str, str]] = Field(default_factory=list)
    truncated_rows: bool = False
    truncated_columns: bool = False

    def to_prompt_text(self) -> str:
        """Render a compact schema description for a code-generation prompt.

        Returns:
            Column names with dtype, null count, and a few sample values, one
            per line, followed by a small Markdown preview table. Deliberately
            not the full dataset -- the model reasons about shape and sample
            values, and the actual computation runs in the sandbox.
        """
        lines = [f"Rows: {self.row_count}, Columns: {self.column_count}", "", "Columns:"]
        for col in self.columns:
            samples = ", ".join(col.sample_values[:5])
            lines.append(
                f"- `{col.name}` ({col.dtype}), {col.null_count} nulls, "
                f"{col.unique_count} unique. Sample values: {samples}"
            )

        if self.preview_rows:
            lines += ["", "Preview rows:"]
            headers = list(self.preview_rows[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in self.preview_rows:
                lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")

        if self.truncated_rows:
            lines.append("\n(Note: rows were truncated for size; work with what is shown.)")
        if self.truncated_columns:
            lines.append("\n(Note: columns were truncated for size.)")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Model output
# --------------------------------------------------------------------------- #


class GeneratedCode(_Strict):
    """The code-generation call's reply, before it reaches the sandbox.

    Attributes:
        code: Pandas code operating on `df`, assigning its answer to `result`.
        summary: One line describing what the code computes, shown in the UI
            and passed to the answer-synthesis call.
    """

    code: str = ""
    summary: str = ""

    @field_validator("code")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        """Reject empty code, which the sandbox would reject anyway but later."""
        if not value.strip():
            raise ValueError("code must not be empty")
        return value


class AnalysisAnswer(_Strict):
    """The answer-synthesis call's reply.

    Attributes:
        answer: The natural-language answer, grounded in the computed result
            the prompt handed back to the model -- see
            :mod:`app.prompts` for the constraint that produces this.
        caveats: One line of honest hedging, when the result is partial,
            truncated, or the question could not be fully answered by the code.
    """

    answer: str = ""
    caveats: str = ""


# --------------------------------------------------------------------------- #
# The finished answer
# --------------------------------------------------------------------------- #


class QuestionAnswer(BaseModel):
    """One question, fully answered.

    Attributes:
        question: What the user asked.
        code: The pandas code that actually ran.
        summary: The model's one-line description of that code.
        result_kind: What shape the computed result took.
        result_rows: The result rendered as rows of strings, for a dataframe or
            series result. Empty for a scalar or other result.
        result_scalar: The result as a string, for a scalar or "other" result.
        result_row_count: Row count of the *un-truncated* result, so the UI can
            say "showing 50 of 3,214 rows".
        truncated_result: Whether the rendered result was capped.
        answer: The model's natural-language answer.
        caveats: Honest hedging about the answer, when present.
    """

    question: str
    code: str
    summary: str = ""
    result_kind: ResultKind = "empty"
    result_rows: list[dict[str, str]] = Field(default_factory=list)
    result_scalar: str = ""
    result_row_count: int = 0
    truncated_result: bool = False
    answer: str = ""
    caveats: str = ""


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #


class GuardrailFinding(BaseModel):
    """One heuristic match from the input scanner.

    Attributes:
        field: Where the match was found (`"question"`, or a column name).
        pattern: The trigger phrase that matched, for a specific error message.
        severity: `"high"` findings block the run when blocking is enabled.
    """

    field: str
    pattern: str
    severity: Literal["high"] = "high"
