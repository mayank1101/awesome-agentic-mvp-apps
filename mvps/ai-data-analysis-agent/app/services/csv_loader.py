"""Turning an uploaded CSV into a dataframe and a schema profile.

The app is built for a *clean* CSV, per the product scope -- there is no header
inference heuristics, no encoding sniffing beyond what pandas does by default,
and a malformed file is a stated failure rather than a best-effort repair. That
mirrors the resume app's stance on scanned PDFs elsewhere in this repo: some
inputs are out of scope, and saying so immediately beats a wrong analysis.
"""

import io

import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import CsvEmptyError, CsvParseError, CsvTooLargeError
from app.core.logging import get_logger
from app.models.schemas import ColumnProfile, CsvProfile

logger = get_logger(__name__)


def _truncate_cell(value: object, cap: int) -> str:
    """Render one cell as a string, capped for prompt and display budgets."""
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= cap else text[: cap - 1] + "…"


def load_csv(data: bytes) -> tuple[pd.DataFrame, CsvProfile]:
    """Parse CSV bytes into a dataframe and a schema profile.

    Args:
        data: The uploaded file's raw bytes.

    Returns:
        The loaded dataframe (capped to the configured row/column limits) and a
        :class:`CsvProfile` describing it.

    Raises:
        CsvTooLargeError: The upload exceeds the byte cap.
        CsvEmptyError: The file parsed but has no rows or no columns.
        CsvParseError: The bytes are not readable as CSV.
    """
    settings = get_settings()

    if len(data) > settings.max_upload_bytes:
        raise CsvTooLargeError(
            f"That file is {len(data) / 1_048_576:.1f} MB. "
            f"The limit is {settings.max_upload_bytes / 1_048_576:.0f} MB."
        )

    try:
        df = pd.read_csv(io.BytesIO(data))
    except pd.errors.EmptyDataError as exc:
        raise CsvEmptyError("That file has no data to read.") from exc
    except UnicodeDecodeError as exc:
        raise CsvParseError(
            "That file could not be decoded as text. Export it as UTF-8 CSV and try again."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - pandas raises assorted parse errors
        raise CsvParseError(
            "That file could not be read as CSV. Check that it is a plain, "
            "comma-separated file with a header row."
        ) from exc

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise CsvEmptyError("That CSV has no rows, or no columns, to analyse.")

    truncated_rows = len(df) > settings.max_rows
    if truncated_rows:
        df = df.head(settings.max_rows).copy()
        logger.info("CSV truncated to %d rows", settings.max_rows)

    truncated_columns = df.shape[1] > settings.max_columns
    if truncated_columns:
        df = df.iloc[:, : settings.max_columns].copy()
        logger.info("CSV truncated to %d columns", settings.max_columns)

    profile = _build_profile(df, truncated_rows=truncated_rows, truncated_columns=truncated_columns)
    return df, profile


def _build_profile(
    df: pd.DataFrame, *, truncated_rows: bool, truncated_columns: bool
) -> CsvProfile:
    """Compute per-column stats and a row preview, entirely in code."""
    settings = get_settings()
    cap = settings.max_cell_chars

    columns: list[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        non_null = series.dropna()
        samples = [_truncate_cell(v, cap) for v in non_null.unique()[:5]]
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                non_null_count=int(series.notna().sum()),
                null_count=int(series.isna().sum()),
                unique_count=int(series.nunique(dropna=True)),
                sample_values=samples,
            )
        )

    preview = df.head(settings.preview_rows)
    preview_rows = [
        {str(col): _truncate_cell(row[col], cap) for col in preview.columns}
        for _, row in preview.iterrows()
    ]

    return CsvProfile(
        row_count=len(df),
        column_count=df.shape[1],
        columns=columns,
        preview_rows=preview_rows,
        truncated_rows=truncated_rows,
        truncated_columns=truncated_columns,
    )
