import pytest

from app.core.exceptions import CsvEmptyError, CsvError, CsvParseError, CsvTooLargeError
from app.services.csv_loader import load_csv


def test_loads_a_clean_csv(sample_csv_bytes: bytes) -> None:
    df, profile = load_csv(sample_csv_bytes)

    assert list(df.columns) == ["category", "revenue"]
    assert profile.row_count == 6
    assert profile.column_count == 2
    assert not profile.truncated_rows
    assert not profile.truncated_columns


def test_column_stats_are_computed(sample_csv_bytes: bytes) -> None:
    _, profile = load_csv(sample_csv_bytes)
    revenue = next(c for c in profile.columns if c.name == "revenue")

    assert revenue.dtype == "int64"
    assert revenue.null_count == 0
    assert revenue.non_null_count == 6


def test_preview_rows_are_capped_to_setting(
    monkeypatch: pytest.MonkeyPatch, sample_csv_bytes: bytes
) -> None:
    from app.core.config import Settings

    settings = Settings(groq_api_key="x", preview_rows=2)
    monkeypatch.setattr("app.services.csv_loader.get_settings", lambda: settings)

    _, profile = load_csv(sample_csv_bytes)
    assert len(profile.preview_rows) == 2


def test_empty_file_is_rejected() -> None:
    with pytest.raises(CsvEmptyError):
        load_csv(b"")


def test_header_only_csv_is_rejected() -> None:
    with pytest.raises(CsvEmptyError):
        load_csv(b"a,b,c\n")


def test_unreadable_bytes_are_rejected() -> None:
    # Binary garbage with no rows pandas can make sense of: some byte
    # sequences surface as an empty parse rather than a decode error, so the
    # meaningful assertion is "rejected as unusable", not the exact subtype.
    with pytest.raises(CsvError):
        load_csv(b"\x00\x01\x02not,a,csv\x00\x00")


def test_invalid_utf8_is_rejected() -> None:
    with pytest.raises(CsvParseError):
        load_csv(b"a,b\n\xff\xfe,2\n")


def test_oversized_upload_is_rejected(
    monkeypatch: pytest.MonkeyPatch, sample_csv_bytes: bytes
) -> None:
    from app.core.config import Settings

    settings = Settings(groq_api_key="x", max_upload_bytes=10)
    monkeypatch.setattr("app.services.csv_loader.get_settings", lambda: settings)

    with pytest.raises(CsvTooLargeError):
        load_csv(sample_csv_bytes)


def test_rows_are_truncated_to_the_cap(
    monkeypatch: pytest.MonkeyPatch, sample_csv_bytes: bytes
) -> None:
    from app.core.config import Settings

    settings = Settings(groq_api_key="x", max_rows=3)
    monkeypatch.setattr("app.services.csv_loader.get_settings", lambda: settings)

    df, profile = load_csv(sample_csv_bytes)
    assert len(df) == 3
    assert profile.row_count == 3
    assert profile.truncated_rows
