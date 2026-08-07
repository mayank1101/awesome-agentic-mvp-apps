"""Tests for query normalisation."""

from app.services.normalize import is_meaningful_query, normalize_query


def test_collapses_whitespace():
    assert normalize_query("  Notion   Calendar  ") == "Notion Calendar"


def test_strips_invisible_characters():
    assert normalize_query("Not​ion") == "Notion"


def test_truncates_at_max_chars():
    assert len(normalize_query("x" * 500, max_chars=50)) == 50


def test_non_latin_names_are_preserved():
    assert normalize_query("日本語アプリ") == "日本語アプリ"


def test_meaningful_query_requires_alnum():
    assert is_meaningful_query("Notion") is True
    assert is_meaningful_query("!!! ---") is False
    assert is_meaningful_query("") is False
