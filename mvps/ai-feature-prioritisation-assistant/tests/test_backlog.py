"""Tests for the backlog parser.

The parser is the app's only input surface, and people paste from everywhere, so
the shapes here are drawn from what a backlog actually looks like rather than
from what would be convenient to parse.
"""

import pytest

from app.core.exceptions import BacklogParseError
from app.services.backlog import parse_backlog


def test_one_feature_per_line():
    backlog = parse_backlog("Bulk CSV export\nDark mode\nSSO")

    assert [feature.title for feature in backlog.features] == [
        "Bulk CSV export",
        "Dark mode",
        "SSO",
    ]
    assert [feature.id for feature in backlog.features] == ["F1", "F2", "F3"]


def test_bullets_and_numbers_are_stripped():
    backlog = parse_backlog("- Bulk export\n* Dark mode\n1. SSO\n2) Webhooks")

    assert [feature.title for feature in backlog.features] == [
        "Bulk export",
        "Dark mode",
        "SSO",
        "Webhooks",
    ]


@pytest.mark.parametrize("separator", ["—", "–", "-", ":"])
def test_inline_notes_are_split_from_the_title(separator):
    backlog = parse_backlog(f"Bulk CSV export {separator} sales asks every week, about a sprint")

    feature = backlog.features[0]
    assert feature.title == "Bulk CSV export"
    assert "sales asks every week" in feature.notes


def test_a_hyphen_inside_a_word_is_not_a_separator():
    backlog = parse_backlog("Single-sign-on for enterprise")

    assert backlog.features[0].title == "Single-sign-on for enterprise"
    assert backlog.features[0].notes == ""


def test_blank_lines_make_paragraphs_into_features_with_multiline_notes():
    raw = (
        "Bulk CSV export\n"
        "Sales asks for this every week.\n"
        "Blocked two renewals last quarter.\n"
        "\n"
        "Dark mode\n"
        "Everyone asks, nobody pays for it."
    )

    backlog = parse_backlog(raw)

    assert len(backlog.features) == 2
    assert backlog.features[0].title == "Bulk CSV export"
    assert "Blocked two renewals" in backlog.features[0].notes
    assert backlog.features[1].title == "Dark mode"


def test_a_fully_bulleted_paragraph_is_a_run_of_features_not_one_feature():
    # Someone's notes doc: a heading paragraph, then a bulleted list of ideas.
    raw = "Q3 candidates\nPicked from the support queue.\n\n- Bulk export\n- Dark mode\n- SSO"

    backlog = parse_backlog(raw)

    assert [feature.title for feature in backlog.features] == [
        "Q3 candidates",
        "Bulk export",
        "Dark mode",
        "SSO",
    ]


def test_a_long_unpunctuated_line_keeps_its_tail_as_notes():
    long_title = "Allow administrators to export every record in the workspace " + "word " * 40

    backlog = parse_backlog(long_title)

    feature = backlog.features[0]
    assert len(feature.title) <= 120
    assert feature.notes  # nothing the user wrote is silently dropped
    assert not feature.title.endswith(" ")


def test_the_feature_cap_is_refused_rather_than_truncated(monkeypatch):
    monkeypatch.setenv("MAX_FEATURES", "3")
    from app.core.config import get_settings

    get_settings.cache_clear()

    with pytest.raises(BacklogParseError, match="limit is 3"):
        parse_backlog("A\nB\nC\nD")


@pytest.mark.parametrize("raw", ["", "   ", "\n\n\n"])
def test_empty_input_is_rejected(raw):
    with pytest.raises(BacklogParseError, match="No features found"):
        parse_backlog(raw)


def test_product_context_is_carried_and_capped(monkeypatch):
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "10")
    from app.core.config import get_settings

    get_settings.cache_clear()

    backlog = parse_backlog("Bulk export", product_context="x" * 50)

    assert backlog.product_context == "x" * 10


def test_windows_line_endings_parse_the_same_as_unix():
    assert parse_backlog("A\r\nB\r\nC").features == parse_backlog("A\nB\nC").features
