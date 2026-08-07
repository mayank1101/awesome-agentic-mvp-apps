"""Tests for the input boundary.

Covers E-01 (empty or punctuation-only), E-02 (homoglyphs and invisibles),
E-03 (non-Latin names survive intact), E-04 (cap truncates, does not reject),
E-05 (domain forms and rejected schemes), and E-55 (filename slugs).
"""

import pytest

from app.services.normalize import (
    InvalidDomainError,
    is_meaningful_name,
    normalize_domain,
    normalize_name,
    registrable_host,
    slugify,
)

CAP = 120


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def test_plain_name_passes_through():
    assert normalize_name("  Notion  ", max_chars=CAP) == "Notion"


def test_zero_width_characters_are_stripped():
    # E-02: invisible characters defeat search and would slip past a literal
    # fence-marker comparison later in the pipeline.
    assert normalize_name("No​tion‍", max_chars=CAP) == "Notion"


def test_bidirectional_override_is_stripped():
    assert normalize_name("Acme‮", max_chars=CAP) == "Acme"


def test_control_characters_are_stripped_but_spacing_survives():
    assert normalize_name("Acme\x00\x07 Corp", max_chars=CAP) == "Acme Corp"


def test_nfkc_folds_compatibility_forms():
    # Fullwidth Latin is a real thing users paste from Japanese input methods.
    assert normalize_name("Ｎｏｔｉｏｎ", max_chars=CAP) == "Notion"


@pytest.mark.parametrize("name", ["日本電気", "Яндекс", "Grüner Punkt", "Épicerie"])
def test_non_latin_names_are_not_mangled(name: str):
    # E-03: NFKC only. Transliteration would break the app for every non-Latin
    # market, and the search provider handles these fine.
    assert normalize_name(name, max_chars=CAP) == name


def test_whitespace_runs_collapse():
    assert normalize_name("Acme   \n  Corp", max_chars=CAP) == "Acme Corp"


def test_long_name_is_truncated_not_rejected():
    # E-04
    result = normalize_name("A" * 500, max_chars=CAP)
    assert len(result) == CAP


@pytest.mark.parametrize("raw", ["", "   ", "​​", "!!!", "--- ...", "•"])
def test_empty_or_punctuation_only_is_not_meaningful(raw: str):
    # E-01: refused before a credit is spent.
    assert not is_meaningful_name(normalize_name(raw, max_chars=CAP))


@pytest.mark.parametrize("raw", ["Notion", "37signals", "X", "日本電気"])
def test_real_names_are_meaningful(raw: str):
    assert is_meaningful_name(normalize_name(raw, max_chars=CAP))


# --------------------------------------------------------------------------- #
# Domains (E-05)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("example.com", "example.com"),
        ("  Example.COM  ", "example.com"),
        ("www.example.com", "example.com"),
        ("https://example.com", "example.com"),
        ("http://www.example.com/pricing?utm=1", "example.com"),
        ("https://example.com:8443/path", "example.com"),
        ("https://user:pass@example.com/x", "example.com"),
        ("example.com/", "example.com"),
        ("example.com.", "example.com"),
        ("sub.example.co.uk", "sub.example.co.uk"),
    ],
)
def test_domain_forms_normalise_to_a_bare_host(raw: str, expected: str):
    assert normalize_domain(raw) == expected


def test_internationalised_domain_is_idna_encoded():
    assert normalize_domain("münchen.de") == "xn--mnchen-3ya.de"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_blank_domain_is_none(raw: str | None):
    assert normalize_domain(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html,<script>alert(1)</script>",
        "ftp://example.com",
    ],
)
def test_disallowed_schemes_are_rejected(raw: str):
    # Rejected, not sanitised: a value that had to be cleaned to be safe is a
    # value the user should be told about.
    with pytest.raises(InvalidDomainError):
        normalize_domain(raw)


@pytest.mark.parametrize("raw", ["not a domain", "localhost", "...", "example", "-.com"])
def test_non_hosts_are_rejected(raw: str):
    with pytest.raises(InvalidDomainError):
        normalize_domain(raw)


# --------------------------------------------------------------------------- #
# Host comparison and slugs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.notion.so/pricing", "notion.so"),
        ("https://help.linear.app/docs", "linear.app"),
        ("http://EXAMPLE.com:80/x", "example.com"),
        ("https://example.com", "example.com"),
    ],
)
def test_registrable_host(url: str, expected: str):
    assert registrable_host(url) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Notion", "notion"),
        ("37signals", "37signals"),
        ("Acme Corp / Ltd.", "acme-corp-ltd"),
        ("Grüner Punkt", "gruner-punkt"),
        ("  spaced  out  ", "spaced-out"),
    ],
)
def test_slugify(text: str, expected: str):
    assert slugify(text) == expected


@pytest.mark.parametrize("text", ["日本電気", "🙂", "///"])
def test_slugify_falls_back_when_nothing_survives(text: str):
    # E-55: a download called `-.md` helps nobody.
    assert slugify(text) == "competitor"
