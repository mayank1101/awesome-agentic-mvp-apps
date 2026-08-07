import logging

import pytest

from app.core import logging as app_logging
from app.core.config import DEFAULT_JOB_SITES, get_settings
from app.services import sites


def test_both_service_keys_are_required() -> None:
    assert get_settings().missing_credentials() == ["GROQ_API_KEY", "TAVILY_API_KEY"]


def test_the_embedding_key_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.missing_credentials() == []
    assert settings.semantic_available is False


def test_the_default_site_list_is_usable_as_a_whitelist() -> None:
    accepted, rejected = sites.normalize_sites(list(DEFAULT_JOB_SITES))

    assert rejected == []
    assert len(accepted) == len(DEFAULT_JOB_SITES)


@pytest.mark.parametrize(
    "secret",
    [
        "gsk_abcdefghijklmnop",
        "tvly-abcdefghijklmnop",
        "api_key=abcdefghijklmnop",
        "Authorization: Bearer abcdefghijklmnop",
    ],
)
def test_credentials_are_redacted(secret: str) -> None:
    assert "abcdefghijklmnop" not in app_logging.redact(f"call failed: {secret}")


def test_contact_details_are_redacted() -> None:
    redacted = app_logging.redact("priya.raman@example.com called from +91 98765 43210")

    assert "priya.raman@example.com" not in redacted
    assert "98765 43210" not in redacted


def test_the_filter_redacts_a_real_record() -> None:
    record = logging.LogRecord(
        name="app.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="rejected key %s",
        args=("tvly-abcdefghijklmnop",),
        exc_info=None,
    )

    app_logging._RedactingFilter().filter(record)

    assert "abcdefghijklmnop" not in record.getMessage()
