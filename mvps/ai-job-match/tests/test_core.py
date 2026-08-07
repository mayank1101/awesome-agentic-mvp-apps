"""Tests for configuration, logging redaction, and the validation wrapper."""

import logging

import pytest

from app.core.config import get_settings
from app.core.logging import redact
from app.services import validation

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_missing_groq_key_is_a_setup_error() -> None:
    assert get_settings().missing_credentials() == ["GROQ_API_KEY"]


def test_mistral_key_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.missing_credentials() == []
    assert not settings.semantic_matching_available


def test_semantic_matching_flag_follows_the_mistral_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral_test")
    get_settings.cache_clear()

    assert get_settings().semantic_matching_available


def test_settings_are_cached() -> None:
    assert get_settings() is get_settings()


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_groq_keys_are_redacted() -> None:
    assert "gsk_" not in redact("failed with key gsk_abcdefghijklmnop")


def test_authorization_headers_are_redacted() -> None:
    redacted = redact("Authorization: Bearer abc123def456")
    assert "abc123def456" not in redacted


def test_candidate_contact_details_are_redacted() -> None:
    """The input here is a real person's resume; a leaked log line is a leaked resume."""
    redacted = redact("parsed priya.raman@example.com and 98765 43210")

    assert "priya.raman@example.com" not in redacted
    assert "98765 43210" not in redacted


def test_redaction_leaves_ordinary_text_alone() -> None:
    assert redact("Analysis complete: score=68 requirements=12") == (
        "Analysis complete: score=68 requirements=12"
    )


def test_the_filter_redacts_records(caplog: pytest.LogCaptureFixture) -> None:
    from app.core.logging import _RedactingFilter

    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="key gsk_supersecretvalue",
        args=(),
        exc_info=None,
    )
    _RedactingFilter().filter(record)

    assert "supersecret" not in str(record.msg)


# --------------------------------------------------------------------------- #
# Validation wrapper
# --------------------------------------------------------------------------- #


def test_validation_passes_a_faithful_rewrite(resume_text: str) -> None:
    outcome = validation.check_tailored_resume(
        "# Priya Raman\n- Built Django services\n", resume_text
    )

    assert outcome.passed
    assert outcome.violations == []


def test_validation_reports_violations(resume_text: str) -> None:
    outcome = validation.check_tailored_resume("- Shipped Kubernetes operators\n", resume_text)

    assert not outcome.passed
    assert any(v.text == "Kubernetes" for v in outcome.violations)


def test_validation_falls_back_when_guardrails_is_absent(
    resume_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing optional dependency degrades the engine, never the check."""
    monkeypatch.setattr(validation, "_build_guard", lambda: None)
    validation._guard.cache_clear()

    outcome = validation.check_tailored_resume("- Shipped Kubernetes operators\n", resume_text)

    assert outcome.engine == "builtin"
    assert not outcome.passed


def test_validation_survives_a_broken_guard(
    resume_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Exploding:
        def validate(self, *_: object, **__: object) -> None:
            raise RuntimeError("guardrails internals moved")

    monkeypatch.setattr(validation, "_build_guard", _Exploding)
    validation._guard.cache_clear()

    outcome = validation.check_tailored_resume("# Priya Raman\n", resume_text)

    assert outcome.engine == "builtin"
    assert outcome.passed


def test_active_markdown_fails_validation(resume_text: str) -> None:
    outcome = validation.check_tailored_resume("<script>alert(1)</script>", resume_text)

    assert outcome.unsafe_markdown
    assert not outcome.passed


# --------------------------------------------------------------------------- #
# Schema hygiene
# --------------------------------------------------------------------------- #


def test_a_textual_null_is_treated_as_absent() -> None:
    """Seen live: a posting with no company came back as the string "null"."""
    from app.models.schemas import JobPosting

    posting = JobPosting.model_validate(
        {"title": "Senior AI/ML Engineer", "company": "null", "seniority": "N/A"}
    )

    assert posting.company == ""
    assert posting.seniority == ""


def test_real_values_survive_the_nullish_check() -> None:
    from app.models.schemas import JobPosting

    posting = JobPosting.model_validate({"title": "Analyst", "company": "Nullsoft"})

    assert posting.company == "Nullsoft"
