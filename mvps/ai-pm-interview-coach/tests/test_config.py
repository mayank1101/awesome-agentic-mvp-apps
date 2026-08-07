"""Settings load, validate, and cache.

Every test constructs ``Settings`` with ``_env_file=None`` so a developer's real
`.env` cannot change the result. Without that, these tests pass or fail
depending on whose machine they run on.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults_match_documented_values():
    settings = _settings()

    assert settings.model_provider == "openrouter"
    assert settings.structured_output_mode == "json_object"
    # The interviewer cap is deliberately far below the document-wide one: a
    # question is a question, and headroom is what lets a model lecture.
    assert settings.interviewer_max_tokens < settings.max_tokens
    assert settings.report_max_tokens == 4096
    assert settings.max_answer_chars == 4000
    assert settings.stream_timeout_seconds == 90
    assert settings.guardrails_enabled is True
    assert settings.block_flagged_input is True


def test_unknown_provider_is_rejected():
    with pytest.raises(ValidationError):
        _settings(model_provider="not-a-provider")


def test_unknown_structured_output_mode_is_rejected():
    with pytest.raises(ValidationError):
        _settings(structured_output_mode="yaml")


@pytest.mark.parametrize(
    "field",
    [
        "max_tokens",
        "interviewer_max_tokens",
        "report_max_tokens",
        "max_answer_chars",
        "stream_timeout_seconds",
    ],
)
def test_positive_only_fields_reject_zero(field):
    with pytest.raises(ValidationError):
        _settings(**{field: 0})


@pytest.mark.parametrize(
    "field",
    [
        "min_answer_hint_chars",
        "max_interviews_per_session",
    ],
)
def test_zero_is_allowed_where_it_means_disabled(field):
    # 0 is a meaningful value for both: no hint, and no interview cap. Rejecting it
    # would remove the off switch.
    assert getattr(_settings(**{field: 0}), field) == 0


@pytest.mark.parametrize("field", ["max_tokens", "max_answer_chars", "stream_timeout_seconds"])
def test_negative_values_are_rejected(field):
    with pytest.raises(ValidationError):
        _settings(**{field: -1})


def test_temperature_is_bounded():
    with pytest.raises(ValidationError):
        _settings(model_temperature=-0.1)
    with pytest.raises(ValidationError):
        _settings(model_temperature=2.1)


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setenv("MAX_ANSWER_CHARS", "1234")

    settings = _settings()

    assert settings.model_provider == "gemini"
    assert settings.max_answer_chars == 1234


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
