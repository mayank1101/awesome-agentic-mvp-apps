"""Tests for settings loading.

Chiefly a guard against the defaults drifting somewhere they cannot run: the
`.env.example` provider has to be one a reader can use for free, and the token
cap has to be large enough for a full backlog rather than a single answer.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_defaults_are_free_tier_runnable():
    settings = Settings()

    assert settings.model_provider == "groq"
    assert settings.model_name


def test_the_token_cap_is_sized_for_a_full_backlog():
    settings = Settings()

    # Roughly 120 tokens of JSON per feature, and the cap must not truncate the
    # tail of the list -- a truncated reply loses features silently.
    assert settings.max_tokens >= settings.max_features * 120


def test_temperature_is_low_because_the_task_is_classification():
    assert Settings().model_temperature <= 0.3


def test_an_unknown_provider_is_rejected_at_load():
    with pytest.raises(ValidationError):
        Settings(model_provider="not-a-provider")


def test_settings_are_cached_per_process():
    assert get_settings() is get_settings()


def test_environment_overrides_the_default(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    get_settings.cache_clear()

    assert get_settings().model_provider == "ollama"
