"""Tests for configuration, credential reporting, and log redaction.

Covers E-59 (no `.env` present), E-60 (a real environment variable wins),
E-61 (which key is missing is named), and E-47 (keys never reach the logs).
"""

import logging

import pytest

from app.core.config import Settings
from app.core.logging import _RedactingFilter, redact_secrets


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real `.env` and shell keys out of these assertions."""
    for var in (
        "TAVILY_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "MODEL_PROVIDER",
        "MODEL_NAME",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**overrides: object) -> Settings:
    """Build settings without reading any `.env` file (E-59)."""
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------- #
# Defaults and environment precedence
# --------------------------------------------------------------------------- #


def test_defaults_match_the_architecture():
    settings = _settings()

    assert settings.model_provider == "groq"
    assert settings.search_depth == "basic"
    assert settings.max_search_credits_per_report == 8
    # Groq counts the max_tokens reservation toward its per-minute limit, so the
    # prompt and the cap together have to fit inside 8k.
    assert settings.max_tokens <= 3_000
    assert settings.evidence_char_budget == 12_000
    # The deadline is the guarantee; the 90s median in SC-7 is the target.
    assert settings.run_deadline_seconds > 90


def test_no_env_file_is_not_an_error():
    # E-59: on Streamlit Cloud there is no `.env` at all.
    assert _settings().tavily_api_key is None


def test_environment_variable_is_read(monkeypatch: pytest.MonkeyPatch):
    # E-60: the secrets bridge uses setdefault, so a real variable must win --
    # which only holds if the variable is read in the first place.
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-environment")
    assert _settings().tavily_api_key == "tvly-from-environment"


def test_unknown_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SOMETHING_ELSE_ENTIRELY", "1")
    assert _settings().model_provider == "groq"


# --------------------------------------------------------------------------- #
# Credential reporting (E-61)
# --------------------------------------------------------------------------- #


def test_missing_credentials_names_both_keys():
    assert _settings().missing_credentials() == ["TAVILY_API_KEY", "GROQ_API_KEY"]


def test_missing_credentials_names_only_the_absent_one():
    settings = _settings(tavily_api_key="tvly-present")
    assert settings.missing_credentials() == ["GROQ_API_KEY"]

    settings = _settings(groq_api_key="gsk_present")
    assert settings.missing_credentials() == ["TAVILY_API_KEY"]


def test_missing_credentials_follows_the_selected_provider():
    settings = _settings(
        model_provider="openrouter",
        tavily_api_key="tvly-present",
        groq_api_key="gsk_present",
    )
    # A Groq key is irrelevant when OpenRouter is selected.
    assert settings.missing_credentials() == ["OPENROUTER_API_KEY"]


def test_keyless_providers_need_no_model_credential():
    settings = _settings(model_provider="ollama", tavily_api_key="tvly-present")
    assert settings.missing_credentials() == []


def test_nothing_missing_when_both_are_set():
    settings = _settings(tavily_api_key="tvly-present", groq_api_key="gsk_present")
    assert settings.missing_credentials() == []


# --------------------------------------------------------------------------- #
# Secret redaction (E-47)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "secret",
    [
        "gsk_abcdefghijklmnopqrstuvwx",
        "sk-abcdefghijklmnopqrstuvwx",
        "tvly-devabcdefghijklmnop",
    ],
)
def test_key_shapes_are_redacted(secret: str):
    assert secret not in redact_secrets(f"provider rejected request with {secret}")


def test_labelled_credentials_are_redacted():
    text = redact_secrets("Request failed: api_key=hunter2 Authorization: Bearer xyz123")

    assert "hunter2" not in text
    assert "xyz123" not in text


def test_redaction_keeps_the_rest_of_the_message():
    text = redact_secrets("search failed for acme with gsk_abcdefghijklmnopqrst")

    assert "search failed for acme" in text
    assert "[redacted]" in text


def test_redaction_leaves_ordinary_text_alone():
    message = "resolved acme to acme.com with 5 hits"
    assert redact_secrets(message) == message


def test_filter_redacts_the_formatted_message():
    # The leak that matters is the one nobody wrote by hand: a key arriving
    # through a lazily formatted argument.
    record = logging.LogRecord(
        name="app.search",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="auth failed for %s",
        args=("tvly-devabcdefghijklmnop",),
        exc_info=None,
    )

    assert _RedactingFilter().filter(record) is True
    assert "tvly-dev" not in record.getMessage()
    assert "[redacted]" in record.getMessage()


def test_env_file_path_is_absolute_and_beside_the_app():
    # E-63: `streamlit run` from the repo root, from mvps/, and from the app
    # folder are three different working directories, and Streamlit Cloud uses
    # the repo root. A relative env_file turns that into a silent
    # "missing configuration" screen.
    from app.core.config import _ENV_FILE

    assert _ENV_FILE.is_absolute()
    assert _ENV_FILE.name == ".env"
    assert (_ENV_FILE.parent / "streamlit_app.py").exists()
