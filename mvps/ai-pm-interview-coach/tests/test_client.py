"""Provider dialects, per-call options, and the credential preflight.

No network and no SDK construction: every test drives the pure translation
functions with patched settings. The dialect table is worth this much attention
because getting it wrong degrades silently -- a provider that ignores an
unrecognised `response_format` returns prose, and the failure surfaces much later
as a parse error.
"""

import pytest

from app.agents import client as client_module
from app.agents.client import (
    _JSON_OBJECT_DIALECT,
    _REQUIRED_CREDENTIAL,
    _SCHEMA_ONLY_PROVIDERS,
    build_options,
    preflight,
    structured_response_format,
)
from app.core.config import ModelProvider, Settings, get_settings
from app.core.exceptions import PreflightError
from app.models.schemas import FeedbackReport

ALL_PROVIDERS = list(ModelProvider.__args__)


@pytest.fixture
def patch_settings(monkeypatch):
    """Replace the cached settings with an isolated instance per test."""

    def _apply(**overrides) -> Settings:
        settings = Settings(_env_file=None, **overrides)
        monkeypatch.setattr("app.agents.client.get_settings", lambda: settings)
        return settings

    return _apply


# --- structured output dialects ----------------------------------------------


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_prompt_mode_sends_no_response_format(provider, patch_settings):
    patch_settings(model_provider=provider, structured_output_mode="prompt")
    assert structured_response_format(FeedbackReport) is None


@pytest.mark.parametrize("provider", ALL_PROVIDERS)
def test_json_schema_mode_always_sends_a_schema(provider, patch_settings):
    patch_settings(model_provider=provider, structured_output_mode="json_schema")
    assert structured_response_format(FeedbackReport) is not None


@pytest.mark.parametrize("provider", ["openrouter", "openai", "foundry"])
def test_openai_dialect_uses_json_object(provider, patch_settings):
    patch_settings(model_provider=provider, structured_output_mode="json_object")
    assert structured_response_format(FeedbackReport) == {"type": "json_object"}


def test_ollama_spells_json_object_as_a_bare_string(patch_settings):
    patch_settings(model_provider="ollama", structured_output_mode="json_object")
    assert structured_response_format(FeedbackReport) == "json"


@pytest.mark.parametrize("provider", sorted(_SCHEMA_ONLY_PROVIDERS))
def test_schema_only_providers_get_a_schema_under_json_object(provider, patch_settings):
    # Sending the schema is *stronger* than what was asked for, not weaker: the
    # alternative for these two would be prompt-only.
    patch_settings(model_provider=provider, structured_output_mode="json_object")
    assert structured_response_format(FeedbackReport) is not None


def test_gemini_receives_a_mapping_not_the_class(patch_settings):
    # Gemini's client only forwards a mapping to response_schema; handing it the
    # class yields a JSON mime type with no server-side schema at all.
    patch_settings(model_provider="gemini", structured_output_mode="json_schema")
    payload = structured_response_format(FeedbackReport)

    assert isinstance(payload, dict)
    assert "schema" in payload
    assert payload["schema"]["title"] == "FeedbackReport"


@pytest.mark.parametrize("provider", ["openrouter", "openai", "foundry", "anthropic", "ollama"])
def test_non_gemini_providers_receive_the_class(provider, patch_settings):
    patch_settings(model_provider=provider, structured_output_mode="json_schema")
    assert structured_response_format(FeedbackReport) is FeedbackReport


def test_every_provider_is_covered_by_a_dialect_or_is_schema_only():
    # A provider in neither table would silently return None under json_object,
    # which reads as "prompt mode" and loses structured output without an error.
    covered = set(_JSON_OBJECT_DIALECT) | set(_SCHEMA_ONLY_PROVIDERS)
    assert covered == set(ALL_PROVIDERS)


# --- per-call options ---------------------------------------------------------


def test_build_options_applies_the_configured_temperature(patch_settings):
    patch_settings(model_temperature=0.9)
    assert build_options()["temperature"] == 0.9


def test_build_options_defaults_to_the_document_wide_cap(patch_settings):
    settings = patch_settings()
    assert build_options()["max_tokens"] == settings.max_tokens


def test_build_options_honours_an_explicit_cap(patch_settings):
    settings = patch_settings()
    assert build_options(max_tokens=settings.interviewer_max_tokens)["max_tokens"] == (
        settings.interviewer_max_tokens
    )


def test_build_options_omits_response_format_when_none(patch_settings):
    patch_settings()
    assert "response_format" not in build_options()


def test_build_options_includes_response_format_when_given(patch_settings):
    patch_settings()
    assert build_options(response_format={"type": "json_object"})["response_format"] == {
        "type": "json_object"
    }


# --- preflight ---------------------------------------------------------------


@pytest.mark.parametrize(("provider", "env_var"), sorted(_REQUIRED_CREDENTIAL.items()))
def test_preflight_names_the_missing_credential(provider, env_var, patch_settings):
    attribute, expected_env_var = env_var
    patch_settings(model_provider=provider, **{attribute: None})

    with pytest.raises(PreflightError, match=expected_env_var):
        preflight()


@pytest.mark.parametrize(("provider", "env_var"), sorted(_REQUIRED_CREDENTIAL.items()))
def test_preflight_rejects_a_whitespace_only_credential(provider, env_var, patch_settings):
    attribute, _ = env_var
    patch_settings(model_provider=provider, **{attribute: "   "})

    with pytest.raises(PreflightError):
        preflight()


@pytest.mark.parametrize(("provider", "env_var"), sorted(_REQUIRED_CREDENTIAL.items()))
def test_preflight_passes_with_a_credential_present(provider, env_var, patch_settings):
    attribute, _ = env_var
    patch_settings(model_provider=provider, **{attribute: "a-value"})

    preflight()  # must not raise


def test_preflight_for_ollama_needs_no_key(patch_settings):
    # Reachability is the only gate for a local server, so an absent key must not
    # block a start.
    patch_settings(model_provider="ollama", openrouter_api_key=None)
    preflight()


def test_preflight_for_ollama_rejects_a_blank_host(patch_settings):
    patch_settings(model_provider="ollama", ollama_host="  ")
    with pytest.raises(PreflightError, match="OLLAMA_HOST"):
        preflight()


def test_every_provider_either_needs_a_credential_or_is_ollama():
    # A provider missing from the table would raise KeyError inside preflight --
    # a crash on the start button rather than a message naming the fix.
    assert set(_REQUIRED_CREDENTIAL) | {"ollama"} == set(ALL_PROVIDERS)


def test_preflight_reads_settings_at_call_time():
    """The module must not hold a settings snapshot.

    Asserted structurally rather than by manipulating the environment: a test
    that relied on there being no `.env` would start failing the moment anyone
    created one, which is exactly what stage 10b does.
    """
    assert client_module.get_settings is get_settings
