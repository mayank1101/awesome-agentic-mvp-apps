"""Application configuration.

Settings come from environment variables, falling back to a local `.env` file.
Access them through :func:`get_settings` rather than instantiating `Settings`
directly -- the result is cached, so the file is read once per process.

On Streamlit Community Cloud there is no `.env`; secrets arrive via
`st.secrets` and are bridged into the environment by `streamlit_app.py` before
the first call here, which is why this module has no Streamlit import.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

#: The project root, found from this file rather than from the working
#: directory. `streamlit run mvps/<app>/streamlit_app.py` from a parent
#: directory is a normal way to launch this, and a cwd-relative `.env` silently
#: resolves to nothing there -- the app starts, reads no key, and fails on the
#: first model call with an authentication error that points nowhere near the
#: real cause.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

ModelProvider = Literal[
    "openrouter",
    "groq",
    "openai",
    "anthropic",
    "ollama",
    "gemini",
    "foundry",
]
StructuredOutputMode = Literal["json_schema", "json_object", "prompt"]


class Settings(BaseSettings):
    """Runtime configuration, validated on load.

    Attributes:
        model_provider: Which Agent Framework chat client backs the estimator.
            Each value maps to one client, and only the credentials for the
            selected one need to be set:

            ============  ===============================================
            Value         Client and what it needs
            ============  ===============================================
            openrouter    ``OpenAIChatCompletionClient`` against
                          :attr:`openrouter_base_url`
            groq          ``OpenAIChatCompletionClient`` against
                          :attr:`groq_base_url`
            openai        ``OpenAIChatCompletionClient``
            anthropic     ``AnthropicClient``
            ollama        ``OllamaChatClient`` against :attr:`ollama_host`;
                          no key, since it is a local server
            gemini        ``GeminiChatClient`` (Google AI Studio)
            foundry       ``FoundryChatClient`` (Azure AI Foundry), which
                          authenticates with an Azure credential rather
                          than an API key
            ============  ===============================================

        model_name: Provider-native model id, so swapping models is config-only.
            For Foundry this is the deployment name.
        model_temperature: Sampling temperature for the estimation call. Low by
            default: the task is classification onto fixed scales, and a
            creative reading of "Impact: 2" has no upside.
        max_tokens: Output cap for the estimation reply. Sized for a full
            backlog -- every feature costs roughly 120 tokens of JSON, so a
            25-item list needs several thousand and a 4k cap would truncate the
            tail of the list rather than fail loudly.
        structured_output_mode: How hard the estimator leans on the provider for
            JSON. ``json_schema`` asks it to enforce the ``BacklogEstimate``
            schema, which every provider supports but not every *model* does;
            ``json_object`` asks only for valid JSON and relies on the shape
            spelled out in the instructions, which is the widest-compatibility
            default; ``prompt`` sends no ``response_format`` at all.
        max_features: Hard cap on backlog size. The estimation call sees the
            whole list at once so features are calibrated against each other,
            and that design is what makes the list length a token-budget
            question rather than a UI preference.
        max_feature_chars: Per-feature cap on title plus notes.
        max_context_chars: Cap on the product-context field.
        guardrails_enabled: Master switch for input scanning and output
            sanitising. The prompt fence is *not* covered by this and is always
            applied -- it costs nothing and is the layer worth trusting.
        block_flagged_input: Whether a high-severity finding stops estimation
            (``True``) or only warns (``False``). Turn it off if false positives
            get in the way of a demo; the fence still stands either way.
        max_estimations_per_session: Cost guard for public deployments -- caps
            how many estimation calls one browser session can make. Not a
            security control: a new session resets it. Set to ``0`` to disable.
            Editing factors never counts against it, because editing never calls
            a model.
        ollama_host: Base URL of the Ollama server.
        gemini_api_key: Google AI Studio key, for ``gemini``.
        azure_ai_project_endpoint: Azure AI Foundry project endpoint, e.g.
            ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
        log_level: Root level for this application's loggers.
    """

    model_provider: ModelProvider = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.2
    max_tokens: int = 8192

    structured_output_mode: StructuredOutputMode = "json_object"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"

    max_features: int = 25
    max_feature_chars: int = 400
    max_context_chars: int = 600

    guardrails_enabled: bool = True
    block_flagged_input: bool = True
    max_estimations_per_session: int = 20

    ollama_host: str = "http://localhost:11434"
    gemini_api_key: str | None = None
    azure_ai_project_endpoint: str | None = None

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=_PROJECT_ROOT / ".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
