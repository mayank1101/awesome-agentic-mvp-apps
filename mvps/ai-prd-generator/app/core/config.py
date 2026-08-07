"""Application configuration.

Settings come from environment variables, falling back to a local `.env` file.
Access them through :func:`get_settings` rather than instantiating `Settings`
directly -- the result is cached, so the file is read once per process.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

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
        model_provider: Which Agent Framework chat client backs the agents. Each
            value maps to one client, and only the credentials for the selected
            one need to be set:

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
        model_temperature: Sampling temperature applied to every agent run.
        max_tokens: Default output cap. The outline step overrides it, as its
            reply is short and structured.
        structured_output_mode: How hard the outline step leans on the provider
            for JSON. ``json_schema`` asks it to enforce the ``PRDOutline``
            schema, which every provider supports but not every *model* does;
            ``json_object`` asks only for valid JSON and relies on the shape
            spelled out in the instructions, which is the widest-compatibility
            default; ``prompt`` sends no ``response_format`` at all. Anthropic
            and Gemini have no schema-less JSON mode, so ``json_object`` sends
            the schema for them -- see
            :func:`app.agents.client.outline_response_format`.
        guardrails_enabled: Master switch for input scanning and output
            sanitising. The prompt fence is *not* covered by this and is always
            applied -- it costs nothing and is the layer worth trusting.
        block_flagged_input: Whether a high-severity finding stops generation
            (``True``) or only warns (``False``). Turn it off if false positives
            get in the way of a demo; the fence still stands either way.
        max_generations_per_session: Cost guard for public deployments -- caps
            how many PRDs one browser session can start. Not a security control:
            a new session resets it. Set to ``0`` to disable.
        ollama_host: Base URL of the Ollama server.
        gemini_api_key: Google AI Studio key, for ``gemini``.
        azure_ai_project_endpoint: Azure AI Foundry project endpoint, e.g.
            ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
        log_level: Root level for this application's loggers.
    """

    model_provider: ModelProvider = "openrouter"
    model_name: str = "deepseek/deepseek-chat-v3.1:free"
    model_temperature: float = 0.4
    max_tokens: int = 4096

    structured_output_mode: StructuredOutputMode = "json_object"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"

    guardrails_enabled: bool = True
    block_flagged_input: bool = True
    max_generations_per_session: int = 20

    ollama_host: str = "http://localhost:11434"
    gemini_api_key: str | None = None
    azure_ai_project_endpoint: str | None = None

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
