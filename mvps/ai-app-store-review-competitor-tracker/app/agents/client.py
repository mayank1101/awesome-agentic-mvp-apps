"""Chat-client construction and per-call options.

One client is built for the whole process and cached; the agent is a thin
wrapper created per run. Providers are registered in :data:`_CLIENT_BUILDERS`,
one builder per ``MODEL_PROVIDER`` value, and each imports its SDK lazily so a
deployment that only uses Groq never pays for the others' imports.

Ported as-is from this repo's `ai-competitor-analyzer` (itself ported from
`ai-prd-generator`) -- this registry is shared across the repo's apps, and an
app that only works against one provider is one model retirement away from not
working at all.

The cached client is bound to the event loop that first used it. See
:mod:`app.services.async_bridge` for why every call routes through a single
long-lived loop.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from agent_framework import ChatOptions, SupportsChatGetResponse

from app.core.config import ModelProvider, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: How each provider spells "return JSON, any shape". Providers missing from
#: this map have no schema-less JSON mode; for those, the shape is carried by
#: the instructions alone, which is why the analyzer validates rather than
#: trusts.
_JSON_OBJECT_DIALECT: dict[str, Any] = {
    "openrouter": {"type": "json_object"},
    "groq": {"type": "json_object"},
    "openai": {"type": "json_object"},
    "foundry": {"type": "json_object"},
    "ollama": "json",
}


def _build_openrouter_client() -> SupportsChatGetResponse:
    """OpenRouter: OpenAI-compatible, so the stock client with a custom base URL."""
    from agent_framework.openai import OpenAIChatCompletionClient

    settings = get_settings()
    return OpenAIChatCompletionClient(
        model=settings.model_name,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _build_groq_client() -> SupportsChatGetResponse:
    """Groq: OpenAI-compatible, so the stock client with a custom base URL."""
    from agent_framework.openai import OpenAIChatCompletionClient

    settings = get_settings()
    return OpenAIChatCompletionClient(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
    )


def _build_openai_client() -> SupportsChatGetResponse:
    """OpenAI's own chat-completions endpoint."""
    from agent_framework.openai import OpenAIChatCompletionClient

    settings = get_settings()
    return OpenAIChatCompletionClient(model=settings.model_name, api_key=settings.openai_api_key)


def _build_anthropic_client() -> SupportsChatGetResponse:
    """Anthropic's Messages API."""
    from agent_framework.anthropic import AnthropicClient

    settings = get_settings()
    return AnthropicClient(model=settings.model_name, api_key=settings.anthropic_api_key)


def _build_ollama_client() -> SupportsChatGetResponse:
    """A local Ollama server. No credentials -- reachability is the only gate."""
    from agent_framework.ollama import OllamaChatClient

    settings = get_settings()
    return OllamaChatClient(model=settings.model_name, host=settings.ollama_host)


def _build_gemini_client() -> SupportsChatGetResponse:
    """Google AI Studio (Gemini)."""
    from agent_framework.gemini import GeminiChatClient

    settings = get_settings()
    return GeminiChatClient(model=settings.model_name, api_key=settings.gemini_api_key)


def _build_foundry_client() -> SupportsChatGetResponse:
    """Azure AI Foundry, which authenticates with a credential rather than a key."""
    from agent_framework.foundry import FoundryChatClient
    from azure.identity.aio import DefaultAzureCredential

    settings = get_settings()
    return FoundryChatClient(
        model=settings.model_name,
        project_endpoint=settings.azure_ai_project_endpoint,
        credential=DefaultAzureCredential(),
    )


#: One builder per ``MODEL_PROVIDER`` value.
_CLIENT_BUILDERS: dict[ModelProvider, Callable[[], SupportsChatGetResponse]] = {
    "openrouter": _build_openrouter_client,
    "groq": _build_groq_client,
    "openai": _build_openai_client,
    "anthropic": _build_anthropic_client,
    "ollama": _build_ollama_client,
    "gemini": _build_gemini_client,
    "foundry": _build_foundry_client,
}


@lru_cache
def get_chat_client() -> SupportsChatGetResponse:
    """Return the process-wide chat client for the configured provider.

    Returns:
        A client implementing the Agent Framework chat protocol, ready to be
        turned into an agent via ``.as_agent(...)``.

    Raises:
        ImportError: The selected provider's optional package is not
            installed, re-raised with the package to install.
    """
    settings = get_settings()
    logger.info(
        "Building chat client: provider=%s model=%s", settings.model_provider, settings.model_name
    )

    try:
        return _CLIENT_BUILDERS[settings.model_provider]()
    except ImportError as exc:
        raise ImportError(
            f"MODEL_PROVIDER={settings.model_provider!r} needs a package that is not "
            f"installed. Install the full framework with `pip install agent-framework`. "
            f"Original error: {exc}"
        ) from exc


def json_response_format() -> Any | None:
    """Ask the provider for JSON, in whichever dialect it speaks.

    Returns:
        The provider's ``response_format`` value, or ``None`` for providers
        with no schema-less JSON mode.
    """
    return _JSON_OBJECT_DIALECT.get(get_settings().model_provider)


def build_options(*, max_tokens: int | None = None, as_json: bool = True) -> ChatOptions:
    """Assemble the per-call options for the analysis run.

    Args:
        max_tokens: Output cap for this call. Defaults to the configured value.
        as_json: Whether to request JSON. The repair retry keeps this on;
            only a plain-text probe would turn it off.

    Returns:
        A ``ChatOptions`` mapping suitable for ``agent.run(..., options=...)``.
    """
    settings = get_settings()
    options = ChatOptions(
        temperature=settings.model_temperature,
        max_tokens=max_tokens or settings.max_tokens,
    )

    response_format = json_response_format() if as_json else None
    if response_format is not None:
        options["response_format"] = response_format
    return options
