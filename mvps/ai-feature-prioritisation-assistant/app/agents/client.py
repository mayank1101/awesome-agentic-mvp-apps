"""Chat-client construction and per-call options.

One client is built for the whole process and cached. The estimator agent is a
cheap wrapper around it, created per request (its instructions carry the
product context and the backlog's own ids) while the client -- and its
connection pool -- is shared.

Providers are registered in :data:`_CLIENT_BUILDERS`, one builder per
``MODEL_PROVIDER`` value. Each builder imports its SDK lazily, so a deployment
that only uses one provider never pays for the others' imports, and a missing
optional package fails with a clear message at selection time rather than at
application import.

Note that the cached client is bound to the event loop that first used it. See
:mod:`app.services.async_bridge` for why every call routes through a single
long-lived loop.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from agent_framework import ChatOptions, SupportsChatGetResponse
from pydantic import BaseModel

from app.core.config import ModelProvider, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: How each provider spells "return JSON, any shape". Providers missing from
#: this map have no schema-less JSON mode at all -- see
#: :data:`_SCHEMA_ONLY_PROVIDERS`.
_JSON_OBJECT_DIALECT: dict[str, Any] = {
    "openrouter": {"type": "json_object"},
    "groq": {"type": "json_object"},
    "openai": {"type": "json_object"},
    "foundry": {"type": "json_object"},
    "ollama": "json",
}

#: Providers whose only structured-output path is a schema. Both support it
#: natively -- Anthropic via ``output_config.format``, Gemini via
#: ``response_schema`` -- so ``json_object`` sends the schema for these rather
#: than degrading to prompt-only.
_SCHEMA_ONLY_PROVIDERS = frozenset({"anthropic", "gemini"})


def _build_openrouter_client() -> SupportsChatGetResponse:
    """OpenRouter: OpenAI-compatible, so the stock client with a custom base URL.

    Deliberately ``OpenAIChatCompletionClient`` and not ``OpenAIChatClient`` --
    the latter speaks the Responses API, which OpenRouter does not serve.
    """
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
    return OpenAIChatCompletionClient(
        model=settings.model_name,
        api_key=settings.openai_api_key,
    )


def _build_anthropic_client() -> SupportsChatGetResponse:
    """Anthropic's Messages API."""
    from agent_framework.anthropic import AnthropicClient

    settings = get_settings()
    return AnthropicClient(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
    )


def _build_ollama_client() -> SupportsChatGetResponse:
    """A local Ollama server. No credentials -- reachability is the only gate."""
    from agent_framework.ollama import OllamaChatClient

    settings = get_settings()
    return OllamaChatClient(
        model=settings.model_name,
        host=settings.ollama_host,
    )


def _build_gemini_client() -> SupportsChatGetResponse:
    """Google AI Studio (Gemini)."""
    from agent_framework.gemini import GeminiChatClient

    settings = get_settings()
    return GeminiChatClient(
        model=settings.model_name,
        api_key=settings.gemini_api_key,
    )


def _build_foundry_client() -> SupportsChatGetResponse:
    """Azure AI Foundry.

    Foundry authenticates with an Azure credential rather than an API key.
    ``DefaultAzureCredential`` covers the usual paths in order: environment
    service-principal variables (``AZURE_CLIENT_ID`` / ``AZURE_TENANT_ID`` /
    ``AZURE_CLIENT_SECRET``), managed identity, then a developer's ``az login``.
    Only the first of those works in a container, so set the service-principal
    variables when deploying.
    """
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
        turned into agents via ``.as_agent(...)``.

    Raises:
        ImportError: If the selected provider's optional package is not
            installed, re-raised with the package name to install.
    """
    settings = get_settings()
    logger.info(
        "Building chat client: provider=%s model=%s",
        settings.model_provider,
        settings.model_name,
    )

    try:
        return _CLIENT_BUILDERS[settings.model_provider]()
    except ImportError as exc:
        raise ImportError(
            f"MODEL_PROVIDER={settings.model_provider!r} needs a package that is not "
            f"installed. Install the full framework with `pip install agent-framework`. "
            f"Original error: {exc}"
        ) from exc


def build_options(
    *,
    response_format: Any | None = None,
    max_tokens: int | None = None,
) -> ChatOptions:
    """Assemble the per-call options every agent run is given.

    Args:
        response_format: Structured-output request, as produced by
            :func:`structured_response_format`. Omitted entirely when ``None``.
        max_tokens: Output cap for this call. Defaults to the configured value,
            which is sized for a full backlog rather than a single answer.

    Returns:
        A ``ChatOptions`` mapping suitable for ``agent.run(..., options=...)``.
    """
    settings = get_settings()
    options = ChatOptions(
        temperature=settings.model_temperature,
        max_tokens=max_tokens or settings.max_tokens,
    )
    if response_format is not None:
        options["response_format"] = response_format
    return options


def _schema_payload(provider: str, schema: type[BaseModel]) -> Any:
    """Express `schema` in the shape the provider's client actually forwards.

    Every client but Gemini's takes the Pydantic class directly. Gemini's client
    only forwards a *mapping* to ``GenerateContentConfig.response_schema``; hand
    it the class and you get ``response_mime_type='application/json'`` with no
    server-side schema at all, which is a materially weaker guarantee.
    """
    if provider == "gemini":
        return {"schema": schema.model_json_schema()}
    return schema


def structured_response_format(schema: type[BaseModel]) -> Any | None:
    """Translate ``STRUCTURED_OUTPUT_MODE`` into this provider's dialect.

    Every provider supports schema-based structured output, so ``json_schema``
    always sends the schema. Schema-*less* JSON mode is the part that varies --
    it is spelled differently where it exists, and two providers do not have it
    at all:

    ==========  =========================================================
    Provider    ``json_object`` becomes
    ==========  =========================================================
    openai-ish  ``{"type": "json_object"}`` (openrouter, openai, foundry)
    ollama      ``"json"``, which is how Ollama spells the same thing
    anthropic   the schema -- structured outputs are schema-only, via
                ``output_config.format``
    gemini      the schema as a mapping -- ``response_schema`` is
                schema-only too
    ==========  =========================================================

    For the last two, sending the schema is *stronger* than what was asked for,
    not weaker: the alternative would be prompt-only, and both vendors enforce
    schemas natively.

    Args:
        schema: The Pydantic model the reply should conform to.

    Returns:
        The value to pass as ``response_format``, or ``None`` under ``prompt``
        mode.
    """
    settings = get_settings()
    provider = settings.model_provider
    mode = settings.structured_output_mode

    if mode == "prompt":
        return None
    if mode == "json_schema" or provider in _SCHEMA_ONLY_PROVIDERS:
        return _schema_payload(provider, schema)
    return _JSON_OBJECT_DIALECT.get(provider)
