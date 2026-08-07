"""Application configuration.

Settings come from environment variables, falling back to a local `.env` file.
Access them through :func:`get_settings` rather than instantiating `Settings`
directly -- the result is cached, so the file is read once per process.

On Streamlit Community Cloud there is no `.env`: secrets arrive through
`st.secrets` and are copied into the environment by `streamlit_app.py` before
this module is first read. That indirection exists so this file stays a plain
`pydantic-settings` class with no Streamlit import, preserving the one-way
dependency from `ui/` to `app/`.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
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
        max_tokens: Default output cap. Both agents override it -- a question is
            short, a report is long.
        structured_output_mode: How hard the report step leans on the provider
            for JSON. ``json_schema`` asks it to enforce the ``FeedbackReport``
            schema, which every provider supports but not every *model* does;
            ``json_object`` asks only for valid JSON and relies on the shape
            spelled out in the instructions, which is the widest-compatibility
            default; ``prompt`` sends no ``response_format`` at all. Anthropic
            and Gemini have no schema-less JSON mode, so ``json_object`` sends
            the schema for them -- see
            :func:`app.agents.client.structured_response_format`.
        interviewer_max_tokens: Output cap for one interview question. Well below
            :attr:`max_tokens` on purpose: the document-wide cap would let a
            chatty model deliver a lecture, which breaks the "one question, no
            coaching" rules the interviewer runs under.
        report_max_tokens: Output cap for the graded report, the long output.
        max_answer_chars: Per-answer cap. Load-bearing rather than cosmetic: the
            whole conversation is resent on every turn, so an uncapped answer is
            multiplied across the remaining turns and can overflow the context
            window of exactly the free-tier models this app targets.
        min_answer_hint_chars: Below this the UI notes that short answers score
            low. Advisory only -- it never blocks a submission, because refusing
            input would be the tool deciding what counts as an answer.
        stream_timeout_seconds: Bounds every streamed call, so an abandoned stream
            cannot leave the interview screen waiting forever.
        guardrails_enabled: Master switch for input scanning and output
            sanitising. The prompt fence around candidate answers is *not*
            covered by this and is always applied -- it costs nothing and is the
            layer worth trusting.
        block_flagged_input: Whether a high-severity finding stops the turn
            (``True``) or only warns (``False``). Turn it off if false positives
            get in the way of a demo; the fence still stands either way.
        max_interviews_per_session: Cost guard -- interviews one browser session
            may start. Not a security control: a reload starts a new session and
            a restart resets the counter. Set to ``0`` to disable.
        ollama_host: Base URL of the Ollama server.
        gemini_api_key: Google AI Studio key, for ``gemini``.
        azure_ai_project_endpoint: Azure AI Foundry project endpoint, e.g.
            ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
        log_level: Root level for this application's loggers.
    """

    model_provider: ModelProvider = "openrouter"
    model_name: str = "deepseek/deepseek-chat-v3.1:free"
    model_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    structured_output_mode: StructuredOutputMode = "json_object"

    # --- interview shape ---
    interviewer_max_tokens: int = Field(default=512, gt=0)
    report_max_tokens: int = Field(default=4096, gt=0)
    max_answer_chars: int = Field(default=4000, gt=0)
    min_answer_hint_chars: int = Field(default=100, ge=0)

    # --- session lifecycle ---
    stream_timeout_seconds: int = Field(default=90, gt=0)

    # --- guardrails ---
    guardrails_enabled: bool = True
    block_flagged_input: bool = True

    # --- cost guard ---
    max_interviews_per_session: int = Field(default=5, ge=0)

    # --- credentials ---
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_host: str = "http://localhost:11434"
    gemini_api_key: str | None = None
    azure_ai_project_endpoint: str | None = None

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
