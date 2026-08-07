"""Application configuration.

Settings come from environment variables, falling back to a local `.env` file.
On Streamlit Community Cloud there is no `.env`: the entry point copies
`st.secrets` into `os.environ` with `setdefault` before the first call here, so a
real environment variable always wins over a secrets entry (E-60). That bridge
lives in `streamlit_app.py` and not in this module on purpose -- `app/` never
imports Streamlit.

Access settings through :func:`get_settings` rather than instantiating
:class:`Settings` directly; the result is cached, so the file is read once per
process.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

#: The `.env` beside the application package, resolved absolutely.
#:
#: A relative path would be read from the *current working directory*, which is
#: the one thing about this app that is never the same twice: `streamlit run`
#: from the repo root, from `mvps/`, or from the app folder all differ, and
#: Streamlit Community Cloud runs from the repo root. A relative path turns that
#: into a silent "missing configuration" screen (E-63).
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

ModelProvider = Literal[
    "openrouter",
    "groq",
    "openai",
    "anthropic",
    "ollama",
    "gemini",
    "foundry",
]

#: Providers that authenticate with something other than a plain API key, and so
#: are exempt from the startup credential check.
_KEYLESS_PROVIDERS = frozenset({"ollama", "foundry"})

#: Which settings field holds each provider's key.
_PROVIDER_KEY_FIELD: dict[str, str] = {
    "openrouter": "openrouter_api_key",
    "groq": "groq_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
    "gemini": "gemini_api_key",
}


class Settings(BaseSettings):
    """Runtime configuration, validated on load.

    Attributes:
        model_provider: Which Agent Framework chat client backs synthesis. Groq
            is the default for this app; the other six stay switchable because
            the client registry is shared across the repo's apps.
        model_name: Provider-native model id, so swapping models is config-only.
            Free-tier ids are retired without notice, which is why alternates are
            listed in `.env.example` and a rejected id is handled at call time
            (E-36).
        model_temperature: Lower than the repo's other apps. This is extraction
            from retrieved text, not authoring.
        max_tokens: Output cap for the single synthesis call. Kept low
            deliberately: Groq counts the *reservation* toward its per-minute
            token limit, so a generous cap is spent whether or not the model
            writes that much. A 4096 cap plus this app's evidence exceeded the
            8000 TPM ceiling on a free-tier key and failed every request. Six
            sections of prose need roughly 1500 tokens. A reasoning model would
            need this raised substantially, since reasoning tokens come out of
            the same budget as visible text -- which is why the default model is
            an instruct one (E-37).
        model_timeout_seconds: Per-call cap on the synthesis request (E-32).
        tavily_api_key: Search credentials. Absent means the setup-error screen;
            there is no useful degraded mode, because search without synthesis is
            a link list and synthesis without search is the ungrounded chatbot
            this app exists to replace.
        search_depth: Tavily depth. ``basic`` costs 1 credit per call, ``advanced``
            costs 2. Basic is the default the budget is sized around.
        search_timeout_seconds: Per-call cap on a search request. A timed-out
            search becomes an empty section, not a failed run (E-25).
        max_search_credits_per_report: The app's cap on itself (SC-8). Six section
            searches, plus one optional resolution search and one optional pricing
            retry, is eight in the worst case.
        max_results_per_section: How many hits to request per section search.
        max_results_recent_moves: Requested for the news section, where items are
            then filtered by date, so more are asked for than will survive.
        run_deadline_seconds: Global cap on one report. Deliberately above the 90s
            median target: this is the guarantee, not the goal (E-31).
        evidence_char_budget: Total characters of retrieved text handed to the
            model. Sized against the provider's free-tier *rate* limits rather
            than its context window: the window is 131k tokens and the binding
            constraints are 8k tokens per minute and 100k per day. Roughly 3k
            tokens of evidence plus the max_tokens reservation fits inside both.
        snippet_char_cap: Per-snippet trim, applied on a word boundary.
        max_company_name_chars: Input cap. Longer names are truncated with a
            notice rather than rejected (E-04).
        search_cache_ttl_seconds: How long a search result stays reusable. A
            credit optimisation only -- correctness never depends on a cache hit.
        max_reports_per_session: Cost speed bump for a public deployment. Not a
            security control: a reload resets it, and the UI says so (E-52).
        guardrails_enabled: Master switch for input scanning and output
            sanitising. The prompt fence is *not* covered by this and is always
            applied -- it costs nothing and is the layer worth trusting.
        block_flagged_input: Whether a high-severity finding stops the run
            (``True``) or only warns (``False``).
        log_level: Root level for this application's loggers.
    """

    # -- model ------------------------------------------------------------- #
    model_provider: ModelProvider = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.2
    max_tokens: int = 2500
    model_timeout_seconds: float = 60.0

    # -- search ------------------------------------------------------------ #
    tavily_api_key: str | None = None
    search_depth: Literal["basic", "advanced"] = "basic"
    search_timeout_seconds: float = 20.0
    max_search_credits_per_report: int = 8
    max_results_per_section: int = 5
    max_results_recent_moves: int = 8

    # -- budgets ----------------------------------------------------------- #
    run_deadline_seconds: float = 120.0
    evidence_char_budget: int = 12_000
    snippet_char_cap: int = 1_200
    max_company_name_chars: int = 120

    # -- session ----------------------------------------------------------- #
    search_cache_ttl_seconds: int = 3_600
    max_reports_per_session: int = 5

    # -- guardrails -------------------------------------------------------- #
    guardrails_enabled: bool = True
    block_flagged_input: bool = True

    # -- provider credentials ---------------------------------------------- #
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

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    def missing_credentials(self) -> list[str]:
        """Return the names of required credentials that are not set.

        Both keys are required and there is no partial mode, so the setup screen
        names exactly which one is missing rather than saying "check your
        configuration" (E-61).

        Returns:
            Environment-variable names, uppercased, in the order a reader should
            fix them. Empty when the app can start.
        """
        missing: list[str] = []

        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")

        if self.model_provider not in _KEYLESS_PROVIDERS:
            field = _PROVIDER_KEY_FIELD[self.model_provider]
            if not getattr(self, field):
                missing.append(field.upper())

        return missing


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
