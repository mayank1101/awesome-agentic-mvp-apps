"""Application configuration.

Settings come from environment variables, falling back to a local `.env` file.
On Streamlit Community Cloud there is no `.env`: the entry point copies
`st.secrets` into `os.environ` with `setdefault` before the first call here, so
a real environment variable always wins over a secrets entry. That bridge
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

#: The `.env` beside the application package, resolved absolutely. A relative
#: path would be read from the current working directory, which differs
#: between `streamlit run` from the repo root, from `mvps/`, or from the app
#: folder, and Streamlit Community Cloud runs from the repo root.
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

#: Providers that authenticate with something other than a plain API key, and
#: so are exempt from the startup credential check.
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
        model_provider: Which Agent Framework chat client backs the gap
            analysis. Groq is the default, same as this repo's other apps.
        model_name: Provider-native model id, so swapping models is
            config-only. Free-tier ids are retired without notice.
        model_temperature: Low -- this is extraction/clustering over supplied
            review text, not open-ended authoring.
        max_tokens: Output cap for the single synthesis call.
        model_timeout_seconds: Per-call cap on the synthesis request.
        appstore_country: Storefront for both app lookup and the review feed.
            Fixed to "us" by default because that is the only storefront the
            public review feed currently serves reliably (see PRD §7) -- kept
            as a setting rather than a constant so a future fix to Apple's
            endpoint is a config change, not a code change.
        max_reviews: How many reviews one fetch can return. Apple's feed
            serves ~50 per request and pagination is currently broken (PRD
            §7), so this is a ceiling on what is used from the response, not
            a page count.
        min_critical_reviews: Floor on ≤3★ reviews in the sample before a gap
            analysis is attempted. Below it, the report shows stats only.
        request_timeout_seconds: Per-call timeout for both App Store HTTP
            calls.
        review_char_cap: Per-review trim applied before packing evidence for
            the model, on a word boundary.
        evidence_char_budget: Total characters of review text handed to the
            model in one call.
        guardrails_enabled: Master switch for input scanning and output
            sanitising. The prompt fence is not covered by this and is always
            applied.
        block_flagged_input: Whether a high-severity finding in the search
            query stops the run or only warns.
        log_level: Root level for this application's loggers.
    """

    # -- model --------------------------------------------------------------#
    model_provider: ModelProvider = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.2
    max_tokens: int = 2000
    model_timeout_seconds: float = 60.0

    # -- App Store (iOS) --------------------------------------------------- #
    #: Fixed at "us" in the UI -- see `app/appstore/reviews.py` for why no
    #: other storefront's reviews are reliably available there. Kept as a
    #: setting, not a constant, so a future fix to Apple's endpoint is a
    #: config change.
    appstore_country: str = "us"
    max_reviews: int = 50
    min_critical_reviews: int = 5
    request_timeout_seconds: float = 15.0

    # -- Play Store (Android) ------------------------------------------------#
    #: Default storefront offered in the UI. Unlike `appstore_country`, this
    #: one is a real, working choice -- US and India were both confirmed by
    #: direct testing; Play's own review system is properly localized per
    #: storefront, unlike Apple's feed.
    playstore_country: str = "us"

    # -- budgets --------------------------------------------------------------#
    review_char_cap: int = 900
    evidence_char_budget: int = 14_000

    # -- guardrails -----------------------------------------------------------#
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

        Returns:
            Environment-variable names, uppercased. Empty when the app can
            start. There is no partial mode: without a model key the app
            could still fetch reviews and show stats, but shipping that as
            the default would make "no gap analysis" look like a bug rather
            than a missing key.
        """
        missing: list[str] = []
        if self.model_provider not in _KEYLESS_PROVIDERS:
            field = _PROVIDER_KEY_FIELD[self.model_provider]
            if not getattr(self, field):
                missing.append(field.upper())
        return missing


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
