"""Application configuration.

Settings come from environment variables, falling back to a local `.env` file.
On Streamlit Community Cloud there is no `.env`: the entry point copies
`st.secrets` into `os.environ` with `setdefault` before the first call here, so a
real environment variable always wins over a secrets entry. That bridge lives in
`streamlit_app.py` and not in this module on purpose -- `app/` never imports
Streamlit.

Access settings through :func:`get_settings` rather than instantiating
:class:`Settings` directly; the result is cached, so the file is read once per
process.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: The `.env` beside the application package, resolved absolutely.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Runtime configuration, validated on load.

    Attributes:
        groq_api_key: Required. Every language-model call goes to Groq. Absent
            means the setup screen -- without a model there is no itinerary.
        groq_base_url: Overridable for a proxy; the SDK default otherwise.
        model_name: Groq model id. Free-tier ids get retired without notice.
        model_temperature: A little higher than a pure-extraction app's,
            because sequencing a day's stops and writing the connective prose
            between them is a light creative task, not extraction. Still low
            enough that the model does not wander from the evidence.
        max_tokens_itinerary: Output cap for the single synthesis call. Sized
            for the maximum trip length this app allows, not the typical one.
        model_timeout_seconds: Per-call cap on a model request.
        tavily_api_key: Required. The only path this app has to the web.
        tavily_base_url: Overridable for a proxy; Tavily's own host otherwise.
        tavily_search_depth: ``basic`` or ``advanced``. Basic is enough here --
            these snippets only have to describe a place well enough to plan
            around, not stand as the sole evidence for a factual claim.
        tavily_timeout_seconds: Per-call cap on a search request.
        results_per_query: Results asked for per search query.
        max_queries: Cap on how many searches one trip request issues. Scales
            with trip length (see `app.services.search`), capped here so a
            typo'd 300-day trip cannot turn into a runaway credit spend.
        max_evidence_per_category: Cap on how many deduplicated results from
            one category (activities, accommodation, tips) reach the prompt.
        max_snippet_chars: Cap on one search result's content, when building
            the evidence block the model reads.
        min_days: Smallest trip length this app plans.
        max_days: Longest trip length this app plans. Not a Tavily or Groq
            limit -- a deliberate product scope, since a 60-day itinerary from
            three searches per category is thin regardless of the token budget.
        max_destination_chars: Cap on the destination field.
        max_interests_chars: Cap on the free-text interests field.
        run_deadline_seconds: Global cap on one run, checked between steps.
        guardrails_enabled: Master switch for the heuristic prompt-injection
            scan over the destination and interests fields.
        block_flagged_input: Whether a high-severity finding stops the run
            (``True``) or only logs a warning (``False``).
        log_level: Root level for this application's loggers.
    """

    # -- model -------------------------------------------------------------- #
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com"
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.3
    max_tokens_itinerary: int = 4000
    model_timeout_seconds: float = 90.0

    # -- search --------------------------------------------------------------#
    tavily_api_key: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    tavily_search_depth: str = "basic"
    tavily_timeout_seconds: float = 45.0
    results_per_query: int = 8
    max_queries: int = 6
    max_evidence_per_category: int = 8
    max_snippet_chars: int = 600

    # -- trip budgets --------------------------------------------------------#
    min_days: int = 1
    max_days: int = 14
    max_destination_chars: int = 100
    max_interests_chars: int = 300
    run_deadline_seconds: float = 180.0

    # -- guardrails --------------------------------------------------------- #
    guardrails_enabled: bool = True
    block_flagged_input: bool = True

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    def missing_credentials(self) -> list[str]:
        """Return the names of required credentials that are not set.

        Both are required: without Groq there is no itinerary; without Tavily
        there is nothing grounding it. Either missing leaves an app that
        cannot do the one thing it exists for.
        """
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
