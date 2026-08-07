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
#:
#: A relative path would be read from the *current working directory*, which is
#: never the same twice: `streamlit run` from the repo root, from `mvps/`, or
#: from the app folder all differ, and Streamlit Community Cloud runs from the
#: repo root. A relative path turns that into a silent "missing configuration"
#: screen.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Runtime configuration, validated on load.

    Attributes:
        groq_api_key: Required. Every language-model call in this app goes to
            Groq. Absent means the setup screen -- there is no degraded mode,
            because without a model there is no analysis and no rewrite.
        groq_base_url: Overridable for a proxy; the SDK default otherwise.
        model_name: Groq model id. Free-tier ids get retired without notice,
            which is why alternates are listed in `.env.example`.
        model_temperature: Low on purpose. Every call here is extraction or
            constrained rewriting, never authoring from nothing.
        max_tokens_extraction: Output cap for the two JSON extraction calls
            (resume structure, job requirements).
        max_tokens_assessment: Output cap for the per-requirement verdict call.
        max_tokens_tailor: Output cap for the rewrite. Larger than the others
            because it emits a whole resume. A reasoning model would need this
            raised substantially -- reasoning tokens come out of the same budget
            as visible text -- which is why the default is an instruct model.
        model_timeout_seconds: Per-call cap on a model request.
        mistral_api_key: Optional. Enables semantic requirement matching via
            `mistral-embed`. Without it the app falls back to lexical overlap
            and says so on screen; the score is coarser but the app still runs.
        embedding_model: Mistral embedding model id.
        embedding_timeout_seconds: Per-call cap on an embedding request.
        embedding_batch_size: Texts per embedding request. Mistral's free tier
            is rate-limited per request, so batching is the cheap win.
        max_resume_chars: Cap on extracted resume text. Longer resumes are
            truncated with a visible notice rather than rejected.
        max_jd_chars: Cap on the pasted job description, same treatment.
        min_resume_chars: Below this, the PDF is treated as scanned or empty and
            the run stops with an explanation. There is no OCR in this app.
        max_resume_pages: Page cap on the uploaded PDF.
        max_upload_bytes: Size cap applied before the PDF is parsed at all.
        max_requirements: Cap on requirements extracted from one job
            description, so a 4000-word posting cannot blow the assessment call.
        run_deadline_seconds: Global cap on one analysis.
        guardrails_enabled: Master switch for input scanning and output
            sanitising. The prompt fence is *not* covered by this and is always
            applied.
        block_flagged_input: Whether a high-severity finding stops the run
            (``True``) or only warns (``False``).
        strict_fabrication_guard: Whether a tailored resume that introduces
            facts absent from the original is rejected outright (``True``) or
            rendered with the offending lines flagged (``False``). True by
            default: an invented employer on a real resume is the one output
            this app must never produce.
        log_level: Root level for this application's loggers.
    """

    # -- model -------------------------------------------------------------- #
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com"
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.2
    max_tokens_extraction: int = 2600
    max_tokens_assessment: int = 2200
    max_tokens_tailor: int = 3200
    model_timeout_seconds: float = 90.0

    # -- embeddings --------------------------------------------------------- #
    mistral_api_key: str | None = None
    embedding_model: str = "mistral-embed"
    embedding_timeout_seconds: float = 30.0
    embedding_batch_size: int = 32

    # -- input budgets ------------------------------------------------------ #
    max_resume_chars: int = 20_000
    max_jd_chars: int = 12_000
    min_resume_chars: int = 200
    max_resume_pages: int = 8
    max_upload_bytes: int = 5 * 1024 * 1024
    max_requirements: int = 25
    run_deadline_seconds: float = 240.0

    # -- guardrails --------------------------------------------------------- #
    guardrails_enabled: bool = True
    block_flagged_input: bool = True
    strict_fabrication_guard: bool = True

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    def missing_credentials(self) -> list[str]:
        """Return the names of required credentials that are not set.

        Only the Groq key is required. Mistral is optional by design, so a
        missing embedding key is a banner, not a setup screen.

        Returns:
            Environment-variable names, uppercased. Empty when the app can start.
        """
        return [] if self.groq_api_key else ["GROQ_API_KEY"]

    @property
    def semantic_matching_available(self) -> bool:
        """Whether requirement matching can use embeddings rather than overlap."""
        return bool(self.mistral_api_key)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
