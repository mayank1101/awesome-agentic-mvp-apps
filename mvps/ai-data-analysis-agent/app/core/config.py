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

#: The `.env` beside the application package, resolved absolutely -- a relative
#: path would be read from the current working directory, which differs between
#: `streamlit run` from the repo root, from `mvps/`, or from this folder.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Runtime configuration, validated on load.

    Attributes:
        groq_api_key: Required. Every language-model call in this app goes to
            Groq. Absent means the setup screen -- there is no degraded mode.
        groq_base_url: Overridable for a proxy; the SDK default otherwise.
        model_name: Groq model id. Free-tier ids get retired without notice.
        model_temperature: Low on purpose -- code generation is not creative
            writing, and a lower temperature produces more consistent pandas.
        max_tokens_code: Output cap for the code-generation call.
        max_tokens_answer: Output cap for the answer-synthesis call.
        model_timeout_seconds: Per-call cap on a model request.
        max_upload_bytes: Size cap applied before the CSV is parsed at all.
        max_rows: Row cap after loading. A larger file is truncated with a
            notice rather than rejected -- most questions ("what's the average
            X") are answerable from a large-but-capped sample, and there is no
            OCR-style hard blocker for tabular data the way there is for a scan.
        max_columns: Column cap, same treatment.
        preview_rows: Rows of the dataset shown to the model verbatim, so it can
            see real values rather than only dtypes.
        max_cell_chars: Cap on any single cell's text when building the schema
            preview shown to the model, so one huge free-text field cannot blow
            the prompt budget.
        max_output_rows: Cap on rows of a computed result rendered back to the
            user and fed to the answer-synthesis call.
        max_question_chars: Cap on the pasted question.
        max_code_chars: Cap on generated code length, checked before parsing --
            a cheap backstop against a pathological reply before it ever reaches
            the AST walk.
        code_timeout_seconds: Wall-clock cap on executing generated code.
        run_deadline_seconds: Global cap on one question, checked between steps.
        guardrails_enabled: Master switch for the heuristic prompt-injection
            scan over the question and the dataset's own text cells.
        block_flagged_input: Whether a high-severity finding stops the run
            (``True``) or only logs a warning (``False``).
        log_level: Root level for this application's loggers.
    """

    # -- model -------------------------------------------------------------- #
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com"
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.1
    max_tokens_code: int = 1200
    max_tokens_answer: int = 700
    model_timeout_seconds: float = 60.0

    # -- data intake ---------------------------------------------------------#
    max_upload_bytes: int = 10 * 1024 * 1024
    max_rows: int = 200_000
    max_columns: int = 200
    preview_rows: int = 8
    max_cell_chars: int = 200
    max_output_rows: int = 50

    # -- question / code budgets --------------------------------------------#
    max_question_chars: int = 500
    max_code_chars: int = 4000
    code_timeout_seconds: float = 10.0
    run_deadline_seconds: float = 90.0

    # -- guardrails --------------------------------------------------------- #
    guardrails_enabled: bool = True
    block_flagged_input: bool = True

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    def missing_credentials(self) -> list[str]:
        """Return the names of required credentials that are not set."""
        return [] if self.groq_api_key else ["GROQ_API_KEY"]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
