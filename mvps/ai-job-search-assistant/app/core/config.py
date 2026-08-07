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

#: Job sites searched unless the user edits the list.
#:
#: Two kinds of domain, chosen for different reasons. The first four are
#: applicant-tracking systems, where the page a search returns is the employer's
#: own posting: one job per URL, full requirements in the text, and no
#: re-listing of the same role by six aggregators. The rest are the boards a
#: seeker already checks, included because leaving them out means missing
#: postings that exist nowhere else -- at the cost of pages that are heavier and
#: more often gated.
#:
#: This is a *whitelist enforced at the search API*, not a filter applied to
#: results: a page on a domain that is not here is never fetched and never paid
#: for.
DEFAULT_JOB_SITES: tuple[str, ...] = (
    "boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "apply.workable.com",
    "wellfound.com",
    "linkedin.com",
    "naukri.com",
)


class Settings(BaseSettings):
    """Runtime configuration, validated on load.

    Attributes:
        groq_api_key: Required. Every language-model call goes to Groq. Absent
            means the setup screen -- without a model there is no scoring, and a
            list of unscored links is what job boards already give you.
        groq_base_url: Overridable for a proxy; the SDK default otherwise.
        model_name: Groq model id. Free-tier ids get retired without notice,
            which is why alternates are listed in `.env.example`.
        model_temperature: Low on purpose. Every call here is extraction or
            judgement against supplied text, never authoring.
        max_tokens_profile: Output cap for the one resume-parsing call.
        max_tokens_assessment: Output cap for a single job's assessment. This
            call emits every requirement it found plus a verdict and an evidence
            quote for each, so it is the largest reply the app asks for.
        model_timeout_seconds: Per-call cap on a model request.
        tavily_api_key: Required. The only path this app has to the web.
        tavily_base_url: Overridable for a proxy; Tavily's own host otherwise.
        tavily_search_depth: ``basic`` or ``advanced``. Advanced costs more
            credits per query and returns better snippets; the default is basic
            because the snippets here only have to rank, not to score -- the
            jobs that get scored have their full text fetched.
        tavily_timeout_seconds: Per-call cap on a search or extract request.
        results_per_query: Results asked for per search query.
        max_queries: Cap on how many distinct queries one run issues. Every
            query is a credit, and past a handful they return the same postings.
        max_results_total: Cap on results carried forward after de-duplication.
        extract_batch_size: URLs per extract request. Tavily accepts up to 20;
            batching is what keeps a run to one or two extract credits.
        deep_score_count: How many top-ranked jobs get the full read: posting
            text fetched, requirements extracted, coverage judged line by line.
            The single most expensive knob in the app.
        mistral_api_key: Optional. Enables semantic ranking and semantic
            evidence matching via `mistral-embed`. Without it both fall back to
            lexical overlap and the UI says so; the app still runs.
        embedding_model: Mistral embedding model id.
        embedding_timeout_seconds: Per-call cap on an embedding request.
        embedding_batch_size: Texts per embedding request. Mistral's free tier
            is rate-limited per request, so batching is the cheap win.
        max_resume_chars: Cap on extracted resume text. Longer resumes are
            truncated with a visible notice rather than rejected.
        min_resume_chars: Below this, the PDF is treated as scanned or empty and
            the run stops with an explanation. There is no OCR in this app.
        max_resume_pages: Page cap on the uploaded PDF.
        max_upload_bytes: Size cap applied before the PDF is parsed at all.
        max_posting_chars: Cap on one fetched posting. Career pages carry
            boilerplate -- benefits, EEO statements, the company's founding
            story -- and the requirements are almost always in the first part.
        min_posting_chars: Below this, an extracted page is treated as a
            JavaScript shell or a login wall, and the job falls back to being
            scored on its search snippet with that stated on screen.
        max_requirements: Cap on requirements taken from one posting. A posting
            with 40 bullets is usually 40 restatements of 12 things.
        run_deadline_seconds: Global cap on one run, checked between steps.
        guardrails_enabled: Master switch for input scanning and output
            sanitising. The prompt fence is *not* covered by this and is always
            applied.
        block_flagged_input: Whether a high-severity finding in the *resume*
            stops the run (``True``) or only warns (``False``). Fetched postings
            are never blocked on -- see :mod:`app.services.guardrails`.
        log_level: Root level for this application's loggers.
    """

    # -- model -------------------------------------------------------------- #
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com"
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.2
    max_tokens_profile: int = 1800
    max_tokens_assessment: int = 2600
    model_timeout_seconds: float = 90.0

    # -- search ------------------------------------------------------------- #
    tavily_api_key: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    tavily_search_depth: str = "basic"
    tavily_timeout_seconds: float = 45.0
    results_per_query: int = 10
    max_queries: int = 4
    max_results_total: int = 40
    extract_batch_size: int = 10
    deep_score_count: int = 8

    # -- embeddings --------------------------------------------------------- #
    mistral_api_key: str | None = None
    embedding_model: str = "mistral-embed"
    embedding_timeout_seconds: float = 30.0
    embedding_batch_size: int = 32

    # -- input budgets ------------------------------------------------------ #
    max_resume_chars: int = 20_000
    min_resume_chars: int = 200
    max_resume_pages: int = 8
    max_upload_bytes: int = 5 * 1024 * 1024
    max_posting_chars: int = 12_000
    min_posting_chars: int = 400
    max_requirements: int = 20
    run_deadline_seconds: float = 300.0

    # -- guardrails --------------------------------------------------------- #
    guardrails_enabled: bool = True
    block_flagged_input: bool = True

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    def missing_credentials(self) -> list[str]:
        """Return the names of required credentials that are not set.

        Two are required and one is not. Without Groq there is no scoring;
        without Tavily there is no search; and either missing leaves an app that
        cannot do the one thing it exists for. Mistral is optional by design, so
        a missing embedding key is a banner, not a setup screen.

        Returns:
            The environment variable names still to be set, in the order a
            reader should set them.
        """
        missing = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        return missing

    @property
    def semantic_available(self) -> bool:
        """Whether embedding-backed ranking and matching can be attempted.

        A configured key is not a promise that the calls will succeed; it only
        decides whether they are tried. Every caller degrades to lexical on
        failure and reports which mode produced the numbers.
        """
        return bool(self.mistral_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, reading configuration once.

    Cached because Streamlit re-executes the entry script on every interaction,
    and re-reading a `.env` file per keystroke is a real cost. Tests clear the
    cache between cases (see `tests/conftest.py`).
    """
    return Settings()
