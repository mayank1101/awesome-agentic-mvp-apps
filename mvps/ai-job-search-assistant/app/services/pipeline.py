"""The run, start to finish: resume in, ranked shortlist out.

Written as a **generator that yields events** rather than a function that takes a
progress callback. That is a deliberate structural choice, and this repo has paid
for the alternative: a callback invoked from anywhere other than the Streamlit
script's own thread raises `NoSessionContext`, and retrofitting a generator later
touches every branch. Yielding events also keeps `app/` free of any opinion about
how progress is displayed -- the UI consumes them on its own thread and paints
whatever it likes.

The run is a fixed sequence: parse the resume, build queries, search, rank,
fetch the top few, score them. No step decides what the next step is, which is
why there is no agent framework here -- there is nothing for a model to choose.

**Partial results beat no results, everywhere.** A job whose page will not load
keeps its snippet score. A job whose assessment call fails keeps its row and
carries the error. A run that hits the deadline mid-scoring returns what it has
with a notice. The only fatal failures are the two that leave nothing to show:
an unreadable resume, and a search that returned nothing at all.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.exceptions import (
    InputBlocked,
    ModelError,
    ModelQuotaExhausted,
    SearchError,
)
from app.core.logging import get_logger
from app.models.schemas import (
    CandidateProfile,
    RunResult,
    RunSummary,
    ScoredJob,
    SearchFilters,
)
from app.services import guardrails, queries, ranking, scoring, sites
from app.services.assess import assess_job
from app.services.fetch import fetch_postings
from app.services.profile import extract_profile
from app.services.search import search_jobs

logger = get_logger(__name__)


@dataclass(frozen=True)
class Progress:
    """A step starting or finishing.

    Attributes:
        message: What is happening, in words for the person waiting.
        fraction: Rough completion, 0-1. Rough on purpose: the honest number
            depends on how many jobs turn out to be worth scoring, which is not
            known until half way through.
    """

    message: str
    fraction: float = 0.0


@dataclass(frozen=True)
class ProfileReady:
    """The resume has been parsed, before any search has run.

    Emitted separately so the UI can show how the resume was read while the
    search is still going. A resume read wrong produces thirty plausible, wrong
    results, and this is the moment where that is cheap to notice.
    """

    profile: CandidateProfile


@dataclass(frozen=True)
class Finished:
    """The run is over and this is everything it produced."""

    result: RunResult


#: What a caller iterating the pipeline receives.
RunEvent = Progress | ProfileReady | Finished


@dataclass
class _Deadline:
    """Wall-clock budget for one run, checked between steps and never mid-call."""

    seconds: float
    started: float = field(default_factory=time.monotonic)

    @property
    def expired(self) -> bool:
        """Whether the budget is spent."""
        return time.monotonic() - self.started > self.seconds


def run_search(
    resume_text: str,
    filters: SearchFilters,
) -> Iterator[RunEvent]:
    """Run one job search, yielding progress as it goes.

    Args:
        resume_text: Normalised text from the uploaded resume.
        filters: What the user asked to narrow to, including the site whitelist.

    Yields:
        :class:`Progress` events throughout, one :class:`ProfileReady` once the
        resume is parsed, and exactly one :class:`Finished` at the end.

    Raises:
        InputBlocked: The resume was flagged and blocking is enabled.
        ModelError: The resume could not be parsed. Nothing downstream works
            without a profile.
        SearchError: The search provider refused every query, or the whitelist
            was empty.
    """
    settings = get_settings()
    deadline = _Deadline(settings.run_deadline_seconds)
    notices: list[str] = []

    _scan_resume(resume_text, notices)

    site_list, rejected = sites.normalize_sites(filters.sites)
    if rejected:
        notices.append(
            "These entries were not understood as domains and were not searched: "
            + ", ".join(rejected)
        )
    if not site_list:
        raise SearchError(
            "No usable job sites in the list. Add at least one, for example boards.greenhouse.io."
        )

    yield Progress("Reading the resume…", 0.05)
    profile = extract_profile(resume_text)
    yield ProfileReady(profile)

    query_list = queries.build_queries(profile, filters, limit=settings.max_queries)
    yield Progress(f"Searching {len(site_list)} sites with {len(query_list)} queries…", 0.2)

    hits, raw_count = search_jobs(query_list, site_list, recency_days=filters.recency_days)

    summary = RunSummary(
        queries=query_list,
        sites=site_list,
        results_found=raw_count,
        results_kept=len(hits),
        notices=notices,
    )

    if not hits:
        summary.notices.append(
            "No job postings came back. Try a broader role, a longer recency window, or more sites."
        )
        yield Finished(RunResult(profile=profile, jobs=[], summary=summary))
        return

    yield Progress(f"Ranking {len(hits)} postings against the resume…", 0.35)
    rankings, mode = ranking.rank_hits(profile, hits, filters)
    summary.matching_mode = mode

    if mode == "lexical":
        # Both branches say the same thing about the *results* and different
        # things about what to do. A configured key that got rate-limited looks
        # identical on screen to no key at all unless this distinction is made --
        # observed on a live run, where the second query's embedding call was
        # refused and the run silently produced word-overlap numbers.
        summary.notices.append(
            "The embedding service was unavailable for this run, so ranking and matching "
            "used word overlap rather than meaning. Scores are coarser than usual; running "
            "again in a minute usually restores it."
            if settings.semantic_available
            else "Ranking and matching used word overlap, not meaning. Set MISTRAL_API_KEY "
            "for semantic matching -- without it, 'shipped services in Go' does not match "
            "'Golang experience'."
        )

    deep_count = min(settings.deep_score_count, len(rankings))
    deep, shallow = rankings[:deep_count], rankings[deep_count:]

    yield Progress(f"Fetching {deep_count} postings…", 0.45)
    postings = fetch_postings([item.hit.url for item in deep])

    jobs: list[ScoredJob] = []
    quota_gone = False

    for index, item in enumerate(deep, start=1):
        if deadline.expired or quota_gone:
            jobs.append(_shallow_job(item, mode))
            continue

        posting = postings.get(item.hit.url)
        readable = bool(posting and posting.ok)
        yield Progress(
            f"Scoring {index} of {deep_count}: {item.hit.title or item.hit.domain}",
            0.45 + 0.5 * index / max(deep_count, 1),
        )

        if not readable:
            summary.postings_unreadable += 1
            jobs.append(
                _shallow_job(
                    item,
                    mode,
                    note=(posting.reason if posting else "The page could not be fetched."),
                )
            )
            continue

        try:
            job = _score_one(item, posting.text, resume_text)
        except ModelQuotaExhausted as exc:
            # Every remaining job would fail identically, so stop paying for the
            # attempt and finish the run with what has been scored.
            quota_gone = True
            summary.notices.append(f"{exc} The remaining jobs are ranked, not scored.")
            jobs.append(_shallow_job(item, mode))
            continue
        except ModelError as exc:
            logger.warning("Assessment failed for %s: %s", item.hit.url, exc)
            jobs.append(_shallow_job(item, mode, error=str(exc)))
            continue

        summary.deep_scored += 1
        jobs.append(job)

    if deadline.expired:
        summary.notices.append(
            f"The run hit its {settings.run_deadline_seconds:.0f}s limit. Jobs below the ones "
            "already scored are ranked on their search snippet."
        )

    jobs.extend(_shallow_job(item, mode) for item in shallow)
    jobs.sort(key=lambda job: (job.tier == "deep", job.score), reverse=True)

    yield Finished(RunResult(profile=profile, jobs=jobs, summary=summary))


def _scan_resume(resume_text: str, notices: list[str]) -> None:
    """Run input scanning, blocking only when configured to.

    Postings are scanned too, but later and never fatally -- see
    :mod:`app.services.guardrails` for why the two inputs are treated
    differently.
    """
    settings = get_settings()
    if not settings.guardrails_enabled:
        return

    findings = guardrails.scan_resume(resume_text)
    if not findings:
        return

    if settings.block_flagged_input and guardrails.has_severity(findings, "high"):
        raise InputBlocked(
            "The resume contains text that reads as an instruction to the model rather "
            "than as resume content. Remove it and upload again.",
            findings=findings,
        )

    notices.append(
        f"The resume contains {len(findings)} pattern(s) that read as instructions. They were "
        "treated as text, not followed."
    )


def _score_one(
    item: ranking.Ranking,
    posting_text: str,
    resume_text: str,
) -> ScoredJob:
    """Assess and score one job whose posting was readable."""
    findings = guardrails.scan_posting(posting_text, item.hit.title or item.hit.domain)

    assessment = assess_job(item.hit, posting_text, resume_text)
    score, breakdown, checked, match_mode = scoring.score_assessment(assessment, resume_text)

    reason = scoring.explain(checked, breakdown)
    if findings:
        reason += " (This posting contains text aimed at the grader; it was ignored.)"

    return ScoredJob(
        hit=item.hit,
        tier="deep",
        score=score,
        reason=reason,
        company=checked.company,
        title=checked.title or item.hit.title,
        location=checked.location,
        remote=checked.remote,
        assessment=checked,
        breakdown=breakdown,
        matching_mode=match_mode,
        posting_ok=True,
    )


def _shallow_job(
    item: ranking.Ranking,
    mode: str,
    *,
    note: str = "",
    error: str = "",
) -> ScoredJob:
    """Build a row from the cheap tier alone.

    Every path that cannot produce a deep score ends here rather than dropping
    the job: an unreadable page, an exhausted budget, an expired deadline, a
    failed call, or simply a job that ranked below the deep-scoring cut. The row
    says which, and the tier says the number came from a snippet.
    """
    reason = note or error or "Ranked on its title and search snippet; the posting was not read."
    return ScoredJob(
        hit=item.hit,
        tier="shallow",
        score=item.score,
        reason=reason,
        title=item.hit.title,
        matching_mode=mode,  # type: ignore[arg-type]
        posting_ok=not note,
        error=error,
    )


def collect(events: Iterator[RunEvent]) -> RunResult:
    """Drain a run and return its result. For tests and non-interactive callers.

    Raises:
        RuntimeError: The iterator ended without a :class:`Finished` event,
            which can only mean a `return` was added to the pipeline without a
            final yield.
    """
    result: RunResult | None = None
    for event in events:
        if isinstance(event, Finished):
            result = event.result
    if result is None:
        raise RuntimeError("The pipeline ended without producing a result.")
    return result
