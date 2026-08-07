r"""The orchestrator: resolve, fetch, compute stats, (maybe) analyze, render.

A **plain synchronous generator** yielding :class:`ProgressEvent`\s, consumed
by the UI on Streamlit's own script thread -- same pattern as
`ai-competitor-analyzer`, and for the same reason: never paint from a worker
thread, and the cheapest way to guarantee that is to have no worker thread at
all. Only the single model call crosses into the shared event loop.

App resolution is a separate, earlier step (see `app/appstore/search.py` and
`ui/input_form.py`): a name search can return multiple candidates that need a
human to pick one, which does not fit a single linear run. This module starts
from an already-chosen :class:`AppIdentity`.
"""

import time
from collections.abc import Callable, Iterator
from datetime import date

from app.agents.analyzer import analyze
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.schemas import AppIdentity, FeatureGap, Platform, ProgressEvent, Report, Review
from app.services.evidence import select_critical
from app.services.renderer import render_report
from app.services.stats import compute_stats

logger = get_logger(__name__)

#: Takes the resolved identity (which carries its own platform, id, and
#: country) and settings, returns the fetched reviews. Identity-shaped rather
#: than id-shaped so one signature covers both platforms without the caller
#: needing to know which fields matter for which store.
FetchReviews = Callable[[AppIdentity, Settings], list[Review]]


def default_fetch_reviews(identity: AppIdentity, settings: Settings) -> list[Review]:
    """Dispatch to the right store's review client based on `identity.platform`."""
    if identity.platform is Platform.IOS:
        from app.appstore.reviews import fetch_reviews

        return fetch_reviews(identity.track_id, settings=settings)

    from app.playstore.reviews import fetch_reviews

    return fetch_reviews(identity.package_name, country=identity.country, settings=settings)


def run(
    identity: AppIdentity,
    *,
    settings: Settings | None = None,
    fetch_reviews: FetchReviews | None = None,
) -> Iterator[ProgressEvent]:
    """Produce one report, yielding progress as it goes.

    Args:
        identity: The app to analyze, already resolved and confirmed.
        settings: Runtime configuration; defaults to the process settings.
        fetch_reviews: Injected review-fetch callable, `(identity, settings)
            -> list[Review]`. Tests pass a fake; the UI passes the cached
            wrapper; the default is :func:`default_fetch_reviews`, which
            dispatches on `identity.platform`.

    Yields:
        A :class:`ProgressEvent` per step. The final event carries
        `stage="done"`, and the finished :class:`Report` is the generator's
        return value.

    Raises:
        ReviewFetchError: The review feed request failed outright -- unlike
            the multi-section competitor-analyzer pipeline, there is exactly
            one fetch here, so its failure is the run's failure.
        AnalysisError: The model provider failed terminally on the gap
            analysis call.
    """
    fetch = fetch_reviews or default_fetch_reviews
    settings = settings or get_settings()
    started = time.monotonic()

    yield ProgressEvent(
        stage="resolving", message=f"Using {identity.track_name} ({identity.artist_name})"
    )

    yield ProgressEvent(stage="fetching", message="Fetching recent reviews…")
    reviews = fetch(identity, settings)
    logger.info("fetched %d reviews in %.1fs", len(reviews), time.monotonic() - started)

    stats = compute_stats(reviews)
    critical = select_critical(reviews)

    insufficient_signal = len(critical) < settings.min_critical_reviews
    analysis_failed = False
    gaps: tuple[FeatureGap, ...] = ()

    if insufficient_signal:
        yield ProgressEvent(
            stage="analyzing",
            message=(
                f"Only {len(critical)} critical review(s) — need "
                f"{settings.min_critical_reviews} to run a gap analysis."
            ),
        )
    else:
        yield ProgressEvent(
            stage="analyzing", message=f"Analyzing {len(critical)} critical reviews…"
        )
        result, analysis_failed = analyze(identity.track_name, critical)
        gaps = result.gaps

    yield ProgressEvent(stage="rendering", message="Building the report…")
    generated_on = date.today()
    markdown, resolved_gaps = render_report(
        identity,
        stats,
        tuple(reviews),
        gaps,
        generated_on=generated_on,
        insufficient_signal=insufficient_signal,
        analysis_failed=analysis_failed,
    )

    report = Report(
        identity=identity,
        stats=stats,
        reviews=tuple(reviews),
        gaps=resolved_gaps,
        markdown=markdown,
        generated_on=generated_on,
        insufficient_signal=insufficient_signal,
        analysis_failed=analysis_failed,
    )

    yield ProgressEvent(
        stage="done",
        message=f"Done — {len(reviews)} reviews, {len(resolved_gaps)} gap(s) found.",
        ok=not analysis_failed,
    )
    return report


def run_to_completion(
    identity: AppIdentity,
    *,
    settings: Settings | None = None,
    fetch_reviews: FetchReviews | None = None,
) -> tuple[list[ProgressEvent], Report]:
    """Drive a run to the end, collecting its events. Used by tests and scripts."""
    events: list[ProgressEvent] = []
    generator = run(identity, settings=settings, fetch_reviews=fetch_reviews)

    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return events, stop.value
