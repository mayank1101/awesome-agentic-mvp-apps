"""The orchestrator: validate, resolve, search, pack, synthesise, render.

A **plain synchronous generator** yielding :class:`ProgressEvent`s, consumed by
the UI on Streamlit's own script thread. Not async, no worker thread: the six
searches are blocking HTTP calls made in sequence, and only the single model call
crosses into the shared event loop. The standing rule in this repo is never to
paint from a worker thread, and the cheapest way to obey it is to have no worker
thread at all.

The design rule throughout: **one failed section must not fail the report.** A
search that errors, times out, or returns nothing becomes an empty section, and
the run carries on. Only three things end a run early — an unidentifiable
company, exhausted credentials, and a dead model provider — and each of those has
its own screen.
"""

import time
from collections.abc import Iterator
from datetime import date

from app.agents.synthesizer import synthesize
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AnalyzerError,
    BudgetExhausted,
    SearchAuthError,
    SearchError,
    SearchQuotaError,
)
from app.core.logging import get_logger
from app.models.schemas import (
    SECTION_TITLES,
    AnalysisRequest,
    ProgressEvent,
    Report,
    RunBudget,
    SearchHit,
    SectionEvidence,
    SectionKey,
)
from app.search.client import SearchClient
from app.search.queries import build_query, search_params
from app.search.resolver import resolve
from app.services.evidence import fit_budget, pack_section
from app.services.guardrails import has_severity, scan_input
from app.services.normalize import registrable_host
from app.services.renderer import render_report

logger = get_logger(__name__)


class InputRejected(AnalyzerError):
    """The request was refused before any credit was spent (E-41)."""


def _deadline_passed(started: float, settings: Settings) -> bool:
    """Whether the global run deadline has expired (E-31)."""
    return (time.monotonic() - started) >= settings.run_deadline_seconds


def _search_section(
    section: SectionKey,
    request_name: str,
    domain: str | None,
    client: SearchClient,
    settings: Settings,
) -> list[SearchHit]:
    """Run one section's search, returning an empty list on any failure.

    A section that cannot be searched is a section that says "Not found in public
    sources" — never a failed report (E-17, E-25).
    """
    query = build_query(section, request_name, domain)

    try:
        return client.search(query, **search_params(section, settings))
    except (SearchAuthError, SearchQuotaError):
        # These are terminal for the whole run: every remaining search would fail
        # the same way, and spending the wall-clock to prove it helps nobody.
        raise
    except (SearchError, BudgetExhausted) as exc:
        logger.warning("section %s search failed: %s", section.value, exc)
        return []


def _pricing_retry_needed(hits: list[SearchHit], domain: str | None) -> bool:
    """Whether the pricing search should be retried against the company's own site.

    Pricing is the section where the primary source matters most (SC-3), and
    third-party "X pricing explained" posts are exactly what an open search
    returns for it (E-22).
    """
    if not domain:
        return False
    own = registrable_host(f"https://{domain}")
    return not any(hit.host == own for hit in hits)


def run(
    request: AnalysisRequest,
    *,
    settings: Settings | None = None,
    transport: object | None = None,
) -> Iterator[ProgressEvent]:
    """Produce one report, yielding progress as it goes.

    Args:
        request: The validated, already-normalised request.
        settings: Runtime configuration; defaults to the process settings.
        transport: Optional search transport. The UI passes a caching one so
            Streamlit's constant reruns cannot re-spend credits; tests pass a
            fake. The default builds the real SDK client.

    Yields:
        A :class:`ProgressEvent` per step. The final event carries `stage="done"`,
        and the finished :class:`Report` is available from
        :func:`run_to_completion`.

    Raises:
        InputRejected: A high-severity injection finding, with blocking enabled.
        CompanyNotFound: Identity could not be resolved (SC-5).
        SearchAuthError: Credentials were rejected.
        SearchQuotaError: The search credit pool is exhausted.
        SynthesisError: The model provider failed terminally.
    """
    settings = settings or get_settings()
    started = time.monotonic()

    # -- validate ---------------------------------------------------------- #
    yield ProgressEvent(stage="validating", message="Checking the request…")

    if settings.guardrails_enabled:
        findings = scan_input(request.name, request.domain)
        if findings and settings.block_flagged_input and has_severity(findings, "high"):
            raise InputRejected(findings[0].message)

    client = SearchClient(
        settings=settings,
        budget=RunBudget(limit=settings.max_search_credits_per_report),
        transport=transport,
    )

    # -- resolve ----------------------------------------------------------- #
    yield ProgressEvent(stage="resolving", message=f"Identifying “{request.name}”…")
    identity = resolve(request.name, request.domain, client)

    yield ProgressEvent(
        stage="resolving",
        message=f"Profiling {identity.name}" + (f" ({identity.domain})" if identity.domain else ""),
    )

    # -- search ------------------------------------------------------------ #
    packed: list[SectionEvidence] = []

    for section in SectionKey:
        title = SECTION_TITLES[section]
        query = build_query(section, identity.name, identity.domain)

        if _deadline_passed(started, settings):
            # Out of time: the remaining sections are empty and synthesis still
            # runs. A late report beats no report; a guaranteed report beats both.
            logger.warning("run deadline reached; skipping %s", section.value)
            packed.append(SectionEvidence(section=section, query=query))
            continue

        yield ProgressEvent(stage="searching", message=f"Searching: {title}…", section=section)
        hits = _search_section(section, identity.name, identity.domain, client, settings)

        if (
            section is SectionKey.PRICING
            and _pricing_retry_needed(hits, identity.domain)
            and client.budget.can_afford()
        ):
            yield ProgressEvent(
                stage="searching",
                message="Checking their own pricing page…",
                section=section,
            )
            hits += _search_section_own_domain(section, identity, client, settings)

        packed_section = pack_section(
            section,
            query,
            hits,
            name=identity.name,
            domain=identity.domain,
            settings=settings,
        )

        # The news index carries publication dates but has thin coverage of
        # smaller companies; the general index has the coverage and no dates.
        # Try news first, and fall back once when it produced nothing usable --
        # gated on the budget, so the credit cap still decides (SC-8, E-22).
        if (
            section is SectionKey.RECENT_MOVES
            and packed_section.is_empty
            and client.budget.can_afford()
        ):
            yield ProgressEvent(
                stage="searching",
                message="No company news found — widening the search…",
                section=section,
            )
            fallback_query = build_query(section, identity.name, identity.domain)
            fallback_hits = _search_recent_moves_fallback(identity, client, settings)
            packed_section = pack_section(
                section,
                fallback_query,
                fallback_hits,
                name=identity.name,
                domain=identity.domain,
                settings=settings,
            )

        packed.append(packed_section)

    # -- pack -------------------------------------------------------------- #
    yield ProgressEvent(stage="packing", message="Selecting the strongest sources…")
    evidence = fit_budget(packed, settings=settings)

    # -- synthesise and render --------------------------------------------- #
    yield ProgressEvent(stage="synthesising", message="Writing the brief…")
    result, synthesis_failed = synthesize(identity, evidence)

    generated_on = date.today()
    report = Report(
        identity=identity,
        markdown=render_report(
            identity,
            result,
            evidence,
            generated_on=generated_on,
            synthesis_failed=synthesis_failed,
        ),
        generated_on=generated_on,
        evidence=evidence,
        credits_spent=client.budget.spent,
        synthesis_failed=synthesis_failed,
    )

    yield ProgressEvent(
        stage="done",
        message=f"Done — {report.source_count} sources, {report.credits_spent} credits.",
        ok=not synthesis_failed,
    )
    # The report rides out on the generator's return value, so a caller that only
    # wants progress never has to handle it.
    return report


def _search_recent_moves_fallback(
    identity,  # noqa: ANN001 - CompanyIdentity; importing it for typing would cycle
    client: SearchClient,
    settings: Settings,
) -> list[SearchHit]:
    """Search the general index for recent activity when the news index had none.

    Items found this way usually carry no `published_date`, so whether they
    survive is decided at render time by whether the model can date them from the
    snippet itself. That is the intended order: the recency rule stays in code,
    and this only widens what the rule gets to look at.
    """
    query = (
        f"{identity.name} {identity.domain or ''} news funding round product launch announcement"
    )

    try:
        return client.search(
            " ".join(query.split()),
            topic="general",
            time_range="year",
            max_results=settings.max_results_recent_moves,
        )
    except (SearchError, BudgetExhausted) as exc:
        logger.info("recent-moves fallback failed: %s", exc)
        return []


def _search_section_own_domain(
    section: SectionKey,
    identity,  # noqa: ANN001 - CompanyIdentity, imported for typing would cycle
    client: SearchClient,
    settings: Settings,
) -> list[SearchHit]:
    """Retry one section restricted to the company's own domain (E-22)."""
    query = build_query(section, identity.name, identity.domain)
    params = search_params(section, settings) | {"include_domains": [identity.domain]}

    try:
        return client.search(query, **params)
    except (SearchError, BudgetExhausted) as exc:
        logger.info("own-domain retry for %s failed: %s", section.value, exc)
        return []


def run_to_completion(
    request: AnalysisRequest,
    *,
    settings: Settings | None = None,
    transport: object | None = None,
) -> tuple[list[ProgressEvent], Report]:
    """Drive a run to the end, collecting its events.

    Used by tests and scripts. The UI consumes :func:`run` directly, so it can
    paint each event as it arrives.

    Args:
        request: The validated request.
        settings: Runtime configuration.
        transport: Optional search transport, as for :func:`run`.

    Returns:
        Every event yielded, and the finished report.
    """
    events: list[ProgressEvent] = []
    generator = run(request, settings=settings, transport=transport)

    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return events, stop.value
