"""Orchestrates one trip: validate, search, pack evidence, synthesise.

The pipeline is linear and each step's failure is distinct:

1. Validate the request against the trip-length bounds
   (:mod:`app.core.config`).
2. Guard the destination and interests fields against obvious prompt-injection
   phrasing (:mod:`app.services.guardrails`).
3. Search the web for activities, accommodation areas, and practical tips
   (:mod:`app.services.search`). One failed query does not fail the run.
4. Ask the model to write the itinerary from that evidence
   (:mod:`app.prompts`, :mod:`app.services.llm`). The model never sees a URL.
5. Assemble the finished plan, attaching the real sources the app gathered --
   not whatever the model chose to mention.
"""

import time
from collections.abc import Callable

from app.core.config import get_settings
from app.core.exceptions import InvalidTripRequest, RunDeadlineExceeded
from app.core.logging import get_logger
from app.models.schemas import GeneratedItinerary, TripPlan, TripRequest
from app.prompts import REPAIR_INSTRUCTION, build_user_message, system_instructions
from app.services import guardrails, llm
from app.services.search import gather_evidence

logger = get_logger(__name__)

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    return None


def validate_request(destination: str, days: int, interests: str, budget_level: str) -> TripRequest:
    """Build a :class:`TripRequest`, rejecting anything outside this app's scope.

    Raises:
        InvalidTripRequest: The destination is blank or too long, or the day
            count is outside `[min_days, max_days]`.
    """
    settings = get_settings()

    stripped = (destination or "").strip()
    if not stripped:
        raise InvalidTripRequest("Enter a destination.")
    if len(stripped) > settings.max_destination_chars:
        raise InvalidTripRequest(
            f"That destination is longer than {settings.max_destination_chars} characters."
        )

    if not (settings.min_days <= days <= settings.max_days):
        raise InvalidTripRequest(
            f"Trip length must be between {settings.min_days} and {settings.max_days} days."
        )

    trimmed_interests = (interests or "").strip()[: settings.max_interests_chars]

    return TripRequest(
        destination=stripped,
        days=days,
        interests=trimmed_interests,
        budget_level=budget_level,  # type: ignore[arg-type]
    )


def _check_deadline(started_at: float, deadline: float) -> None:
    if time.monotonic() - started_at > deadline:
        raise RunDeadlineExceeded(f"This trip plan took longer than the {deadline:.0f}s limit.")


def plan_trip(request: TripRequest, *, progress: ProgressFn | None = None) -> TripPlan:
    """Produce a full trip plan for an already-validated request.

    Args:
        request: The validated trip request.
        progress: Optional callback for UI status updates.

    Returns:
        The finished :class:`TripPlan`.

    Raises:
        DestinationBlocked: Guardrail scanning found a high-severity match.
        SearchError: Every search query failed.
        ModelError: The provider failed in a way retries could not absorb.
        RunDeadlineExceeded: The run exceeded its time budget.
    """
    report = progress or _noop
    settings = get_settings()
    started_at = time.monotonic()

    report("Checking the request…")
    guardrails.guard(request.destination, request.interests)
    _check_deadline(started_at, settings.run_deadline_seconds)

    report(f"Searching the web for {request.destination}…")
    sections = gather_evidence(request)
    _check_deadline(started_at, settings.run_deadline_seconds)

    report("Writing the itinerary…")
    generated, degraded = llm.complete_model(
        system=system_instructions(),
        user=build_user_message(request, sections),
        schema=GeneratedItinerary,
        max_tokens=settings.max_tokens_itinerary,
    )

    if len(generated.days) != request.days:
        # A day count that does not match the request is repaired once, the
        # same courtesy a parse failure gets -- the itinerary is otherwise
        # unusable regardless of how well-formed its JSON was.
        logger.warning(
            "Model returned %d day(s) for a %d-day trip; retrying once",
            len(generated.days),
            request.days,
        )
        repaired, _ = llm.complete_model(
            system=system_instructions(),
            user=f"{build_user_message(request, sections)}\n\n{REPAIR_INSTRUCTION}",
            schema=GeneratedItinerary,
            max_tokens=settings.max_tokens_itinerary,
            temperature=0.0,
        )
        if len(repaired.days) == request.days:
            generated, degraded = repaired, True
        else:
            # Still wrong: ship what exists rather than fail the run outright.
            # A day short beats no itinerary; the count mismatch is visible on
            # screen either way.
            degraded = True

    sources = [item for section in sections for item in section.items]

    return TripPlan(
        request=request,
        summary=generated.summary,
        accommodation_advice=generated.accommodation_advice,
        practical_tips=generated.practical_tips,
        itinerary=sorted(generated.days, key=lambda d: d.day),
        sources=sources,
        synthesis_degraded=degraded,
    )
