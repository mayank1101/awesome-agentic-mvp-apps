"""Orchestration tests: no network.

`search.gather_evidence` and `llm.complete_model` are patched throughout.
"""

import pytest

from app.core.exceptions import DestinationBlocked, InvalidTripRequest, SearchError
from app.models.schemas import CategoryEvidence, DayPlan, EvidenceItem, GeneratedItinerary
from app.services import planner


def _evidence() -> list[CategoryEvidence]:
    return [
        CategoryEvidence(
            category="activities",
            query="q",
            items=[
                EvidenceItem(
                    id=1,
                    category="activities",
                    title="A",
                    content="c",
                    url="https://a.example.com",
                    domain="a.example.com",
                )
            ],
        ),
        CategoryEvidence(category="accommodation", query="q", items=[]),
        CategoryEvidence(category="tips", query="q", items=[]),
    ]


# --------------------------------------------------------------------------- #
# validate_request
# --------------------------------------------------------------------------- #


def test_blank_destination_is_rejected() -> None:
    with pytest.raises(InvalidTripRequest):
        planner.validate_request("   ", 3, "", "not specified")


def test_days_below_minimum_is_rejected() -> None:
    with pytest.raises(InvalidTripRequest):
        planner.validate_request("Lisbon", 0, "", "not specified")


def test_days_above_maximum_is_rejected() -> None:
    with pytest.raises(InvalidTripRequest):
        planner.validate_request("Lisbon", 999, "", "not specified")


def test_valid_request_is_built() -> None:
    request = planner.validate_request("  Lisbon  ", 5, "  food  ", "budget")
    assert request.destination == "Lisbon"
    assert request.days == 5
    assert request.interests == "food"
    assert request.budget_level == "budget"


# --------------------------------------------------------------------------- #
# plan_trip
# --------------------------------------------------------------------------- #


def test_plan_trip_assembles_sources_from_real_evidence(
    monkeypatch: pytest.MonkeyPatch, trip_request
) -> None:
    monkeypatch.setattr(planner, "gather_evidence", lambda request: _evidence())
    monkeypatch.setattr(
        planner.llm,
        "complete_model",
        lambda **_: (
            GeneratedItinerary(
                summary="A short trip.",
                accommodation_advice="Stay central.",
                practical_tips="Use the metro.",
                days=[
                    DayPlan(day=1, title="Old Town", morning="m", afternoon="a", evening="e"),
                    DayPlan(day=2, title="Belem", morning="m", afternoon="a", evening="e"),
                    DayPlan(day=3, title="Sintra", morning="m", afternoon="a", evening="e"),
                ],
            ),
            False,
        ),
    )

    result = planner.plan_trip(trip_request)

    assert result.summary == "A short trip."
    assert len(result.itinerary) == 3
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://a.example.com"
    assert result.synthesis_degraded is False


def test_wrong_day_count_triggers_a_repair_retry(
    monkeypatch: pytest.MonkeyPatch, trip_request
) -> None:
    monkeypatch.setattr(planner, "gather_evidence", lambda request: _evidence())

    replies = iter(
        [
            (GeneratedItinerary(days=[DayPlan(day=1)]), False),  # wrong count for a 3-day trip
            (
                GeneratedItinerary(days=[DayPlan(day=1), DayPlan(day=2), DayPlan(day=3)]),
                False,
            ),
        ]
    )
    monkeypatch.setattr(planner.llm, "complete_model", lambda **_: next(replies))

    result = planner.plan_trip(trip_request)

    assert len(result.itinerary) == 3
    assert result.synthesis_degraded is True


def test_destination_blocked_before_any_search(
    monkeypatch: pytest.MonkeyPatch, trip_request
) -> None:
    calls = []
    monkeypatch.setattr(
        planner, "gather_evidence", lambda request: calls.append(request) or _evidence()
    )
    monkeypatch.setattr(
        planner.guardrails,
        "guard",
        lambda destination, interests: (_ for _ in ()).throw(DestinationBlocked("blocked")),
    )

    with pytest.raises(DestinationBlocked):
        planner.plan_trip(trip_request)

    assert not calls


def test_every_search_failing_propagates(monkeypatch: pytest.MonkeyPatch, trip_request) -> None:
    def always_fails(request):
        raise SearchError("boom")

    monkeypatch.setattr(planner, "gather_evidence", always_fails)

    with pytest.raises(SearchError):
        planner.plan_trip(trip_request)
