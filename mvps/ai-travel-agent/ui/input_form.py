"""The input screen: destination, trip length, and optional preferences.

Validation happens before any search or model call is made: a blank
destination or an out-of-range trip length fails here, in milliseconds,
rather than after a spinner and a spent search credit.
"""

from dataclasses import dataclass

import streamlit as st

from app.core.exceptions import InvalidTripRequest
from app.models.schemas import BudgetLevel, TripRequest
from app.services.planner import validate_request
from ui import state as S

_BUDGET_OPTIONS: tuple[BudgetLevel, ...] = ("not specified", "budget", "mid-range", "luxury")


@dataclass(frozen=True)
class Submission:
    """What the form produced on this rerun."""

    submitted: bool = False
    request: TripRequest | None = None
    error: str = ""

    @property
    def ready(self) -> bool:
        """Whether this submission should start a plan."""
        return self.submitted and not self.error and self.request is not None


def render() -> Submission:
    """Draw the form and return what it produced."""
    settings = S.settings()

    st.write(
        "Enter a destination and how many days you have. The assistant searches the "
        "web for current activities, places to stay, and practical tips, then builds a "
        "day-by-day itinerary from what it finds."
    )

    destination = st.text_input(
        "Destination",
        placeholder="e.g. Lisbon, Portugal",
        max_chars=settings.max_destination_chars,
    )
    days = st.number_input(
        "Trip length (days)",
        min_value=settings.min_days,
        max_value=settings.max_days,
        value=min(5, settings.max_days),
        step=1,
    )
    interests = st.text_area(
        "Interests (optional)",
        placeholder="e.g. street food, museums, hiking, nightlife",
        max_chars=settings.max_interests_chars,
        height=80,
    )
    budget_level = st.selectbox("Budget level (optional)", options=_BUDGET_OPTIONS, index=0)

    submitted = st.button("Plan my trip", type="primary", use_container_width=True)

    if not submitted:
        return Submission()

    try:
        request = validate_request(destination, int(days), interests, budget_level)
    except InvalidTripRequest as exc:
        return Submission(submitted=True, error=str(exc))

    return Submission(submitted=True, request=request)
