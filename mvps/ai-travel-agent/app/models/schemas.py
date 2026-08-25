"""Pydantic models for every boundary in the app.

:class:`GeneratedItinerary` is parsed straight from model output and is
therefore the app's real validation layer. Everything else describes a trip
request or search evidence the app assembled itself.

The load-bearing design choice, shared with this repo's other search-and-
synthesise apps: the model never sees a URL. Evidence items reach the prompt
labelled only by a small integer id (see `to_prompt_text`); the sources list
shown to the user is built by the app from the real search hits afterward.
A model that never sees a link cannot reproduce, mistype, or invent one.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EvidenceCategory = Literal["activities", "accommodation", "tips"]

BudgetLevel = Literal["not specified", "budget", "mid-range", "luxury"]


class _Strict(BaseModel):
    """Base for parsed model output: unknown keys are dropped, not fatal."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


class TripRequest(BaseModel):
    """What the user asked for, already validated.

    Attributes:
        destination: Free-text place name, as typed.
        days: Trip length, within `[min_days, max_days]`.
        interests: Optional free text ("street food, hiking, museums").
        budget_level: Optional budget framing, shown to the model so its
            suggestions and pacing match; never used to invent price figures.
    """

    destination: str
    days: int
    interests: str = ""
    budget_level: BudgetLevel = "not specified"

    @field_validator("destination")
    @classmethod
    def _non_empty_destination(cls, value: str) -> str:
        """Reject a blank destination, which would search for nothing useful."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("destination must not be empty")
        return stripped


# --------------------------------------------------------------------------- #
# Search evidence
# --------------------------------------------------------------------------- #


class SearchHit(BaseModel):
    """One raw search result, as the provider returned it."""

    title: str
    url: str
    domain: str
    content: str
    score: float = 0.0


class EvidenceItem(BaseModel):
    """One search result, packed for the prompt and for the sources list.

    Attributes:
        id: Small integer id, unique across the whole evidence set, used as
            the model's only handle on this item -- see the module docstring.
        category: Which section this evidence supports.
        title: Result title.
        content: Snippet, truncated to the configured cap.
        url: The real URL. Present on the object, but deliberately excluded
            from the text the model reads (`to_prompt_text`); kept here so the
            app can render a sources list afterward.
        domain: The result's host, shown beside the title in the sources list.
    """

    id: int
    category: EvidenceCategory
    title: str
    content: str
    url: str
    domain: str


class CategoryEvidence(BaseModel):
    """All evidence gathered for one category.

    Attributes:
        category: Which section this is.
        query: The search query that produced it, shown to the model so it
            knows what the absence of a result means.
        items: The deduplicated, capped results.
    """

    category: EvidenceCategory
    query: str
    items: list[EvidenceItem] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether the search for this category returned nothing usable."""
        return not self.items


# --------------------------------------------------------------------------- #
# Model output
# --------------------------------------------------------------------------- #


class DayPlan(_Strict):
    """One day of the itinerary.

    Attributes:
        day: 1-indexed day number.
        title: Short theme for the day ("Old Town & harbourfront").
        morning: What to do in the morning, in prose.
        afternoon: What to do in the afternoon, in prose.
        evening: What to do in the evening, in prose.
        note: One optional line of practical advice specific to this day
            (booking ahead, an opening-day closure, travel time between stops).
    """

    day: int = 0
    title: str = ""
    morning: str = ""
    afternoon: str = ""
    evening: str = ""
    note: str = ""


class GeneratedItinerary(_Strict):
    """The synthesis call's reply, before the app attaches real sources.

    Attributes:
        summary: A short framing of the trip as a whole, 2-3 sentences.
        accommodation_advice: Where to stay and why, grounded in the
            accommodation evidence -- areas and property types, not a
            fabricated list of specific hotel availability or prices.
        practical_tips: Getting around, timing, and other logistics grounded
            in the tips evidence.
        days: One entry per day of the trip, in order.
    """

    summary: str = ""
    accommodation_advice: str = ""
    practical_tips: str = ""
    days: list[DayPlan] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# The finished plan
# --------------------------------------------------------------------------- #


class TripPlan(BaseModel):
    """One finished trip plan, ready to render.

    Attributes:
        request: The originating request.
        summary: The model's framing of the trip.
        accommodation_advice: Where to stay, per the model.
        practical_tips: Logistics advice, per the model.
        itinerary: The day-by-day plan.
        sources: Every evidence item actually gathered, grouped by category in
            list order -- the app's own record of what grounded the plan,
            shown to the user regardless of which items the model leaned on.
        synthesis_degraded: Whether the model reply needed the JSON repair
            retry to parse. Shown on screen rather than hidden, the same way
            this repo's other synthesis apps report a fallback.
    """

    request: TripRequest
    summary: str = ""
    accommodation_advice: str = ""
    practical_tips: str = ""
    itinerary: list[DayPlan] = Field(default_factory=list)
    sources: list[EvidenceItem] = Field(default_factory=list)
    synthesis_degraded: bool = False


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #


class GuardrailFinding(BaseModel):
    """One heuristic match from the input scanner.

    Attributes:
        field: Where the match was found (`"destination"` or `"interests"`).
        pattern: The trigger phrase that matched.
        severity: `"high"` findings block the run when blocking is enabled.
    """

    field: str
    pattern: str
    severity: Literal["high"] = "high"
