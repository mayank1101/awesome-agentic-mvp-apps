"""Domain models for the prioritisation pipeline.

The flow through these types is:

    raw paste  --(backlog parser)-->  BacklogInput  (FeatureIdea per item)
               --(estimator agent)->  BacklogEstimate  (FactorSet + rationales)
               --(scorer, in code)->  RankedBacklog  (ScoredFeature per item)

Two properties of this contract are load-bearing.

**No score type is estimable.** :class:`FeatureEstimate` -- the only model the
agent layer ever produces -- has no ``rice`` or ``ice`` field to put a number in.
A model cannot supply a score here even if it tries, because there is nowhere for
one to go. :class:`ScoredFeature` carries the scores and is only ever built by
:mod:`app.services.scoring`.

**Factors are snapped on the way in.** The validators call the same functions the
scorer does, so a value is normalised once, at the boundary, and the number the
user sees is the number that was multiplied.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

from app.services.scales import (
    clamp_reach,
    ease_from_effort,
    ice_confidence,
    ice_impact,
    snap_confidence,
    snap_effort,
    snap_impact,
)

#: Which of the four factors a user has overridden, for attribution in the UI
#: and the export. A number the user chose and a number the model chose are
#: different kinds of claim and must not render identically.
FactorName = Literal["reach", "impact", "confidence", "effort_months"]

FeatureId = Annotated[str, StringConstraints(min_length=1, max_length=8)]
Rationale = Annotated[str, StringConstraints(max_length=400)]


class FeatureIdea(BaseModel):
    """One feature as the user wrote it, plus the id used to track it.

    The id is assigned by the parser rather than by the model, so a reply that
    invents, drops, or reorders entries can be reconciled against what was
    actually sent.
    """

    id: FeatureId = Field(..., description="Stable handle, e.g. 'F3'")
    title: str = Field(..., min_length=1, max_length=120)
    notes: str = Field(default="", max_length=400, description="The user's rough notes, verbatim")


class BacklogInput(BaseModel):
    """A parsed backlog, ready to estimate."""

    features: list[FeatureIdea] = Field(..., min_length=1)
    product_context: str = Field(
        default="",
        max_length=600,
        description=(
            "One or two lines about the product: business model, user or account "
            "counts, team size, what this quarter is about. This is what anchors "
            "Reach in absolute units and Effort in this team's person-months; "
            "without it both become assumptions and are labelled as such."
        ),
    )


class FactorSet(BaseModel):
    """The four RICE factors for one feature, snapped onto their scales.

    ICE's inputs are derived from these rather than estimated separately -- see
    :mod:`app.services.scales` for why -- so this is the single description of a
    feature that both frameworks read.
    """

    reach: float = Field(..., description="Users or accounts affected per quarter")
    impact: float = Field(..., description="One of 3, 2, 1, 0.5, 0.25")
    confidence: float = Field(..., description="One of 1.0, 0.8, 0.5")
    effort_months: float = Field(..., description="Person-months, total across the team")

    @field_validator("reach", mode="after")
    @classmethod
    def _clamp_reach(cls, value: float) -> float:
        return clamp_reach(value)

    @field_validator("impact", mode="after")
    @classmethod
    def _snap_impact(cls, value: float) -> float:
        return snap_impact(value)

    @field_validator("confidence", mode="after")
    @classmethod
    def _snap_confidence(cls, value: float) -> float:
        return snap_confidence(value)

    @field_validator("effort_months", mode="after")
    @classmethod
    def _snap_effort(cls, value: float) -> float:
        return snap_effort(value)

    @property
    def ease(self) -> int:
        """ICE Ease (1-10), derived from :attr:`effort_months`."""
        return ease_from_effort(self.effort_months)

    @property
    def ice_impact(self) -> int:
        """ICE Impact (1-10), derived from :attr:`impact`."""
        return ice_impact(self.impact)

    @property
    def ice_confidence(self) -> int:
        """ICE Confidence (1-10), derived from :attr:`confidence`."""
        return ice_confidence(self.confidence)


class FeatureEstimate(BaseModel):
    """What the estimator returns for one feature: factors and their reasoning.

    The field descriptions are part of the prompt contract -- they appear in the
    JSON schema sent to the provider under ``json_schema`` output mode, and are
    restated in the instructions for the modes that send no schema.

    Note what is absent: there is no score field. The estimator's job ends at the
    factors.
    """

    id: FeatureId = Field(..., description="The feature id exactly as it was given to you")
    reach: float = Field(
        ...,
        description=(
            "Users or accounts affected per quarter, as an absolute count anchored "
            "to the product context. Not a 1-10 rating."
        ),
    )
    reach_rationale: Rationale = Field(
        ..., description="One line: where this count came from, referencing the user's own words"
    )
    impact: float = Field(
        ...,
        description="Exactly one of 3 (massive), 2 (high), 1 (medium), 0.5 (low), 0.25 (minimal)",
    )
    impact_rationale: Rationale = Field(..., description="One line justifying that rung")
    confidence: float = Field(
        ...,
        description=(
            "Exactly one of 1.0, 0.8, 0.5 — how much evidence the user's own note "
            "carries, not how sure you feel"
        ),
    )
    confidence_rationale: Rationale = Field(
        ..., description="One line: what evidence is or is not there"
    )
    effort_months: float = Field(
        ...,
        description="Person-months across the whole team, at least 0.25 (about one week)",
    )
    effort_rationale: Rationale = Field(..., description="One line justifying the size")
    assumptions: list[Rationale] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Everything you had to assume because the notes did not say. Leave "
            "empty only when the notes genuinely covered everything."
        ),
    )

    @field_validator("reach", mode="after")
    @classmethod
    def _clamp_reach(cls, value: float) -> float:
        return clamp_reach(value)

    @field_validator("impact", mode="after")
    @classmethod
    def _snap_impact(cls, value: float) -> float:
        return snap_impact(value)

    @field_validator("confidence", mode="after")
    @classmethod
    def _snap_confidence(cls, value: float) -> float:
        return snap_confidence(value)

    @field_validator("effort_months", mode="after")
    @classmethod
    def _snap_effort(cls, value: float) -> float:
        return snap_effort(value)

    def factors(self) -> FactorSet:
        """The four factors, split out from the reasoning that accompanies them."""
        return FactorSet(
            reach=self.reach,
            impact=self.impact,
            confidence=self.confidence,
            effort_months=self.effort_months,
        )

    def rationales(self) -> dict[str, str]:
        """Per-factor reasoning, keyed by factor name."""
        return {
            "reach": self.reach_rationale,
            "impact": self.impact_rationale,
            "confidence": self.confidence_rationale,
            "effort_months": self.effort_rationale,
        }


class BacklogEstimate(BaseModel):
    """The estimator's reply for a whole backlog, in one pass.

    One call covers the whole list on purpose: a feature scored blind to its
    neighbours is scored on an arbitrary scale, and the ranking that results is
    noise dressed as arithmetic.

    :attr:`reach_unit` is declared once, for the whole backlog, because that is
    the level the error occurs at. A live run against a real model produced a
    list where one feature's Reach counted blocked *deals* (3) and another's
    counted *seats* (12,000) -- both defensible in isolation, and together they
    inverted the ranking. Forcing one declared unit is the fix, and showing it
    in the table header is what lets a reader catch a violation.
    """

    reach_unit: str = Field(
        default="users",
        max_length=30,
        description="What every Reach number counts, e.g. 'accounts', 'users', 'seats'",
    )
    estimates: list[FeatureEstimate] = Field(
        ..., description="One entry per feature id you were given, in any order"
    )


class ScoredFeature(BaseModel):
    """One ranked row: the idea, its factors, both scores, and both ranks.

    Built only by :mod:`app.services.scoring`. Every numeric field here is the
    output of a formula applied to :attr:`factors`.
    """

    idea: FeatureIdea
    factors: FactorSet
    rationales: dict[str, str] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    overridden: list[FactorName] = Field(
        default_factory=list, description="Factors the user changed by hand"
    )

    rice: float
    ice: int
    rice_rank: int
    ice_rank: int

    @property
    def rank_shift(self) -> int:
        """How many places ICE moves this feature up relative to RICE.

        Positive means ICE ranks it higher. A large positive shift is almost
        always a narrow feature that is cheap to build: ICE cannot see the small
        Reach that holds it back under RICE.
        """
        return self.rice_rank - self.ice_rank

    @property
    def is_low_confidence(self) -> bool:
        """Whether this row's ranking rests on evidence the notes did not supply."""
        return self.factors.confidence <= 0.5 or bool(self.assumptions)


class RankedBacklog(BaseModel):
    """The finished ranking, plus everything the UI needs to caveat it.

    Attributes:
        rows: Every scored feature, ordered by RICE rank.
        unestimated: Ids the estimator did not return an entry for. Surfaced
            rather than filled in -- a feature with invented factors is worse
            than a feature visibly missing from the table.
        divergence: Plain-language notes on where RICE and ICE disagree, and
            which factor is responsible.
        reach_unit: What the Reach column counts, carried through from the
            estimate so the table header can say it rather than leaving the
            reader to guess.
    """

    rows: list[ScoredFeature] = Field(default_factory=list)
    unestimated: list[FeatureId] = Field(default_factory=list)
    divergence: list[str] = Field(default_factory=list)
    reach_unit: str = "users"

    def by_ice(self) -> list[ScoredFeature]:
        """The same rows, ordered by ICE rank instead."""
        return sorted(self.rows, key=lambda row: row.ice_rank)
