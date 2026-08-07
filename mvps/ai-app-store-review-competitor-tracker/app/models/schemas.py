"""Domain models.

Every boundary in this app is a Pydantic model: what App Store search and the
review feed returned, what the model replied, and what the renderer turns into
Markdown. Pure data -- no I/O, no Streamlit, no provider SDKs.

The one load-bearing model is :class:`GapAnalysisResult`. Its gaps cite reviews
by **id**, never by quoting them -- see `app/services/guardrails.py` and
`app/services/renderer.py` for why that is what makes "zero invented review
excerpts" a structural property rather than a prompt instruction.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# App identity
# --------------------------------------------------------------------------- #


class Platform(StrEnum):
    """Which store an app was resolved from.

    Everything downstream of resolution (stats, evidence packing, the
    analyzer, guardrails, the renderer) is platform-agnostic and works from
    :class:`Review` alone -- this enum only decides which `app/<platform>/`
    client resolution and fetching dispatch to.
    """

    IOS = "ios"
    ANDROID = "android"


class AppCandidate(BaseModel):
    """One match from a name search, shown for disambiguation.

    `track_id` (iOS) and `package_name` (Android) are mutually exclusive,
    populated according to `platform` -- see :attr:`external_id`.
    """

    model_config = ConfigDict(frozen=True)

    platform: Platform
    track_id: int | None = None
    package_name: str | None = None
    track_name: str
    artist_name: str
    primary_genre_name: str | None = None
    artwork_url: str | None = None
    app_store_url: str
    average_user_rating: float | None = None
    user_rating_count: int = 0

    @property
    def external_id(self) -> str:
        """The id to pass to this platform's review-fetch client."""
        return str(self.track_id) if self.platform is Platform.IOS else (self.package_name or "")


class AppIdentity(BaseModel):
    """The one app this report is about."""

    model_config = ConfigDict(frozen=True)

    platform: Platform
    track_id: int | None = None
    package_name: str | None = None
    track_name: str
    artist_name: str
    primary_genre_name: str | None = None
    artwork_url: str | None = None
    app_store_url: str
    #: The store's own published rating -- distinct from the stats computed
    #: over this run's fetched sample, and rendered separately so the two are
    #: never confused (the sample is at most ~50 recent reviews, not the
    #: app's full history).
    published_average_rating: float | None = None
    published_rating_count: int = 0
    #: Storefront the identity (and, for Android, the reviews) were resolved
    #: against, e.g. "us", "in". Always "us" for iOS -- see
    #: `app/appstore/reviews.py` for why no other storefront's reviews are
    #: reliably available there.
    country: str = "us"

    @property
    def external_id(self) -> str:
        """The id to pass to this platform's review-fetch client."""
        return str(self.track_id) if self.platform is Platform.IOS else (self.package_name or "")


# --------------------------------------------------------------------------- #
# Reviews
# --------------------------------------------------------------------------- #


class Review(BaseModel):
    """One fetched review, exactly as the feed returned it (trimmed for length).

    This is the app's only source of quotable text. The model never writes a
    review excerpt; the renderer looks one up here by `id` (SC-1).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    rating: int = Field(ge=1, le=5)
    #: `None` on Android -- Play reviews have no title field at all, unlike
    #: iOS. Rendered as absent, not as a placeholder string.
    title: str | None = None
    content: str
    author: str
    version: str | None = None
    updated: datetime | None = None

    @property
    def is_critical(self) -> bool:
        """Whether this review counts toward the gap-analysis corpus (≤3★)."""
        return self.rating <= 3


class ReviewStats(BaseModel):
    """Rating arithmetic over the fetched sample, computed in code (SC-3).

    Never touches the model -- a star count is either right or a bug, not
    something worth spending a call on.
    """

    model_config = ConfigDict(frozen=True)

    fetched_count: int
    distribution: dict[int, int]
    critical_count: int
    oldest: datetime | None = None
    newest: datetime | None = None

    @property
    def critical_share(self) -> float:
        """Fraction of the sample that is ≤3★, or 0.0 for an empty sample."""
        return self.critical_count / self.fetched_count if self.fetched_count else 0.0

    @property
    def average(self) -> float:
        """Mean star rating of the fetched sample, or 0.0 for an empty sample."""
        if not self.fetched_count:
            return 0.0
        total = sum(stars * count for stars, count in self.distribution.items())
        return total / self.fetched_count


# --------------------------------------------------------------------------- #
# Gap analysis
# --------------------------------------------------------------------------- #

Severity = Literal["high", "medium", "low"]


class FeatureGap(BaseModel):
    """One recurring complaint pattern, as the model reported it.

    `review_ids` are citations, not quotes -- the model is never asked to
    reproduce review text, so there is nothing for it to get subtly wrong. The
    renderer resolves these against the app's own fetched reviews and drops
    any id that does not resolve (SC-2).
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    severity: Severity
    review_ids: tuple[str, ...] = ()

    @field_validator("review_ids", mode="before")
    @classmethod
    def _coerce_ids(cls, value: object) -> object:
        """Accept a list of ints or a single string, not only a list of strings.

        Models asked for an array of string ids sometimes hand back integers
        instead; rejecting the reply over that would throw away a good
        analysis over punctuation.
        """
        if isinstance(value, list | tuple):
            return tuple(str(item) for item in value)
        if value is None:
            return ()
        return (str(value),)


class GapAnalysisResult(BaseModel):
    """The model's reply: a short list of gaps, and nothing else."""

    model_config = ConfigDict(extra="forbid")

    gaps: tuple[FeatureGap, ...] = ()

    @classmethod
    def empty(cls) -> GapAnalysisResult:
        """The fallback used when synthesis fails outright."""
        return cls(gaps=())


class StoreSection(StrEnum):
    """Report sections, in render order."""

    SNAPSHOT = "snapshot"
    GAPS = "gaps"
    RAW_REVIEWS = "raw_reviews"


# --------------------------------------------------------------------------- #
# Progress, and the finished report
# --------------------------------------------------------------------------- #

ProgressStage = Literal["validating", "resolving", "fetching", "analyzing", "rendering", "done"]


class ProgressEvent(BaseModel):
    """One step of a run, yielded to the UI as it happens."""

    model_config = ConfigDict(frozen=True)

    stage: ProgressStage
    message: str
    ok: bool = True


class Report(BaseModel):
    """The finished report: one Markdown string, plus what produced it."""

    identity: AppIdentity
    stats: ReviewStats
    reviews: tuple[Review, ...] = ()
    gaps: tuple[FeatureGap, ...] = ()
    markdown: str
    generated_on: date
    #: True when the sample had too few critical reviews for a gap analysis to
    #: be attempted at all (SC-4) -- distinct from `analysis_failed`, which
    #: means an attempt was made and the model could not produce a usable reply.
    insufficient_signal: bool = False
    analysis_failed: bool = False

    @field_validator("markdown")
    @classmethod
    def _must_not_be_empty(cls, value: str) -> str:
        """A report with no body is a bug, not a degraded result."""
        if not value.strip():
            raise ValueError("report markdown is empty")
        return value
