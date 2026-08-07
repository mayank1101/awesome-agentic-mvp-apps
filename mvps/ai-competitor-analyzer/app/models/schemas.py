"""Domain models.

Every boundary in this app is a Pydantic model: what search returned, what was
packed for the prompt, what the model replied, and what the renderer turns into
Markdown. Pure data -- no I/O, no Streamlit, no provider SDKs.

The one load-bearing model is :class:`SynthesisResult`. Its six required fields
are what make "all six sections are always present" (SC-2) a structural property
rather than a prompt instruction that mostly holds.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


class SectionKey(StrEnum):
    """The six report sections, in render order.

    The value doubles as the JSON field name the model fills in and as the prefix
    of every evidence id in that section, so there is exactly one spelling of
    "the pricing section" in the whole app.
    """

    SNAPSHOT = "snapshot"
    PRODUCT = "product"
    PRICING = "pricing"
    POSITIONING = "positioning"
    RECENT_MOVES = "recent_moves"
    STRENGTHS_WEAKNESSES = "strengths_weaknesses"


#: Heading text for each section, and the short id used in the docs (R-1..R-6).
SECTION_TITLES: dict[SectionKey, str] = {
    SectionKey.SNAPSHOT: "Company snapshot",
    SectionKey.PRODUCT: "Product & capabilities",
    SectionKey.PRICING: "Pricing & packaging",
    SectionKey.POSITIONING: "Positioning & target customer",
    SectionKey.RECENT_MOVES: "Recent moves (last 12 months)",
    SectionKey.STRENGTHS_WEAKNESSES: "Strengths & weaknesses",
}

#: What a section says when its searches supported nothing. Stated plainly is a
#: result; filled with model-memory prose is the failure this app exists to
#: avoid, so this string is a first-class output rather than an error path.
NOT_FOUND = "Not found in public sources."


# --------------------------------------------------------------------------- #
# Input and identity
# --------------------------------------------------------------------------- #


class AnalysisRequest(BaseModel):
    """A validated request to profile one company.

    Both fields arrive already normalised -- NFKC-folded, control characters
    stripped, domain reduced to a bare host (E-02, E-05). Constructing this model
    from raw user input directly would skip that, so the UI goes through
    `app.services.normalize` first.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    domain: str | None = None


class CompanyIdentity(BaseModel):
    """Which company the app decided it is profiling.

    Rendered at the top of every report. A wrong-company brief is this app's worst
    possible output, and showing the resolution is what turns it from an
    invisible failure into one the reader catches in two seconds (Q-1).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    domain: str | None = None
    description: str | None = None
    #: True when the user supplied the domain, so no resolution search was spent.
    supplied_by_user: bool = False


# --------------------------------------------------------------------------- #
# Search and evidence
# --------------------------------------------------------------------------- #


class SearchHit(BaseModel):
    """One result from the search provider, normalised to what this app uses."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: str | None = None

    @property
    def host(self) -> str:
        """The registrable domain of this hit, lowercased and without `www.`."""
        netloc = urlparse(self.url).netloc.lower().removeprefix("www.")
        parts = netloc.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


class EvidenceItem(BaseModel):
    """One packed, trimmed, id-stamped snippet handed to the model.

    The id is what the model cites; the URL is deliberately *not* shown to it, so
    a fabricated link is impossible rather than merely unlikely (SC-1). The
    renderer attaches URLs afterwards from these same records.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    url: str
    content: str
    published_date: str | None = None


class SectionEvidence(BaseModel):
    """Everything one section will be written from."""

    model_config = ConfigDict(frozen=True)

    section: SectionKey
    query: str
    items: tuple[EvidenceItem, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when no evidence survived, so the section must say NOT_FOUND."""
        return not self.items


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


class SynthesisResult(BaseModel):
    """The model's reply: six sections of prose, and nothing else.

    `extra="forbid"` is deliberate. A reply with an extra key is a reply that did
    not follow the schema, and finding that out here -- where there is a repair
    retry -- is better than finding it out in the renderer (E-34).
    """

    model_config = ConfigDict(extra="forbid")

    snapshot: str
    product: str
    pricing: str
    positioning: str
    recent_moves: str
    strengths_weaknesses: str

    @field_validator("*", mode="before")
    @classmethod
    def _flatten_structures(cls, value: object) -> object:
        """Accept a list or object where a string was asked for.

        Seen live: asked for six string fields, a model returned
        ``strengths_weaknesses`` as ``{"strengths": [...], "weaknesses": [...]}``.
        That is the model answering the question correctly in the wrong shape,
        and rejecting it would throw away a good brief over punctuation. Lists
        become bullets and objects become labelled bullet groups, which is what
        the section would have looked like written out.
        """
        if isinstance(value, str) or value is None:
            return value

        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)

        if isinstance(value, dict):
            blocks: list[str] = []
            for key, nested in value.items():
                label = str(key).replace("_", " ").capitalize()
                if isinstance(nested, list):
                    body = "\n".join(f"- {item}" for item in nested)
                    blocks.append(f"**{label}**\n{body}")
                else:
                    blocks.append(f"**{label}:** {nested}")
            return "\n\n".join(blocks)

        return value

    def section(self, key: SectionKey) -> str:
        """Return the prose for one section.

        Args:
            key: Which section to read.

        Returns:
            The section body, or :data:`NOT_FOUND` if the model left it blank.
        """
        value = getattr(self, key.value, "").strip()
        return value or NOT_FOUND

    @classmethod
    def all_not_found(cls) -> SynthesisResult:
        """Return the fallback used when synthesis fails outright (E-34, E-38).

        A structurally valid report whose sources are real beats an error page:
        the search half of the work still has value to the reader.
        """
        return cls(**{key.value: NOT_FOUND for key in SectionKey})


# --------------------------------------------------------------------------- #
# Budget, progress, and the finished report
# --------------------------------------------------------------------------- #


class RunBudget(BaseModel):
    """The app's cap on its own search spend for one report (SC-8, E-30).

    Checked before every call, never after. This is not the provider's quota --
    that arrives as a 429 and is a different error entirely.
    """

    limit: int = 8
    spent: int = 0

    def can_afford(self, cost: int = 1) -> bool:
        """Whether `cost` more credits fit inside the cap."""
        return self.spent + cost <= self.limit

    def charge(self, cost: int = 1) -> None:
        """Record a spend that has already been authorised by :meth:`can_afford`."""
        self.spent += cost

    @property
    def remaining(self) -> int:
        """Credits left before the cap."""
        return max(0, self.limit - self.spent)


ProgressStage = Literal["validating", "resolving", "searching", "packing", "synthesising", "done"]


class ProgressEvent(BaseModel):
    """One step of a run, yielded to the UI as it happens.

    The pipeline yields these rather than taking a callback: a callback would
    invite painting from whatever thread produced it, and Streamlit only tolerates
    painting from its own script thread.
    """

    model_config = ConfigDict(frozen=True)

    stage: ProgressStage
    message: str
    section: SectionKey | None = None
    #: Set on terminal events so the UI can distinguish "finished" from "gave up".
    ok: bool = True


class Report(BaseModel):
    """The finished brief: one Markdown string, plus what produced it."""

    identity: CompanyIdentity
    markdown: str
    generated_on: date
    evidence: tuple[SectionEvidence, ...] = ()
    credits_spent: int = 0
    synthesis_failed: bool = False

    @field_validator("markdown")
    @classmethod
    def _must_not_be_empty(cls, value: str) -> str:
        """A report with no body is a bug, not a degraded result."""
        if not value.strip():
            raise ValueError("report markdown is empty")
        return value

    @property
    def source_count(self) -> int:
        """How many distinct URLs back this report."""
        return len({item.url for section in self.evidence for item in section.items})
