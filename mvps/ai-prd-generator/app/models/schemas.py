"""Domain models for the PRD pipeline.

The flow through these types is:

    PRDInput  --(outline agent)-->  PRDOutline
              --(section agent)-->  PRDSection, one per outline entry
                                 -->  PRDDocument

They are the contract between the UI and the agent layer; neither side defines
its own copy of a field or a limit.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

PRDLength = Literal["short", "medium", "long"]
PRDScope = Literal["product", "feature"]

# Field caps live here rather than on the widgets so the limits travel with the
# model: the UI's max_chars is a courtesy, PRDInput is what actually enforces
# them, and an uncapped context_notes would blow up prompt size and cost.
Goal = Annotated[str, StringConstraints(max_length=200)]


class PRDInput(BaseModel):
    """The brief a user fills in, and the single source of truth for its limits.

    Deliberately a structured form rather than freeform chat: consistent fields
    give the model consistent signal to work with, while `context_notes` still
    leaves room for pasted research, tickets, and constraints.
    """

    scope: PRDScope = Field(
        default="product",
        description=(
            "Whether this PRD covers a whole product or a single feature inside "
            "an existing one. Drives which sections the outline picks."
        ),
    )
    product_name: str | None = Field(
        default=None, max_length=80, description="Working name of the product/feature"
    )
    parent_product: str | None = Field(
        default=None,
        max_length=600,
        description=(
            "The existing product this feature slots into -- what it does, who "
            "uses it, relevant stack. Required when scope is 'feature'; ignored "
            "for product scope."
        ),
    )
    one_liner: str = Field(
        ...,
        min_length=1,
        max_length=150,
        description="One-sentence description of what you're building",
    )
    problem_statement: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="What problem this solves, and for whom",
    )
    target_users: str = Field(
        ..., min_length=1, max_length=500, description="Primary users / personas"
    )
    goals: list[Goal] | None = Field(
        default=None,
        max_length=20,
        description="Key goals / success metrics, if already known",
    )
    context_notes: str | None = Field(
        default=None,
        max_length=1500,
        description=(
            "Freeform context: research notes, constraints, competitive "
            "landscape, existing tech stack, prior tickets, etc."
        ),
    )
    audience: str = Field(
        default="general",
        min_length=1,
        max_length=100,
        description=(
            "Who this PRD is written for, in plain words -- e.g. 'product "
            "manager', 'engineering team', 'executives', 'general audience'. "
            "Not restricted to any fixed set."
        ),
    )
    length: PRDLength = Field(
        default="medium",
        description="Target PRD length: short (~3-4 pages), medium (~6-8), long (~10-12)",
    )

    @model_validator(mode="after")
    def _require_parent_for_feature(self) -> "PRDInput":
        if self.scope == "feature" and not (self.parent_product or "").strip():
            raise ValueError(
                "parent_product is required when scope is 'feature' -- it describes "
                "the existing product the feature is being added to."
            )
        return self


class PRDSectionOutline(BaseModel):
    """One planned section: its title and what it should cover.

    The field descriptions are part of the prompt contract -- they appear in the
    JSON schema sent to the provider under `json_schema` output mode.
    """

    title: str = Field(..., description="Section title")
    summary: str = Field(..., description="1-2 sentence brief for this section")


class PRDOutline(BaseModel):
    """The plan the outline agent produces, before any section is written."""

    title: str = Field(..., description="Overall PRD title")
    sections: list[PRDSectionOutline] = Field(
        ...,
        description="Ordered sections tailored to this specific product or feature",
    )


class PRDSection(BaseModel):
    """One written section: the outline's title plus the generated body."""

    title: str
    content: str = Field(..., description="Markdown body, without the section heading")


class PRDDocument(BaseModel):
    """A complete PRD, ready to render."""

    title: str
    sections: list[PRDSection]
