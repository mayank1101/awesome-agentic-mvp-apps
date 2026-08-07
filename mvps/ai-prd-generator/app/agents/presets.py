"""The two dials that shape a PRD: scope and length.

Both are data, not logic, and both are read by two very different callers -- the
agents use the prompt-facing fields, the UI uses ``label`` for its widgets. They
live here so neither side owns the other's copy.
"""

from typing import TypedDict

PRODUCT_SECTIONS_HINT = """
Typical sections for a whole-product PRD (adapt, drop, or add as the product
actually needs, do not force-fit generic sections that don't apply):
Overview, Problem Statement, Goals & Success Metrics, Target Users & Personas,
User Stories / Use Cases, Functional Requirements, Non-Functional Requirements
(performance, security, scalability, reliability), Technical Considerations /
Architecture Notes, Out of Scope, Assumptions & Dependencies, Risks &
Mitigations, Milestones / Timeline, Open Questions.
""".strip()

FEATURE_SECTIONS_HINT = """
Typical sections for a feature PRD (adapt, drop, or add as the feature actually
needs, do not force-fit generic sections that don't apply):
Summary, Problem & Motivation, Goals & Success Metrics, Scope (In / Out of
Scope), User Flows & Interaction Details, Functional Requirements, Impact on
Existing Behavior, Technical Approach & Constraints, Edge Cases, Migration &
Rollout (feature flagging, staged release, backwards compatibility), Risks &
Mitigations, Open Questions.

This PRD covers a change to a product that ALREADY EXISTS, not a new product.
Treat the surrounding product, its users, and its tech stack as given, and write
about what changes and what it touches. Do NOT include whole-product sections
such as market analysis, competitive landscape, user personas, monetization or
pricing, org staffing, or multi-quarter roadmaps unless the brief explicitly
calls for them. Scale the plan to a feature: think flags and staged rollout, not
a GA launch programme.
""".strip()


class ScopePreset(TypedDict):
    """How one `scope` value presents itself to the UI and to the agents.

    Attributes:
        label: Human-readable name, used on the scope switch and the badges.
        subject: The noun the prompts use for the thing being specified.
        sections_hint: Candidate sections the outline agent may draw from.
    """

    label: str
    subject: str
    sections_hint: str


class LengthPreset(TypedDict):
    """How one `length` value presents itself to the UI and to the agents.

    Attributes:
        label: Human-readable name, including the rough page count.
        total_words: Soft word budget for the whole document.
        section_count: Section-count range handed to the outline agent, as a
            string because it is guidance ("6-9"), not an exact number.
    """

    label: str
    total_words: int
    section_count: str


SCOPE_PRESETS: dict[str, ScopePreset] = {
    "product": {
        "label": "Product",
        "subject": "product",
        "sections_hint": PRODUCT_SECTIONS_HINT,
    },
    "feature": {
        "label": "Feature",
        "subject": "feature",
        "sections_hint": FEATURE_SECTIONS_HINT,
    },
}

# ~450 words/page. Word budgets are soft guidance passed into the prompt, not
# hard-enforced: LLMs only approximate them, but they keep output far more
# bounded than giving no length instruction at all.
LENGTH_PRESETS: dict[str, LengthPreset] = {
    "short": {"label": "Short · 3-4 pages", "total_words": 1400, "section_count": "4-6"},
    "medium": {"label": "Medium · 6-8 pages", "total_words": 3000, "section_count": "6-9"},
    "long": {"label": "Long · 10-12 pages", "total_words": 5200, "section_count": "9-13"},
}

# ~450 words is one page at typical PRD formatting; used for the page estimate
# shown next to the download button.
WORDS_PER_PAGE = 450

# Floor on the per-section word budget, so a long outline cannot divide the
# total down to a couple of sentences per section.
MIN_SECTION_WORDS = 80
