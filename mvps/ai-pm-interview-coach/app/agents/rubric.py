"""The grading rubric, as data.

Five dimensions, each with four written level anchors. Anchors rather than
adjectives is the point: "was the structure good?" leaves the bar to the model's
taste and produces the failure mode this whole design is built against -- a
grader that returns the same comfortable score for everything.

Being data rather than prose inside a prompt template buys three things:

* the anchors can be asserted on -- five dimensions, four levels, no gaps;
* the same text renders in the UI before the interview starts, so a candidate is
  told what they are being measured on;
* the scale can be reasoned about in one place. There is no midpoint, so every
  dimension is forced into below-bar (1-2) or at-bar (3-4).
"""

from dataclasses import dataclass

from app.models.schemas import ALL_DIMENSIONS, RubricDimension

#: The four score levels, low to high.
LEVELS: tuple[int, ...] = (1, 2, 3, 4)

#: What the halves of the scale mean, stated once and reused in both the
#: evaluator's instructions and the report UI.
BELOW_BAR = (1, 2)
AT_BAR = (3, 4)


@dataclass(frozen=True)
class Dimension:
    """One rubric dimension and the four descriptions that anchor its scale.

    Attributes:
        key: The `RubricDimension` literal this describes.
        label: Display name, used in the UI and the report.
        question: The one-line test a grader applies -- what this dimension is
            actually asking.
        anchors: Score to description, for all four levels. Written so that two
            different readers would land on the same number for the same answer,
            which is the only reason a rubric beats an opinion.
    """

    key: RubricDimension
    label: str
    question: str
    anchors: dict[int, str]


RUBRIC: tuple[Dimension, ...] = (
    Dimension(
        key="structure",
        label="Structure",
        question="Was there a stated approach, and was it actually followed?",
        anchors={
            1: "No approach. Jumps straight to solutions, or wanders between unrelated points.",
            2: "Names a framework but abandons it, or applies it mechanically without adapting it to the question.",
            3: "States an approach up front and mostly follows it. Signposts where they are.",
            4: "Approach is tailored to this specific question, followed throughout, and revisited when a probe exposes a gap.",
        },
    ),
    Dimension(
        key="user_insight",
        label="User & problem insight",
        question="Was there a specific user with a specific pain, or a demographic?",
        anchors={
            1: "No user named, or 'everyone'. The problem is asserted rather than described.",
            2: "Names a broad segment ('small businesses') but the pain is generic and could apply to any product.",
            3: "Names a specific segment and a concrete pain, with a plausible reason it goes unsolved today.",
            4: "Segments by behaviour or context rather than demographics, and shows why this user's pain differs from the adjacent user's.",
        },
    ),
    Dimension(
        key="prioritization",
        label="Prioritization & trade-offs",
        question="Was a choice made, for a reason, with its cost acknowledged?",
        anchors={
            1: "Lists options without choosing, or proposes everything at once.",
            2: "Picks something but the reason is unstated, or the cost of not doing the alternatives is ignored.",
            3: "Makes a clear choice with a stated reason and names what is being given up.",
            4: "Choice follows from a stated criterion, the cost is quantified or bounded, and the answer holds up when the trade-off is challenged.",
        },
    ),
    Dimension(
        key="metrics",
        label="Metrics rigour",
        question="Is there a success metric that would actually move, and a counter-metric?",
        anchors={
            1: "No metric offered, or a vanity metric with no link to the stated goal.",
            2: "Names a plausible metric but nothing that would catch the obvious way to game it.",
            3: "Names a success metric tied to the goal plus a counter-metric or guardrail.",
            4: "Metric, counter-metric, and a rough sense of the magnitude that would count as success -- with the measurement's weakness acknowledged.",
        },
    ),
    Dimension(
        key="communication",
        label="Communication",
        question="Signal per sentence, and did it survive the probes?",
        anchors={
            1: "Rambling or so terse there is nothing to assess. The listener has to reconstruct the point.",
            2: "Understandable but padded, or leans on jargon in place of reasoning.",
            3: "Clear and reasonably tight. Answers the question that was asked.",
            4: "Leads with the answer, then supports it. Concedes cleanly when a probe lands rather than defending a weak point.",
        },
    ),
)

#: Lookup by key, for the places that have a `RubricDimension` in hand.
BY_KEY: dict[RubricDimension, Dimension] = {dimension.key: dimension for dimension in RUBRIC}


def dimension(key: RubricDimension) -> Dimension:
    """Return one dimension by its key."""
    return BY_KEY[key]


def format_for_instructions() -> str:
    """Render the whole rubric for the evaluator's instructions.

    Every anchor is sent, not a summary. The anchors are the calibration -- a
    grader given only dimension names has nothing to calibrate against, which is
    how every answer ends up scoring the same.
    """
    blocks: list[str] = []
    for dim in RUBRIC:
        anchors = "\n".join(f"  {score} = {text}" for score, text in sorted(dim.anchors.items()))
        blocks.append(f"{dim.key} ({dim.label}) -- {dim.question}\n{anchors}")
    return "\n\n".join(blocks)


def _validate() -> None:
    """Fail at import if the rubric is malformed.

    Cheap, and it turns a data-entry slip into an immediate error rather than a
    subtly wrong report much later. The equivalent assertions also exist as tests;
    this is the belt for the places that import the module without running them.
    """
    keys = [dimension.key for dimension in RUBRIC]
    if len(keys) != len(set(keys)):
        raise ValueError("rubric has a duplicate dimension key")
    if set(keys) != set(ALL_DIMENSIONS):
        raise ValueError(f"rubric keys {sorted(keys)} do not match {sorted(ALL_DIMENSIONS)}")
    for dim in RUBRIC:
        if set(dim.anchors) != set(LEVELS):
            raise ValueError(f"{dim.key} must define an anchor for every level in {LEVELS}")
        for score, text in dim.anchors.items():
            if not text.strip():
                raise ValueError(f"{dim.key} level {score} has an empty anchor")


_validate()
