"""Calibration data: what each configuration axis changes about the interview.

Three independent axes, looked up and combined into the interviewer's
instructions:

* **interview type** -- the shape of the question, and which rubric dimensions
  this format can actually exercise;
* **seniority** -- where the bar sits;
* **company archetype** -- what a good answer optimises for, which is the axis
  most often left out of practice tools and the one that most changes the answer.

Kept as data next to the rubric rather than inside prompt strings, so the
combinations are inspectable and testable without calling a model.
"""

from dataclasses import dataclass

from app.models.schemas import (
    ALL_DIMENSIONS,
    CompanyArchetype,
    InterviewType,
    RubricDimension,
    Seniority,
)


@dataclass(frozen=True)
class InterviewTypePreset:
    """One kind of PM interview.

    Attributes:
        label: Display name.
        question_brief: What the opening question should look like, written as an
            instruction to the interviewer.
        probe_angles: The directions a follow-up should press in for this format.
            Gives the "ground your follow-up" rule a target rather than only a
            prohibition.
        primary_dimensions: The rubric dimensions this format genuinely tests.
            All five are still scored -- dropping one would let the model quietly
            skip the dimension it found hard -- but the report leads with these
            and `next_focus` is drawn only from them. Grading metrics rigour in a
            behavioural interview, off three answers where metrics never
            legitimately came up, produces a number that is both unfair and
            uninformative.
    """

    label: str
    question_brief: str
    probe_angles: tuple[str, ...]
    primary_dimensions: tuple[RubricDimension, ...]


@dataclass(frozen=True)
class SeniorityPreset:
    """Where the bar sits for a given level.

    Attributes:
        label: Display name.
        bar: What separates an adequate answer from a strong one at this level,
            phrased for the interviewer's instructions.
    """

    label: str
    bar: str


@dataclass(frozen=True)
class ArchetypePreset:
    """The company context an answer is judged inside.

    Attributes:
        label: Display name.
        optimises_for: What "good" means here -- the same answer is strong at one
            archetype and weak at another, which is exactly the calibration a
            question bank cannot provide.
        context: One line of situational detail for the question.
    """

    label: str
    optimises_for: str
    context: str


INTERVIEW_TYPE_PRESETS: dict[InterviewType, InterviewTypePreset] = {
    "product_design": InterviewTypePreset(
        label="Product design / sense",
        question_brief=(
            "Ask an open design question about improving or building something for a "
            "specific user group. Deliberately under-specified, so the candidate has "
            "to choose a user and a problem."
        ),
        probe_angles=(
            "which user they chose and why that one rather than an adjacent segment",
            "what evidence they have that the pain is real",
            "what they are deliberately not building",
        ),
        primary_dimensions=("user_insight", "structure", "prioritization"),
    ),
    "execution_metrics": InterviewTypePreset(
        label="Execution & metrics",
        question_brief=(
            "Ask about a metric that moved unexpectedly, or how they would measure and "
            "improve a specific outcome. Give a number and a timeframe, and withhold "
            "the cause."
        ),
        probe_angles=(
            "how they would distinguish the cause from a correlated one",
            "which counter-metric would catch the obvious gaming of their fix",
            "what they would do first given a week",
        ),
        primary_dimensions=("metrics", "prioritization", "structure"),
    ),
    "strategy": InterviewTypePreset(
        label="Strategy",
        question_brief=(
            "Ask whether the company should enter, exit, or reposition in some market, "
            "or how to respond to a competitor's move. Force a recommendation."
        ),
        probe_angles=(
            "what would have to be true for their recommendation to be wrong",
            "what they would give up to fund it",
            "why now rather than in a year",
        ),
        primary_dimensions=("prioritization", "structure", "communication"),
    ),
    "behavioral": InterviewTypePreset(
        label="Behavioral",
        question_brief=(
            "Ask for a specific past situation -- a disagreement, a failure, a call made "
            "without enough information. Ask for one instance, not a general policy."
        ),
        probe_angles=(
            "what the other party's case actually was, in their own words",
            "what they would do differently and what they would repeat",
            "what the cost of their decision was to someone else",
        ),
        primary_dimensions=("communication", "user_insight", "prioritization"),
    ),
    "analytical": InterviewTypePreset(
        label="Analytical / estimation",
        question_brief=(
            "Ask for a sized estimate or a quantitative judgement that cannot be looked "
            "up. Give no data; the assumptions are the answer."
        ),
        probe_angles=(
            "which assumption their estimate is most sensitive to",
            "how they would sanity-check the result",
            "what they would measure to replace the weakest assumption",
        ),
        primary_dimensions=("structure", "metrics", "communication"),
    ),
}


SENIORITY_PRESETS: dict[Seniority, SeniorityPreset] = {
    "apm": SeniorityPreset(
        label="APM",
        bar=(
            "A structured, reasoned answer is enough. Do not expect organisational "
            "awareness or a strategy view."
        ),
    ),
    "pm": SeniorityPreset(
        label="PM",
        bar=(
            "Expect a clear choice with a stated reason, a named user, and at least one "
            "success metric."
        ),
    ),
    "senior_pm": SeniorityPreset(
        label="Senior PM",
        bar=(
            "Expect the trade-off to be defended under pressure, second-order effects "
            "considered, and the weakest part of their own answer acknowledged before "
            "you point at it."
        ),
    ),
    "lead_pm": SeniorityPreset(
        label="Lead / Group PM",
        bar=(
            "Expect a position on where this fits in a wider portfolio, what they would "
            "stop doing to fund it, and how they would know to kill it."
        ),
    ),
}


ARCHETYPE_PRESETS: dict[CompanyArchetype, ArchetypePreset] = {
    "big_tech": ArchetypePreset(
        label="Big Tech",
        optimises_for="scale, platform leverage, and not breaking existing users",
        context="Hundreds of millions of users, a slow release process, and adjacent teams who own neighbouring surfaces.",
    ),
    "growth_startup": ArchetypePreset(
        label="Growth-stage startup",
        optimises_for="speed of learning and finding the next step-change in growth",
        context="Eighteen months of runway, a small team, and no established process to lean on.",
    ),
    "b2b_saas": ArchetypePreset(
        label="B2B SaaS",
        optimises_for="contract value, retention, and the buyer being a different person from the user",
        context="A few hundred paying accounts, an enterprise sales cycle, and a roadmap under pressure from named customers.",
    ),
    "consumer_marketplace": ArchetypePreset(
        label="Consumer marketplace",
        optimises_for="liquidity and the balance between both sides of the market",
        context="Supply and demand in different cities at different maturities, where a fix for one side can starve the other.",
    ),
}


def primary_dimensions(interview_type: InterviewType) -> tuple[RubricDimension, ...]:
    """The rubric dimensions this interview format actually tests."""
    return INTERVIEW_TYPE_PRESETS[interview_type].primary_dimensions


def secondary_dimensions(interview_type: InterviewType) -> tuple[RubricDimension, ...]:
    """The remaining dimensions: still scored, just not what the session is judged on."""
    primary = set(primary_dimensions(interview_type))
    return tuple(dimension for dimension in ALL_DIMENSIONS if dimension not in primary)


def _validate() -> None:
    """Fail at import if a preset table is incomplete or names a bad dimension."""
    for interview_type, preset in INTERVIEW_TYPE_PRESETS.items():
        unknown = set(preset.primary_dimensions) - set(ALL_DIMENSIONS)
        if unknown:
            raise ValueError(f"{interview_type} names unknown dimensions: {sorted(unknown)}")
        if len(set(preset.primary_dimensions)) != len(preset.primary_dimensions):
            raise ValueError(f"{interview_type} repeats a primary dimension")
        if not preset.primary_dimensions:
            raise ValueError(f"{interview_type} declares no primary dimensions")
        if len(preset.primary_dimensions) >= len(ALL_DIMENSIONS):
            # If everything is primary, the distinction buys nothing and the
            # report has no ordering to apply.
            raise ValueError(f"{interview_type} marks every dimension primary")


_validate()
