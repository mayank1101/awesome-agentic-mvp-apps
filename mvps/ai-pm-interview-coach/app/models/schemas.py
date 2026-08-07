"""Domain models for the interview pipeline.

The flow through these types is:

    InterviewConfig  --(interviewer agent)-->  a question per turn
    Turn / Transcript                          the conversation, as a projection
                                               over the framework's stored
                                               messages
    Transcript       --(evaluator agent)-->    FeedbackReport

They are the contract between the UI and the agent layer; neither side defines
its own copy of a field or a limit.

:class:`Transcript` is deliberately *not* storage. The verbatim conversation
lives in the framework's ``AgentSession.state["messages"]``; this is the
domain-shaped view of it, built by :mod:`app.services.transcript`.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

InterviewType = Literal[
    "product_design",
    "execution_metrics",
    "strategy",
    "behavioral",
    "analytical",
]
Seniority = Literal["apm", "pm", "senior_pm", "lead_pm"]
CompanyArchetype = Literal["big_tech", "growth_startup", "b2b_saas", "consumer_marketplace"]
FollowUpBudget = Literal[2, 4, 6]
RubricDimension = Literal[
    "structure",
    "user_insight",
    "prioritization",
    "metrics",
    "communication",
]
SessionPhase = Literal["idle", "interviewing", "ending", "reported", "expired"]

#: Every dimension, in the order a report should present them. Derived from the
#: Literal so the two can never drift apart.
ALL_DIMENSIONS: tuple[RubricDimension, ...] = (
    "structure",
    "user_insight",
    "prioritization",
    "metrics",
    "communication",
)

#: The one configuration value that reaches the agent's *instructions* rather
#: than travelling inside the fence, so its length is capped here and its
#: content is defanged in `app.agents.prompts`.
FocusArea = Annotated[str, StringConstraints(max_length=120)]


class InterviewConfig(BaseModel):
    """What the candidate chose before starting, and the limits on it.

    A structured form rather than freeform chat: consistent axes give the model
    consistent signal, and they are what `presets.py` looks up to calibrate the
    question.
    """

    interview_type: InterviewType = Field(
        default="product_design",
        description="Which kind of PM interview to run. Drives the question shape.",
    )
    seniority: Seniority = Field(
        default="pm",
        description="The bar the answer is held to.",
    )
    archetype: CompanyArchetype = Field(
        default="big_tech",
        description=(
            "The company context, which decides what a good answer optimises "
            "for -- scale, speed, contract value, or liquidity."
        ),
    )
    focus_area: FocusArea | None = Field(
        default=None,
        description="Optional domain to anchor the question in, e.g. 'payments'.",
    )
    followup_budget: FollowUpBudget = Field(
        default=4,
        description="How many follow-ups after the opening question. Total questions is this plus one.",
    )

    @property
    def total_questions(self) -> int:
        """Questions in a full session: the opener plus every follow-up."""
        return self.followup_budget + 1


class CandidateAnswer(BaseModel):
    """One answer on its way in, and where the length cap is enforced.

    The cap lives here rather than on :class:`Turn` on purpose. ``Turn`` models
    what is *already stored*, and the projection rebuilds it from the framework's
    message history -- so a length validator there would start raising mid-session
    if ``MAX_ANSWER_CHARS`` were ever lowered, breaking a conversation that was
    valid when it was written. Validating the inbound value instead keeps the cap
    unbypassable at the one point where it matters and leaves stored history
    readable forever.

    The limit is read from settings rather than hard-coded, so the `.env` value is
    the single source of truth for it.
    """

    text: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _enforce_length_cap(self) -> "CandidateAnswer":
        from app.core.config import get_settings

        limit = get_settings().max_answer_chars
        if len(self.text) > limit:
            raise ValueError(f"answer is {len(self.text)} characters; the limit is {limit}")
        return self

    @property
    def stripped(self) -> str:
        """The answer with surrounding whitespace removed."""
        return self.text.strip()


class Turn(BaseModel):
    """One question and the answer to it.

    ``answer is None`` means "asked, awaiting the candidate", which is the state
    the interview sits in for most of its life.

    Deliberately permissive about answer length -- see :class:`CandidateAnswer`.
    """

    index: int = Field(..., ge=0, description="Zero-based position in the interview.")
    question: str = Field(..., min_length=1, description="What the interviewer asked.")
    answer: str | None = Field(
        default=None,
        description="The candidate's reply, or None while the question is still open.",
    )

    @property
    def is_answered(self) -> bool:
        """Whether a non-empty answer has been recorded."""
        return bool(self.answer and self.answer.strip())


class Transcript(BaseModel):
    """The conversation so far, as a view over stored messages.

    Carries no behaviour beyond the three derived properties below. Anything
    that mutates the conversation belongs in the session store, and anything
    that builds this object belongs in :mod:`app.services.transcript`.
    """

    config: InterviewConfig
    turns: list[Turn] = Field(default_factory=list)

    @property
    def answered_turns(self) -> int:
        """How many questions have a real answer behind them."""
        return sum(1 for turn in self.turns if turn.is_answered)

    @property
    def awaiting_answer(self) -> bool:
        """Whether the last question is still open."""
        return bool(self.turns) and not self.turns[-1].is_answered

    @property
    def is_complete(self) -> bool:
        """Whether every question in the budget has been answered.

        The comparison is against answered turns rather than asked ones: a
        question that was asked and abandoned has not used up its slot.
        """
        return self.answered_turns >= self.config.total_questions

    @property
    def is_gradable(self) -> bool:
        """Whether there is anything worth sending to the evaluator.

        One answer is enough. Zero is not: the opening question alone is not an
        interview, and grading it would have the evaluator invent five scores
        from no evidence.
        """
        return self.answered_turns >= 1


class DimensionScore(BaseModel):
    """One rubric dimension, scored with the quote that earned it.

    ``evidence`` is required and non-empty *by schema*. That is the structural
    half of the anti-inflation design: a score the model cannot attach a moment
    from the transcript to fails validation rather than reaching the candidate.
    """

    dimension: RubricDimension = Field(..., description="Which rubric dimension this scores.")
    score: int = Field(
        ...,
        ge=1,
        le=4,
        description=(
            "1-4, where 1-2 is below the bar and 3-4 is at or above it. There is "
            "no midpoint, so every dimension forces a call."
        ),
    )
    justification: str = Field(
        ...,
        min_length=1,
        description="One line on why this score and not the one above or below it.",
    )
    evidence: str = Field(
        ...,
        min_length=1,
        description="A specific moment from the transcript, quoted, that earned this score.",
    )


class Deduction(BaseModel):
    """Something that cost points, and what to have done instead.

    ``stronger_move`` is required so "you were vague" cannot ship without naming
    the alternative -- feedback that only diagnoses is not actionable.
    """

    moment: str = Field(
        ..., min_length=1, description="What happened, quoted or closely described."
    )
    stronger_move: str = Field(
        ...,
        min_length=1,
        description="The specific better move available at that point.",
    )


class RewrittenAnswer(BaseModel):
    """The candidate's weakest answer, rewritten at hire-bar quality.

    Shows the delta rather than describing it, which is the difference between
    recognising a good answer and being able to produce one.
    """

    question: str = Field(..., min_length=1, description="The question being re-answered.")
    rewrite: str = Field(..., min_length=1, description="The stronger answer, in full.")
    why_better: str = Field(
        ...,
        min_length=1,
        description="What the rewrite does that the original did not.",
    )


class FeedbackReport(BaseModel):
    """The graded result of one interview, and the only artifact that outlives it."""

    headline: str = Field(
        ...,
        min_length=1,
        description="One sentence a candidate would remember, naming the main strength and gap.",
    )
    scores: list[DimensionScore] = Field(
        ...,
        description="Exactly one score per rubric dimension.",
    )
    what_worked: list[str] = Field(
        default_factory=list,
        description="Two or three specific moments that went well, quoted from the transcript.",
    )
    what_cost_points: list[Deduction] = Field(
        default_factory=list,
        description="Moments that lost points, each with the stronger move named.",
    )
    rewritten_answer: RewrittenAnswer = Field(
        ...,
        description="The weakest answer, rewritten.",
    )
    next_focus: str = Field(
        ...,
        min_length=1,
        description=(
            "One dimension to practise next and a concrete drill for it. Drawn "
            "from the interview type's primary dimensions, so it never points at "
            "something this format could not exercise."
        ),
    )

    @model_validator(mode="after")
    def _require_every_dimension_once(self) -> "FeedbackReport":
        """Reject a report that is missing a dimension or repeats one.

        A partial report is not partially useful -- it is silently wrong. A
        candidate reading four scores has no way to know the fifth was dropped
        rather than deliberately withheld, so the omission has to be an error
        here rather than a gap on screen.
        """
        seen = [score.dimension for score in self.scores]
        duplicates = {dimension for dimension in seen if seen.count(dimension) > 1}
        if duplicates:
            raise ValueError(f"dimensions scored more than once: {sorted(duplicates)}")

        missing = set(ALL_DIMENSIONS) - set(seen)
        if missing:
            raise ValueError(f"missing scores for dimensions: {sorted(missing)}")

        return self

    def score_for(self, dimension: RubricDimension) -> DimensionScore:
        """Return the score for one dimension.

        Safe to index without a guard: the validator has already established
        that every dimension is present exactly once.
        """
        return next(score for score in self.scores if score.dimension == dimension)

    def ordered_scores(
        self, primary_dimensions: tuple[RubricDimension, ...] = ()
    ) -> list[DimensionScore]:
        """Scores with this interview type's primary dimensions first.

        Args:
            primary_dimensions: The dimensions this interview format actually
                tests. Anything not listed keeps its canonical order behind
                them.

        Returns:
            Every score, reordered for presentation. Nothing is dropped -- the
            secondary dimensions are still scored and still shown, they are just
            not what the session is judged on.
        """
        primary = [self.score_for(dimension) for dimension in primary_dimensions]
        rest = [
            self.score_for(dimension)
            for dimension in ALL_DIMENSIONS
            if dimension not in primary_dimensions
        ]
        return primary + rest
