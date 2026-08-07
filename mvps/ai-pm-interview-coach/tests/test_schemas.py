"""Domain model validation.

The report validator and the budget arithmetic get the most attention here: a
report missing a dimension is silently wrong rather than partially useful, and an
off-by-one in `is_complete` either ends the interview early or asks a question
past the cap.
"""

import pytest
from pydantic import ValidationError

from app.core.config import get_settings
from app.models.schemas import (
    ALL_DIMENSIONS,
    CandidateAnswer,
    Deduction,
    DimensionScore,
    FeedbackReport,
    InterviewConfig,
    RewrittenAnswer,
    Transcript,
    Turn,
)


def _score(dimension: str, score: int = 3) -> DimensionScore:
    return DimensionScore(
        dimension=dimension,
        score=score,
        justification="because of the probe",
        evidence="the candidate said something specific",
    )


def _report(scores: list[DimensionScore] | None = None) -> FeedbackReport:
    return FeedbackReport(
        headline="Strong framing, weak on measurement.",
        scores=scores if scores is not None else [_score(d) for d in ALL_DIMENSIONS],
        what_worked=["named a specific host segment"],
        what_cost_points=[Deduction(moment="no metric offered", stronger_move="name a north star")],
        rewritten_answer=RewrittenAnswer(
            question="How would you price it?",
            rewrite="I would start from the host's willingness to pay...",
            why_better="commits to a metric and a trade-off",
        ),
        next_focus="metrics: state a success metric and a counter-metric for every answer",
    )


def _transcript(answers: list[str | None], budget: int = 4) -> Transcript:
    return Transcript(
        config=InterviewConfig(followup_budget=budget),
        turns=[
            Turn(index=i, question=f"question {i}", answer=answer)
            for i, answer in enumerate(answers)
        ],
    )


# --- InterviewConfig ---------------------------------------------------------


@pytest.mark.parametrize(("budget", "expected"), [(2, 3), (4, 5), (6, 7)])
def test_total_questions_is_budget_plus_the_opener(budget, expected):
    assert InterviewConfig(followup_budget=budget).total_questions == expected


def test_followup_budget_is_restricted_to_the_offered_values():
    with pytest.raises(ValidationError):
        InterviewConfig(followup_budget=3)


def test_focus_area_is_length_capped():
    # focus_area is the one config value interpolated into instructions, where
    # the fence cannot protect it.
    with pytest.raises(ValidationError):
        InterviewConfig(focus_area="x" * 121)


# --- Turn --------------------------------------------------------------------


def test_unanswered_turn_is_not_answered():
    assert Turn(index=0, question="q").is_answered is False


def test_whitespace_only_answer_does_not_count_as_answered():
    assert Turn(index=0, question="q", answer="   \n ").is_answered is False


def test_empty_question_is_rejected():
    # An empty assistant message would corrupt the projection's pairing.
    with pytest.raises(ValidationError):
        Turn(index=0, question="")


# --- CandidateAnswer ---------------------------------------------------------


def test_answer_at_the_cap_is_accepted():
    limit = get_settings().max_answer_chars
    assert len(CandidateAnswer(text="x" * limit).text) == limit


def test_answer_over_the_cap_is_rejected():
    limit = get_settings().max_answer_chars
    with pytest.raises(ValidationError):
        CandidateAnswer(text="x" * (limit + 1))


def test_empty_answer_is_rejected():
    with pytest.raises(ValidationError):
        CandidateAnswer(text="")


def test_stored_turn_is_not_subject_to_the_inbound_cap():
    # Turn models already-stored history, so lowering the cap must not make an
    # existing conversation unreadable.
    long_answer = "x" * (get_settings().max_answer_chars + 500)
    assert Turn(index=0, question="q", answer=long_answer).is_answered is True


# --- Transcript --------------------------------------------------------------


def test_answered_turns_ignores_open_and_blank_answers():
    transcript = _transcript(["real answer", "  ", None])
    assert transcript.answered_turns == 1


def test_awaiting_answer_tracks_the_last_turn():
    assert _transcript(["a", None]).awaiting_answer is True
    assert _transcript(["a", "b"]).awaiting_answer is False


def test_empty_transcript_is_not_awaiting_an_answer():
    assert _transcript([]).awaiting_answer is False


@pytest.mark.parametrize("budget", [2, 4, 6])
def test_is_complete_exactly_at_the_budget_boundary(budget):
    total = budget + 1
    one_short = _transcript(["a"] * (total - 1), budget=budget)
    exact = _transcript(["a"] * total, budget=budget)

    assert one_short.is_complete is False
    assert exact.is_complete is True


def test_asked_but_unanswered_question_does_not_complete_the_interview():
    # Five turns exist, but the last was abandoned, so the slot is unused.
    transcript = _transcript(["a", "a", "a", "a", None], budget=4)
    assert transcript.is_complete is False


def test_one_answer_is_enough_to_grade():
    assert _transcript(["a"]).is_gradable is True


def test_opening_question_alone_is_not_gradable():
    # E-39: grading this would have the evaluator invent five scores from no
    # evidence, which is worse than no report.
    assert _transcript([None]).is_gradable is False
    assert _transcript([]).is_gradable is False


# --- FeedbackReport ----------------------------------------------------------


def test_report_with_every_dimension_validates():
    assert len(_report().scores) == len(ALL_DIMENSIONS)


def test_report_rejects_a_missing_dimension():
    partial = [_score(d) for d in ALL_DIMENSIONS[:-1]]
    with pytest.raises(ValidationError, match="missing scores"):
        _report(partial)


def test_report_rejects_a_duplicated_dimension():
    duplicated = [_score(d) for d in ALL_DIMENSIONS] + [_score("metrics")]
    with pytest.raises(ValidationError, match="more than once"):
        _report(duplicated)


@pytest.mark.parametrize("score", [0, 5, -1, 100])
def test_scores_outside_one_to_four_are_rejected(score):
    with pytest.raises(ValidationError):
        _score("metrics", score=score)


def test_empty_evidence_is_rejected():
    # The structural half of the anti-inflation design: a 4 needs a quotable
    # moment behind it or the report does not validate.
    with pytest.raises(ValidationError):
        DimensionScore(
            dimension="metrics",
            score=4,
            justification="strong",
            evidence="",
        )


def test_deduction_requires_the_stronger_move():
    with pytest.raises(ValidationError):
        Deduction(moment="was vague", stronger_move="")


def test_score_for_returns_the_matching_dimension():
    report = _report([_score(d, score=2 if d == "metrics" else 4) for d in ALL_DIMENSIONS])
    assert report.score_for("metrics").score == 2


def test_ordered_scores_leads_with_primaries_and_drops_nothing():
    report = _report()
    ordered = report.ordered_scores(("metrics", "prioritization"))

    assert [score.dimension for score in ordered[:2]] == ["metrics", "prioritization"]
    assert len(ordered) == len(ALL_DIMENSIONS)
    assert {score.dimension for score in ordered} == set(ALL_DIMENSIONS)


def test_ordered_scores_without_primaries_keeps_canonical_order():
    assert [score.dimension for score in _report().ordered_scores()] == list(ALL_DIMENSIONS)
