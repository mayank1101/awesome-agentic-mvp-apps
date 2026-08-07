"""The messages-to-transcript projection.

Real ``Message`` objects rather than fakes, because the point of these tests is
that the projection survives the shape the *framework* produces -- including a
streamed reply stored as one content part per delta, which is the case that
breaks if the text is read with the framework's own space-joining accessor.
"""

from agent_framework import Message

from app.models.schemas import InterviewConfig
from app.services.guardrails import fence
from app.services.transcript import to_transcript

CONFIG = InterviewConfig(followup_budget=4)


def _q(*parts: str) -> Message:
    """An interviewer question, optionally split across content parts."""
    return Message(role="assistant", contents=list(parts))


def _a(text: str) -> Message:
    """A candidate answer, stored the way the app writes it: fenced."""
    return Message(role="user", contents=[fence(text)])


def _project(*messages: Message):
    return to_transcript(CONFIG, list(messages))


# --- the happy path ----------------------------------------------------------


def test_pairs_questions_with_the_answers_that_follow():
    transcript = _project(
        _q("Design pricing for first-time hosts."),
        _a("I would segment by listing count."),
        _q("Why listing count and not tenure?"),
        _a("Because it proxies for confidence."),
    )

    assert [turn.question for turn in transcript.turns] == [
        "Design pricing for first-time hosts.",
        "Why listing count and not tenure?",
    ]
    assert [turn.answer for turn in transcript.turns] == [
        "I would segment by listing count.",
        "Because it proxies for confidence.",
    ]
    assert transcript.answered_turns == 2


def test_indexes_are_sequential_from_zero():
    transcript = _project(_q("one"), _a("a"), _q("two"), _a("b"), _q("three"))
    assert [turn.index for turn in transcript.turns] == [0, 1, 2]


def test_answers_come_back_unfenced():
    # The fence is a prompt-level construct; no consumer of the transcript wants
    # to see the markers, and the evaluator's document re-applies them itself.
    transcript = _project(_q("q"), _a("my answer"))

    assert transcript.turns[0].answer == "my answer"
    assert "CANDIDATE_ANSWER" not in (transcript.turns[0].answer or "")


def test_config_is_carried_through():
    transcript = to_transcript(InterviewConfig(followup_budget=6), [])
    assert transcript.config.followup_budget == 6
    assert transcript.config.total_questions == 7


# --- streamed text ------------------------------------------------------------


def test_streamed_deltas_are_concatenated_without_inserted_spaces():
    """The case that motivated not using Message.text.

    A streamed reply can be stored as one content part per delta. The framework's
    own accessor joins parts with a space, which would turn "Hel" + "lo" into
    "Hel lo" -- visible corruption on every question.
    """
    transcript = _project(_q("What ", "happens ", "in a ", "thin market?"))
    assert transcript.turns[0].question == "What happens in a thin market?"


def test_single_part_messages_are_unaffected():
    transcript = _project(_q("What happens in a thin market?"))
    assert transcript.turns[0].question == "What happens in a thin market?"


# --- ragged cases -------------------------------------------------------------


def test_empty_history_yields_an_empty_transcript():
    transcript = _project()

    assert transcript.turns == []
    assert transcript.answered_turns == 0
    assert transcript.awaiting_answer is False
    assert transcript.is_gradable is False


def test_trailing_question_is_left_open():
    transcript = _project(_q("q1"), _a("a1"), _q("q2"))

    assert len(transcript.turns) == 2
    assert transcript.turns[-1].answer is None
    assert transcript.awaiting_answer is True


def test_consecutive_questions_merge_into_one_turn():
    """E-49: the alternative would shift every later answer onto the wrong question.

    A retry that partially persisted could leave two questions in a row.
    Concatenating keeps the pairing aligned; treating the second as a new turn
    would mispair every subsequent answer and so corrupt every quote in the
    report.
    """
    transcript = _project(_q("first ask"), _q("second ask"), _a("one answer"))

    assert len(transcript.turns) == 1
    assert "first ask" in transcript.turns[0].question
    assert "second ask" in transcript.turns[0].question
    assert transcript.turns[0].answer == "one answer"


def test_answer_before_any_question_is_ignored():
    # E-50: impossible through the UI, so not trusted either.
    transcript = _project(_a("orphan"), _q("q1"), _a("a1"))

    assert len(transcript.turns) == 1
    assert transcript.turns[0].question == "q1"
    assert transcript.turns[0].answer == "a1"


def test_consecutive_answers_merge_rather_than_being_dropped():
    transcript = _project(_q("q"), _a("first half"), _a("second half"))

    assert len(transcript.turns) == 1
    assert "first half" in (transcript.turns[0].answer or "")
    assert "second half" in (transcript.turns[0].answer or "")


def test_blank_messages_are_skipped():
    # An empty assistant message would otherwise open a turn with no question and
    # fail Turn's min_length validation.
    transcript = _project(_q("q1"), _q("   "), _a("a1"))

    assert len(transcript.turns) == 1
    assert transcript.turns[0].question == "q1"
    assert transcript.turns[0].answer == "a1"


def test_non_conversation_roles_are_ignored():
    transcript = _project(
        Message(role="system", contents=["you are an interviewer"]),
        _q("q1"),
        Message(role="tool", contents=["some tool output"]),
        _a("a1"),
    )

    assert len(transcript.turns) == 1
    assert transcript.turns[0].question == "q1"
    assert transcript.turns[0].answer == "a1"


def test_unfenced_answer_survives_the_projection():
    # Tolerance for history written before fencing, or by a different path.
    transcript = to_transcript(CONFIG, [_q("q"), Message(role="user", contents=["plain answer"])])
    assert transcript.turns[0].answer == "plain answer"


def test_multiline_answer_keeps_its_structure():
    original = "First point.\n\nSecond point."
    transcript = _project(_q("q"), _a(original))
    assert transcript.turns[0].answer == original


# --- interaction with the budget ---------------------------------------------


def test_completion_is_driven_by_answered_turns():
    full = [_q(f"q{i}") if i % 2 == 0 else _a(f"a{i}") for i in range(10)]
    transcript = to_transcript(InterviewConfig(followup_budget=4), full)

    assert transcript.answered_turns == 5
    assert transcript.is_complete is True


def test_abandoned_final_question_does_not_complete_the_interview():
    messages: list[Message] = []
    for i in range(4):
        messages += [_q(f"q{i}"), _a(f"a{i}")]
    messages.append(_q("q4 never answered"))

    transcript = to_transcript(InterviewConfig(followup_budget=4), messages)

    assert transcript.answered_turns == 4
    assert transcript.is_complete is False
    assert transcript.awaiting_answer is True
