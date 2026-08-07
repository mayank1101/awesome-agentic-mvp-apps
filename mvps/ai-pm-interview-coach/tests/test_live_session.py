"""One real interview against a real model. Opt-in.

    pytest -m integration

Deselected by default, because it needs a provider credential and spends tokens.

This is stage 10b of the implementation plan, and it exists to answer the two
questions unit tests structurally cannot:

* does the framework's history actually accumulate across turns, so the
  interviewer sees what was said before;
* and do the anti-inflation measures work -- does a deliberately weak answer set
  score materially lower than a strong one?

It runs before any UI exists on purpose. Prompt tuning against a five-second test
is a different activity from prompt tuning through a browser.
"""

import pytest

from app.agents.interview_agents import (
    aask_followup,
    agenerate_report,
    aopen_interview,
    new_agent_session,
)
from app.models.schemas import InterviewConfig, Transcript
from app.services.transcript import messages_from_state, to_transcript

pytestmark = pytest.mark.integration


class LiveSession:
    """The minimum surface the agents need: config, agent_session, transcript()."""

    def __init__(self, config: InterviewConfig) -> None:
        self.config = config
        self.agent_session = new_agent_session()

    def transcript(self) -> Transcript:
        messages = messages_from_state(getattr(self.agent_session, "state", None))
        return to_transcript(self.config, messages)


async def _drain(stream) -> str:
    return "".join([chunk async for chunk in stream])


STRONG_ANSWERS = [
    (
        "I'd scope this to first-time hosts in thin markets -- cities with fewer than "
        "fifty active listings -- because they have no comparable set to anchor on and "
        "mispricing there costs them their first booking, which is when most of them "
        "churn. The pain is concrete: they either price at the platform median and sit "
        "empty, or undercut and anchor themselves low for the season."
    ),
    (
        "The assumption I'd test first is that thin-market hosts actually want a "
        "recommendation rather than a range. I'd rather ship a range with a stated "
        "confidence than a single number, because a wrong single number destroys trust "
        "in the whole feature and we only get one shot at a first-time host. The cost "
        "is that a range converts worse in usability testing."
    ),
    (
        "Success metric is the share of first-time hosts who get a booking within "
        "fourteen days of publishing, because that is the outcome the pricing is in "
        "service of. My counter-metric is realised nightly rate: if bookings go up "
        "while rate collapses, I have just taught hosts to undercut themselves. I'd "
        "want the first to move five points without the second dropping more than two."
    ),
]

WEAK_ANSWERS = [
    "I would improve the pricing. Users want good prices.",
    "It depends. I would look at the data and talk to users.",
    "Success would be more bookings and happier users overall.",
]


async def _run_interview(config: InterviewConfig, answers: list[str]) -> LiveSession:
    session = LiveSession(config)

    question = await _drain(aopen_interview(session))
    assert question.strip(), "the opening question was empty"

    for answer in answers:
        await _drain(aask_followup(session, answer))

    return session


@pytest.mark.asyncio
async def test_history_accumulates_across_turns():
    """The load-bearing framework assumption.

    If ``after_run`` does not persist into ``AgentSession.state["messages"]``, the
    interviewer is answering each question blind and the evaluator has nothing to
    grade -- and every unit test would still pass, because they all use fakes.
    """
    session = await _run_interview(InterviewConfig(followup_budget=2), STRONG_ANSWERS[:2])
    transcript = session.transcript()

    assert transcript.answered_turns == 2, f"expected 2 answered turns, got {transcript.turns}"
    assert len(transcript.turns) >= 3, "the follow-up questions were not persisted"

    # The candidate's exact words must survive, because the evaluator quotes them.
    assert "thin markets" in (transcript.turns[0].answer or "")


@pytest.mark.asyncio
async def test_followups_are_grounded_in_the_answer():
    """Success criterion 01 §6.2, checked loosely.

    A follow-up that would make sense against any answer is the failure mode. This
    cannot be asserted exactly, so it checks the weaker property that the
    interviewer's second question references the answer's specific subject matter.
    """
    session = await _run_interview(InterviewConfig(followup_budget=2), STRONG_ANSWERS[:1])
    turns = session.transcript().turns

    assert len(turns) >= 2
    followup = turns[1].question.lower()
    anchors = ("thin", "market", "listing", "churn", "median", "first-time", "host", "booking")
    assert any(anchor in followup for anchor in anchors), (
        f"follow-up looks generic rather than grounded: {turns[1].question!r}"
    )


@pytest.mark.asyncio
async def test_interviewer_does_not_answer_its_own_question():
    """Success criterion 01 §6.5, checked by proxy on length and shape."""
    session = await _run_interview(InterviewConfig(followup_budget=2), STRONG_ANSWERS[:1])
    turns = session.transcript().turns

    for turn in turns:
        # A question is a question. A model that starts coaching produces
        # paragraphs, and INTERVIEWER_MAX_TOKENS is what keeps this true.
        assert len(turn.question) < 1200, f"question reads like a lecture: {turn.question[:200]!r}"
        assert "?" in turn.question, f"turn had no question in it: {turn.question[:200]!r}"


@pytest.mark.asyncio
async def test_report_validates_against_the_schema():
    session = await _run_interview(InterviewConfig(followup_budget=2), STRONG_ANSWERS[:2])

    report = await agenerate_report(session.transcript())

    assert len(report.scores) == 5
    assert report.headline.strip()
    assert report.next_focus.strip()
    for score in report.scores:
        assert 1 <= score.score <= 4
        assert score.evidence.strip(), f"{score.dimension} has no evidence"


@pytest.mark.asyncio
async def test_scores_discriminate_between_a_strong_and_a_weak_interview():
    """The risk that would invalidate the whole design.

    A grader returning the same comfortable score for everything makes the report
    worthless (01 §6.3). Two interviews, same configuration, deliberately different
    answer quality.
    """
    config = InterviewConfig(interview_type="product_design", followup_budget=2)

    strong = await _run_interview(config, STRONG_ANSWERS[:2])
    weak = await _run_interview(config, WEAK_ANSWERS[:2])

    strong_report = await agenerate_report(strong.transcript())
    weak_report = await agenerate_report(weak.transcript())

    strong_total = sum(score.score for score in strong_report.scores)
    weak_total = sum(score.score for score in weak_report.scores)

    print(f"\nstrong: {strong_total}/20  {_summarise(strong_report)}")
    print(f"weak:   {weak_total}/20  {_summarise(weak_report)}")

    assert strong_total > weak_total, (
        f"scores did not discriminate: strong={strong_total} weak={weak_total}"
    )
    # A one-point gap would be noise. The weak set answers nothing concretely, so
    # the gap should be substantial.
    assert strong_total - weak_total >= 3, (
        f"gap too small to trust: strong={strong_total} weak={weak_total}"
    )


@pytest.mark.asyncio
async def test_weak_interview_uses_the_bottom_of_the_scale():
    """The other half of anti-inflation: 1s and 2s have to be reachable."""
    weak = await _run_interview(
        InterviewConfig(interview_type="execution_metrics", followup_budget=2),
        WEAK_ANSWERS[:2],
    )

    report = await agenerate_report(weak.transcript())

    print(f"\nweak (metrics interview): {_summarise(report)}")
    assert any(score.score <= 2 for score in report.scores), (
        "no dimension scored below the bar on answers that addressed nothing"
    )


def _summarise(report) -> str:
    return " ".join(f"{score.dimension}={score.score}" for score in report.scores)
