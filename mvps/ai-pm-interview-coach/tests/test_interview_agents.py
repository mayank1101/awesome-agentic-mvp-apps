"""The two agents, driven against fakes.

No network. The chat client is replaced with a stub whose replies each test
chooses, which is what makes the interesting paths -- a fenced JSON reply, a
schema near-miss that gets repaired, a stream that never finishes -- reachable at
all. A live model is exercised separately in `test_live_session.py`.
"""

import asyncio
import json

import pytest

from app.agents import interview_agents
from app.agents.interview_agents import (
    _coerce_report,
    _repair_request,
    _warn_on_unquotable_evidence,
    agenerate_report,
    new_agent_session,
)
from app.core.exceptions import EmptyInterviewError, ReportParseError
from app.models.schemas import (
    ALL_DIMENSIONS,
    Deduction,
    DimensionScore,
    FeedbackReport,
    InterviewConfig,
    RewrittenAnswer,
    Transcript,
    Turn,
)
from app.services.transcript import messages_from_state


def _valid_report_dict(**overrides) -> dict:
    report = {
        "headline": "Strong framing, weak on measurement.",
        "scores": [
            {
                "dimension": dimension,
                "score": 3,
                "justification": "held up under the probe",
                "evidence": "I would segment by listing count",
            }
            for dimension in ALL_DIMENSIONS
        ],
        "what_worked": ["named a specific segment"],
        "what_cost_points": [{"moment": "no metric", "stronger_move": "name a north star"}],
        "rewritten_answer": {
            "question": "How would you price it?",
            "rewrite": "Start from willingness to pay...",
            "why_better": "commits to a trade-off",
        },
        "next_focus": "metrics: state a counter-metric every time",
    }
    report.update(overrides)
    return report


def _report() -> FeedbackReport:
    return FeedbackReport(
        headline="h",
        scores=[
            DimensionScore(dimension=d, score=3, justification="j", evidence="listing count")
            for d in ALL_DIMENSIONS
        ],
        what_worked=[],
        what_cost_points=[Deduction(moment="m", stronger_move="s")],
        rewritten_answer=RewrittenAnswer(question="q", rewrite="r", why_better="w"),
        next_focus="metrics",
    )


def _transcript(answers: list[str | None], **config_overrides) -> Transcript:
    return Transcript(
        config=InterviewConfig(**config_overrides),
        turns=[
            Turn(index=i, question=f"question {i}", answer=answer)
            for i, answer in enumerate(answers)
        ],
    )


class FakeResponse:
    def __init__(self, *, value=None, text=None):
        self.value = value
        self.text = text


class FakeAgent:
    """Returns queued replies, one per run() call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def run(self, message, **kwargs):
        self.prompts.append(message)
        if not self.replies:
            raise AssertionError("FakeAgent ran out of queued replies")
        return self.replies.pop(0)


# --- session construction -----------------------------------------------------


def test_new_agent_session_has_the_state_dict_the_provider_writes_into():
    session = new_agent_session()

    # The provider keeps no instance state; the conversation lives in this dict,
    # under the provider's own source id, which is why the session is the thing
    # held in st.session_state across reruns.
    assert isinstance(session.state, dict)
    assert messages_from_state(session.state) == []


def test_sessions_are_distinct():
    assert new_agent_session().state is not new_agent_session().state


# --- report coercion ----------------------------------------------------------


def test_coerce_accepts_a_model_instance():
    report = _report()
    assert _coerce_report(FakeResponse(value=report)) is report


def test_coerce_accepts_a_dict():
    assert _coerce_report(FakeResponse(value=_valid_report_dict())).headline


def test_coerce_accepts_bare_json_text():
    text = json.dumps(_valid_report_dict())
    assert _coerce_report(FakeResponse(text=text)).headline


def test_coerce_accepts_json_inside_a_code_fence():
    # The case that earns the fallback its keep: free-tier models do this often
    # enough that a retry would otherwise be spent on a rate-limited model.
    text = f"Here is the report:\n```json\n{json.dumps(_valid_report_dict())}\n```\nHope it helps."
    assert _coerce_report(FakeResponse(text=text)).headline


def test_coerce_rejects_prose():
    with pytest.raises(ReportParseError, match="not JSON"):
        _coerce_report(FakeResponse(text="The candidate did quite well overall."))


def test_coerce_rejects_an_empty_reply():
    with pytest.raises(ReportParseError, match="empty"):
        _coerce_report(FakeResponse(text=""))


def test_coerce_rejects_malformed_json_in_a_fence():
    with pytest.raises(ReportParseError):
        _coerce_report(FakeResponse(text="```json\n{not valid at all,,}\n```"))


def test_schema_failure_is_raised_as_validation_not_parse_error():
    """The distinction the repair path depends on.

    Valid JSON that misses a dimension is a near-miss worth repairing; prose is
    not. Collapsing both into ReportParseError would lose the difference and spend
    a user-visible retry on something a correction message would have fixed.
    """
    from pydantic import ValidationError

    partial = _valid_report_dict()
    partial["scores"] = partial["scores"][:-1]

    with pytest.raises(ValidationError):
        _coerce_report(FakeResponse(text=json.dumps(partial)))


# --- the repair message -------------------------------------------------------


def test_repair_request_names_the_actual_problem():
    from pydantic import ValidationError

    partial = _valid_report_dict()
    partial["scores"] = partial["scores"][:-1]
    try:
        FeedbackReport.model_validate(partial)
    except ValidationError as exc:
        message = _repair_request(exc)

    assert "missing scores" in message
    assert "corrected JSON object" in message
    # Telling it to keep what it already justified is what makes this cheaper than
    # a cold regeneration.
    assert "Keep every" in message


# --- grading ------------------------------------------------------------------


@pytest.fixture
def fake_agent(monkeypatch):
    def _install(replies):
        agent = FakeAgent(replies)
        monkeypatch.setattr(interview_agents, "_build_evaluator", lambda transcript: agent)
        return agent

    return _install


def test_grading_returns_a_validated_report(fake_agent):
    fake_agent([FakeResponse(value=_valid_report_dict())])

    report = asyncio.run(agenerate_report(_transcript(["I would segment by listing count"])))

    assert len(report.scores) == len(ALL_DIMENSIONS)


def test_grading_refuses_an_empty_interview(fake_agent):
    # E-39: no call is made at all -- the evaluator would invent five scores from
    # no evidence.
    agent = fake_agent([])

    with pytest.raises(EmptyInterviewError):
        asyncio.run(agenerate_report(_transcript([None])))

    assert agent.prompts == []


def test_grading_refuses_a_transcript_with_no_turns(fake_agent):
    fake_agent([])
    with pytest.raises(EmptyInterviewError):
        asyncio.run(agenerate_report(_transcript([])))


def test_schema_near_miss_is_repaired_without_a_second_generation(fake_agent):
    partial = _valid_report_dict()
    partial["scores"] = partial["scores"][:-1]
    agent = fake_agent(
        [
            FakeResponse(text=json.dumps(partial)),
            FakeResponse(text=json.dumps(_valid_report_dict())),
        ]
    )

    report = asyncio.run(agenerate_report(_transcript(["an answer"])))

    assert len(report.scores) == len(ALL_DIMENSIONS)
    assert len(agent.prompts) == 2
    # The repair carries the original document plus the correction, so the model is
    # not re-grading from scratch.
    assert "corrected JSON object" in agent.prompts[1]
    assert "question 1" in agent.prompts[1]


def test_a_repair_that_also_fails_becomes_a_parse_error(fake_agent):
    partial = _valid_report_dict()
    partial["scores"] = partial["scores"][:-1]
    fake_agent([FakeResponse(text=json.dumps(partial))] * 2)

    with pytest.raises(ReportParseError, match="even after"):
        asyncio.run(agenerate_report(_transcript(["an answer"])))


def test_only_one_repair_is_attempted(fake_agent):
    partial = _valid_report_dict()
    partial["scores"] = partial["scores"][:-1]
    agent = fake_agent([FakeResponse(text=json.dumps(partial))] * 2)

    with pytest.raises(ReportParseError):
        asyncio.run(agenerate_report(_transcript(["an answer"])))

    assert len(agent.prompts) == 2


def test_prose_reply_is_not_repaired(fake_agent):
    # Repairing prose would waste a call: there is nothing to correct.
    agent = fake_agent([FakeResponse(text="The candidate did fine.")])

    with pytest.raises(ReportParseError):
        asyncio.run(agenerate_report(_transcript(["an answer"])))

    assert len(agent.prompts) == 1


def test_grading_sends_the_fenced_transcript(fake_agent):
    agent = fake_agent([FakeResponse(value=_valid_report_dict())])

    asyncio.run(agenerate_report(_transcript(["my exact words"])))

    assert "CANDIDATE_ANSWER" in agent.prompts[0]
    assert "my exact words" in agent.prompts[0]


# --- evidence checking --------------------------------------------------------


def test_unquotable_evidence_is_logged_not_rejected(caplog):
    # E-21, decided as log-only: a false "unverified" badge on a legitimate
    # near-quote damages trust more than the rare fabrication it would catch.
    report = _report()
    transcript = _transcript(["something entirely different"])

    with caplog.at_level("WARNING"):
        _warn_on_unquotable_evidence(report, transcript)

    assert any("may be paraphrased or fabricated" in record.message for record in caplog.records)


def test_quotable_evidence_logs_nothing(caplog):
    report = _report()
    transcript = _transcript(["I would segment by listing count and price from there"])

    with caplog.at_level("WARNING"):
        _warn_on_unquotable_evidence(report, transcript)

    assert not [r for r in caplog.records if "fabricated" in r.message]


def test_evidence_check_is_skipped_when_there_are_no_answers(caplog):
    with caplog.at_level("WARNING"):
        _warn_on_unquotable_evidence(_report(), _transcript([None]))

    assert not caplog.records


# --- streaming ----------------------------------------------------------------


class FakeSession:
    def __init__(self, config=None):
        self.config = config or InterviewConfig()
        self.agent_session = new_agent_session()
        self._turns: list[Turn] = []

    def transcript(self):
        return Transcript(config=self.config, turns=self._turns)


class FakeStreamAgent:
    """An agent whose run() returns an async iterator of updates."""

    def __init__(self, chunks, *, hang=False):
        self.chunks = chunks
        self.hang = hang

    def run(self, message, **kwargs):
        chunks, hang = self.chunks, self.hang

        class _Stream:
            def __aiter__(self):
                return self._gen()

            async def _gen(self):
                for chunk in chunks:
                    yield type("Update", (), {"text": chunk})()
                if hang:
                    await asyncio.sleep(3600)

        return _Stream()


def _install_stream_agent(monkeypatch, chunks, *, hang=False):
    agent = FakeStreamAgent(chunks, hang=hang)
    monkeypatch.setattr(interview_agents, "_build_interviewer", lambda config, n: agent)
    return agent


def test_opening_question_streams_its_chunks(monkeypatch):
    _install_stream_agent(monkeypatch, ["How ", "would ", "you price it?"])

    async def collect():
        return [chunk async for chunk in interview_agents.aopen_interview(FakeSession())]

    assert "".join(asyncio.run(collect())) == "How would you price it?"


def test_streamed_chunks_are_sanitised(monkeypatch):
    _install_stream_agent(monkeypatch, ["![x](https://evil.example/?d=1)"])

    async def collect():
        return [chunk async for chunk in interview_agents.aopen_interview(FakeSession())]

    assert "![x]" not in "".join(asyncio.run(collect()))


def test_empty_stream_is_a_failure_not_an_empty_question(monkeypatch):
    # E-15: an empty assistant message would open a turn with no question and
    # corrupt the projection's pairing.
    _install_stream_agent(monkeypatch, [])

    async def collect():
        return [chunk async for chunk in interview_agents.aopen_interview(FakeSession())]

    with pytest.raises(ReportParseError, match="empty question"):
        asyncio.run(collect())


def test_whitespace_only_stream_still_yields_but_counts_as_emitted(monkeypatch):
    # A model that streams only a space has produced something; the projection's
    # blank-message filter is what drops it later.
    _install_stream_agent(monkeypatch, [" "])

    async def collect():
        return [chunk async for chunk in interview_agents.aopen_interview(FakeSession())]

    assert asyncio.run(collect()) == [" "]


def test_stream_timeout_fires(monkeypatch):
    # E-16: without the bound, a stream that never finalises leaves the interview
    # screen waiting forever.
    from app.core.config import Settings

    settings = Settings(_env_file=None, stream_timeout_seconds=1)
    monkeypatch.setattr(interview_agents, "get_settings", lambda: settings)
    _install_stream_agent(monkeypatch, ["partial"], hang=True)

    async def collect():
        return [chunk async for chunk in interview_agents.aopen_interview(FakeSession())]

    with pytest.raises(TimeoutError):
        asyncio.run(collect())


def test_followup_numbers_the_question_from_the_transcript(monkeypatch):
    seen: list[int] = []

    def build(config, question_number):
        seen.append(question_number)
        return FakeStreamAgent(["next question?"])

    monkeypatch.setattr(interview_agents, "_build_interviewer", build)

    session = FakeSession()
    session._turns = [Turn(index=0, question="q0", answer="a0")]

    async def collect():
        return [chunk async for chunk in interview_agents.aask_followup(session, "my answer")]

    asyncio.run(collect())

    assert seen == [2]
