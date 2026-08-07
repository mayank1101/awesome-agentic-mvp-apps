"""Tests for the gap-analysis call: retry, repair, and fallback behaviour."""

import json
from typing import Any

import pytest

from app.agents import analyzer
from app.agents.prompts import build_user_message, system_instructions
from app.core.exceptions import ModelUnavailable, ProviderQuotaExhausted
from app.models.schemas import Review
from app.services.evidence import format_evidence


class FakePart:
    def __init__(self, text: str | None):
        self.text = text


class FakeMessage:
    def __init__(self, parts: list[str]):
        self.contents = [FakePart(part) for part in parts]


class FakeResponse:
    def __init__(self, parts: list[str]):
        self.messages = [FakeMessage(parts)]


class FakeAgent:
    """Replays scripted replies or raises scripted errors, one per run."""

    def __init__(self, replies: list[Any]):
        self.replies = list(replies)
        self.calls: list[str] = []

    async def run(self, message: str, options: Any = None) -> FakeResponse:
        self.calls.append(message)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply if isinstance(reply, list) else [reply])


def _reviews(n: int = 3) -> list[Review]:
    return [
        Review(id=str(i), rating=1, title=f"Bad {i}", content=f"It broke, case {i}.", author="a")
        for i in range(n)
    ]


def _valid_reply(**overrides: Any) -> str:
    payload = {
        "gaps": [
            {
                "title": "Sync fails across devices",
                "description": "Multiple users report data not syncing.",
                "severity": "high",
                "review_ids": ["0", "1"],
            }
        ]
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def patched_agent(monkeypatch: pytest.MonkeyPatch):
    def install(replies: list[Any]) -> FakeAgent:
        agent = FakeAgent(replies)
        monkeypatch.setattr(analyzer, "_build_agent", lambda: agent)
        monkeypatch.setattr(analyzer, "build_options", lambda **_: {})
        return agent

    return install


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #


def test_evidence_block_labels_reviews_by_id_not_quote():
    from app.core.config import Settings

    reviews = _reviews(2)
    block = format_evidence(reviews, settings=Settings(_env_file=None, groq_api_key="gsk_test"))
    assert "[0]" in block and "[1]" in block


def test_system_prompt_forbids_quoting_review_text():
    assert "quotation marks" in system_instructions()


def test_user_message_includes_the_app_name():
    from app.core.config import Settings

    reviews = _reviews(1)
    block = format_evidence(reviews, settings=Settings(_env_file=None, groq_api_key="gsk_test"))
    message = build_user_message("Acme", reviews, block)
    assert "Acme" in message


# --------------------------------------------------------------------------- #
# Happy path and repair
# --------------------------------------------------------------------------- #


def test_a_valid_reply_produces_gaps(patched_agent):
    patched_agent([_valid_reply()])
    result, failed = analyzer.analyze("Acme", _reviews())

    assert failed is False
    assert len(result.gaps) == 1
    assert result.gaps[0].review_ids == ("0", "1")


def test_invalid_json_gets_one_repair_retry(patched_agent):
    agent = patched_agent(["not json at all", _valid_reply()])
    result, failed = analyzer.analyze("Acme", _reviews())

    assert failed is False
    assert len(agent.calls) == 2
    assert "previous reply was not valid JSON" in agent.calls[1]


def test_two_bad_replies_fall_back_to_empty_result(patched_agent):
    patched_agent(["still not json", "nope"])
    result, failed = analyzer.analyze("Acme", _reviews())

    assert failed is True
    assert result.gaps == ()


def test_a_code_fenced_reply_is_unwrapped(patched_agent):
    patched_agent([f"```json\n{_valid_reply()}\n```"])
    result, failed = analyzer.analyze("Acme", _reviews())
    assert failed is False
    assert len(result.gaps) == 1


def test_daily_quota_exhaustion_is_not_retried(patched_agent):
    patched_agent([Exception("429 tokens per day exceeded")])
    with pytest.raises(ProviderQuotaExhausted):
        analyzer.analyze("Acme", _reviews())


def test_retired_model_id_is_named_in_the_error(patched_agent):
    patched_agent([Exception("model_not_found: the model does not exist")])
    with pytest.raises(ModelUnavailable):
        analyzer.analyze("Acme", _reviews())
