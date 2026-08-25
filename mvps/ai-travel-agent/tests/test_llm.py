"""Tests for JSON recovery, schema repair, and provider-error classification.

No network: `complete` is patched, and the error classifier is fed objects
shaped like the SDK's exceptions rather than the real ones, so the tests do not
break when the SDK renames a class.
"""

import pytest
from pydantic import BaseModel

from app.core.exceptions import (
    ModelQuotaExhausted,
    ModelRateLimited,
    ModelResponseInvalid,
    ModelUnavailable,
)
from app.services import llm


class _Toy(BaseModel):
    value: int


class _FakeProviderError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_parse_plain_json() -> None:
    assert llm.parse_json('{"value": 1}') == {"value": 1}


def test_parse_json_in_a_code_fence() -> None:
    assert llm.parse_json('```json\n{"value": 1}\n```') == {"value": 1}


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(ModelResponseInvalid):
        llm.parse_json("[1, 2, 3]")


def test_unparseable_reply_is_rejected() -> None:
    with pytest.raises(ModelResponseInvalid):
        llm.parse_json("I cannot help with that.")


def test_valid_reply_needs_no_repair_and_is_not_marked_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "complete", lambda **_: '{"value": 7}')
    result, degraded = llm.complete_model(system="s", user="u", schema=_Toy, max_tokens=10)

    assert result.value == 7
    assert degraded is False


def test_a_broken_reply_gets_one_repair_attempt_and_is_marked_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = iter(["not json at all", '{"value": 3}'])
    monkeypatch.setattr(llm, "complete", lambda **_: next(replies))

    result, degraded = llm.complete_model(system="s", user="u", schema=_Toy, max_tokens=10)

    assert result.value == 3
    assert degraded is True


def test_two_failures_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda **_: "still not json")

    with pytest.raises(ModelResponseInvalid):
        llm.complete_model(system="s", user="u", schema=_Toy, max_tokens=10)


def test_daily_quota_is_not_a_retryable_rate_limit() -> None:
    error = llm._classify(
        _FakeProviderError("Rate limit reached: 100000 tokens per day", status_code=429)
    )
    assert isinstance(error, ModelQuotaExhausted)


def test_per_minute_limit_is_retryable() -> None:
    error = llm._classify(_FakeProviderError("Rate limit reached for requests", status_code=429))
    assert isinstance(error, ModelRateLimited)


def test_retired_model_id_is_named() -> None:
    error = llm._classify(_FakeProviderError("model does not exist", status_code=404))
    assert isinstance(error, ModelUnavailable)


def test_missing_key_fails_before_any_call() -> None:
    from app.core.exceptions import ModelError

    with pytest.raises(ModelError, match="GROQ_API_KEY"):
        llm._client()
