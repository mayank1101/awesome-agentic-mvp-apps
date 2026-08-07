"""Tests for JSON recovery, schema repair, and provider-error classification.

No network: `complete` is patched, and the error classifier is fed objects
shaped like the SDK's exceptions rather than the real ones, so the tests do not
break when the SDK renames a class -- which is the same reason the classifier
reads status codes instead of catching by type.
"""

import pytest
from pydantic import BaseModel

from app.core.exceptions import (
    ModelError,
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


# --------------------------------------------------------------------------- #
# JSON recovery
# --------------------------------------------------------------------------- #


def test_parse_plain_json() -> None:
    assert llm.parse_json('{"value": 1}') == {"value": 1}


def test_parse_json_in_a_code_fence() -> None:
    assert llm.parse_json('```json\n{"value": 1}\n```') == {"value": 1}


def test_parse_json_after_a_preamble() -> None:
    assert llm.parse_json('Sure! Here it is:\n{"value": 1}') == {"value": 1}


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(ModelResponseInvalid):
        llm.parse_json("[1, 2, 3]")


def test_unparseable_reply_is_rejected() -> None:
    with pytest.raises(ModelResponseInvalid):
        llm.parse_json("I cannot help with that.")


# --------------------------------------------------------------------------- #
# Schema repair
# --------------------------------------------------------------------------- #


def test_valid_reply_needs_no_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_complete(**kwargs: object) -> str:
        calls.append(str(kwargs["user"]))
        return '{"value": 7}'

    monkeypatch.setattr(llm, "complete", fake_complete)
    result = llm.complete_model(system="s", user="u", schema=_Toy, max_tokens=10)

    assert result.value == 7
    assert len(calls) == 1


def test_a_broken_reply_gets_one_repair_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(["not json at all", '{"value": 3}'])
    prompts: list[str] = []

    def fake_complete(**kwargs: object) -> str:
        prompts.append(str(kwargs["user"]))
        return next(replies)

    monkeypatch.setattr(llm, "complete", fake_complete)
    result = llm.complete_model(system="s", user="u", schema=_Toy, max_tokens=10)

    assert result.value == 3
    assert len(prompts) == 2
    assert "could not be parsed" in prompts[1]


def test_two_failures_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "complete", lambda **_: "still not json")

    with pytest.raises(ModelResponseInvalid):
        llm.complete_model(system="s", user="u", schema=_Toy, max_tokens=10)


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #


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


def test_bad_key_is_reported_as_configuration() -> None:
    error = llm._classify(_FakeProviderError("Invalid API Key", status_code=401))
    assert isinstance(error, ModelError)
    assert "GROQ_API_KEY" in str(error)


def test_unknown_failures_still_classify() -> None:
    assert isinstance(llm._classify(RuntimeError("something odd")), ModelError)


def test_missing_key_fails_before_any_call() -> None:
    with pytest.raises(ModelError, match="GROQ_API_KEY"):
        llm._client()


def test_a_429_whose_request_exceeds_the_limit_is_a_size_problem() -> None:
    """Waiting cannot fix a request that is bigger than the per-minute ceiling."""
    from app.core.exceptions import ModelRequestTooLarge

    error = llm._classify(
        _FakeProviderError(
            "Rate limit reached for model `llama-3.1-8b-instant` on tokens per minute "
            "(TPM): Limit 6000, Used 1235, Requested 6444.",
            status_code=429,
        )
    )

    assert isinstance(error, ModelRequestTooLarge)


def test_a_429_that_merely_needs_a_wait_stays_retryable() -> None:
    error = llm._classify(
        _FakeProviderError(
            "Rate limit reached on tokens per minute (TPM): Limit 6000, Used 1235, "
            "Requested 5632. Please try again in 8.67s.",
            status_code=429,
        )
    )

    assert isinstance(error, ModelRateLimited)


def test_a_413_is_still_a_size_problem() -> None:
    from app.core.exceptions import ModelRequestTooLarge

    assert isinstance(
        llm._classify(_FakeProviderError("Request too large", status_code=413)),
        ModelRequestTooLarge,
    )
