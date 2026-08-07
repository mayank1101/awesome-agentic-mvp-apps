"""Tests for the synthesis call.

Covers E-34 (invalid JSON and the repair retry), E-35 (429 split into rate limit
and daily cap), E-36 (retired model id), E-38 (truncated reply), and the prompt
properties that make SC-1 structural — the model is never shown a URL.
"""

import json
from typing import Any

import pytest

from app.agents import synthesizer
from app.agents.prompts import build_user_message, format_evidence, system_instructions
from app.core.exceptions import (
    ModelUnavailable,
    ProviderQuotaExhausted,
    ProviderRequestTooLarge,
    SynthesisError,
)
from app.models.schemas import (
    NOT_FOUND,
    CompanyIdentity,
    EvidenceItem,
    SectionEvidence,
    SectionKey,
    SynthesisResult,
)


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


def _identity() -> CompanyIdentity:
    return CompanyIdentity(name="Acme", domain="acme.com")


def _evidence() -> tuple[SectionEvidence, ...]:
    return tuple(
        SectionEvidence(
            section=section,
            query=f"{section.value} query",
            items=(
                EvidenceItem(
                    id=f"{section.value}-1",
                    title="Acme pricing",
                    url="https://acme.com/pricing",
                    content="Team plan is $10 per user per month.",
                ),
            ),
        )
        for section in SectionKey
    )


def _valid_reply(**overrides: str) -> str:
    payload = {key.value: f"{key.value} body" for key in SectionKey}
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def patched_agent(monkeypatch: pytest.MonkeyPatch):
    def install(replies: list[Any]) -> FakeAgent:
        agent = FakeAgent(replies)
        monkeypatch.setattr(synthesizer, "_build_agent", lambda: agent)
        monkeypatch.setattr(synthesizer, "build_options", lambda **_: {})
        return agent

    return install


# --------------------------------------------------------------------------- #
# The prompt (SC-1 by construction)
# --------------------------------------------------------------------------- #


def test_evidence_block_never_shows_the_model_a_url():
    # The whole basis of "zero invented links": a model that never sees a URL
    # cannot reproduce one, so the allowlist has nothing to catch.
    block = format_evidence(_evidence())

    assert "https://acme.com/pricing" not in block
    assert "pricing-1" in block


def test_evidence_block_labels_empty_sections_with_their_query():
    empty = (SectionEvidence(section=SectionKey.PRICING, query="acme pricing plans"),)
    assert "acme pricing plans" in format_evidence(empty)


def test_system_prompt_presents_not_found_as_correct():
    instructions = system_instructions()

    assert NOT_FOUND in instructions
    # Shown as a worked example, not merely permitted -- a model that has seen the
    # phrase used approvingly reaches for it.
    assert "correct" in instructions.lower()


def test_user_message_names_the_subject_and_all_six_sections():
    message = build_user_message(_identity(), _evidence())

    assert "Acme (acme.com)" in message
    for section in SectionKey:
        assert section.value in message


# --------------------------------------------------------------------------- #
# Happy path and reply shapes
# --------------------------------------------------------------------------- #


def test_valid_json_is_parsed(patched_agent):
    patched_agent([_valid_reply()])
    result, failed = synthesizer.synthesize(_identity(), _evidence())

    assert isinstance(result, SynthesisResult)
    assert failed is False
    assert result.section(SectionKey.PRICING) == "pricing body"


def test_code_fenced_json_is_unwrapped(patched_agent):
    # Models add these even when told not to; unwrapping is cheaper than a retry.
    patched_agent([f"```json\n{_valid_reply()}\n```"])
    result, failed = synthesizer.synthesize(_identity(), _evidence())

    assert failed is False
    assert result.section(SectionKey.SNAPSHOT) == "snapshot body"


def test_json_with_prose_around_it_is_recovered(patched_agent):
    patched_agent([f"Here is the brief:\n{_valid_reply()}\nHope that helps."])
    _, failed = synthesizer.synthesize(_identity(), _evidence())

    assert failed is False


def test_streamed_parts_are_joined_without_a_separator(patched_agent):
    # The framework's own text accessor joins parts with a space, which corrupts
    # JSON split across them.
    reply = _valid_reply()
    patched_agent([[reply[:20], reply[20:]]])

    _, failed = synthesizer.synthesize(_identity(), _evidence())
    assert failed is False


def test_blank_section_falls_back_to_not_found(patched_agent):
    patched_agent([_valid_reply(pricing="   ")])
    result, _ = synthesizer.synthesize(_identity(), _evidence())

    assert result.section(SectionKey.PRICING) == NOT_FOUND


# --------------------------------------------------------------------------- #
# Repair retry and fallback (E-34, E-38)
# --------------------------------------------------------------------------- #


def test_invalid_json_is_retried_once_then_succeeds(patched_agent):
    agent = patched_agent(["not json at all", _valid_reply()])
    _, failed = synthesizer.synthesize(_identity(), _evidence())

    assert failed is False
    assert len(agent.calls) == 2
    assert "was not valid JSON" in agent.calls[1]


def test_truncated_json_is_retried(patched_agent):
    # E-38: a reply cut off at max_tokens is indistinguishable from bad JSON at
    # the parse boundary, and is handled the same way.
    agent = patched_agent([_valid_reply()[:60], _valid_reply()])
    _, failed = synthesizer.synthesize(_identity(), _evidence())

    assert failed is False
    assert len(agent.calls) == 2


def test_missing_keys_are_retried(patched_agent):
    patched_agent([json.dumps({"snapshot": "only one key"}), _valid_reply()])
    _, failed = synthesizer.synthesize(_identity(), _evidence())

    assert failed is False


def test_extra_keys_are_rejected(patched_agent):
    # extra="forbid": a reply with extra keys did not follow the schema, and this
    # is the layer with a retry available.
    payload = json.loads(_valid_reply())
    payload["editorial_opinion"] = "buy their stock"
    patched_agent([json.dumps(payload), _valid_reply()])

    _, failed = synthesizer.synthesize(_identity(), _evidence())
    assert failed is False


def test_two_bad_replies_fall_back_to_a_valid_report(patched_agent):
    # A structurally valid report whose sources are real beats an error page.
    agent = patched_agent(["nope", "still nope"])
    result, failed = synthesizer.synthesize(_identity(), _evidence())

    assert failed is True
    assert len(agent.calls) == 2
    assert all(result.section(section) == NOT_FOUND for section in SectionKey)


# --------------------------------------------------------------------------- #
# Provider failures (E-35, E-36)
# --------------------------------------------------------------------------- #


def test_per_minute_rate_limit_is_retried(monkeypatch: pytest.MonkeyPatch, patched_agent):
    agent = patched_agent([RuntimeError("429 rate limit reached for requests"), _valid_reply()])
    monkeypatch.setattr(synthesizer, "RATE_LIMIT_BACKOFF_SECONDS", 0.0)

    _, failed = synthesizer.synthesize(_identity(), _evidence())

    assert failed is False
    assert len(agent.calls) == 2


def test_daily_cap_is_not_retried(patched_agent):
    # E-35: same status code, different situation. Retrying a daily cap spends a
    # minute to fail identically.
    agent = patched_agent([RuntimeError("429 limit reached: tokens per day (TPD)")])

    with pytest.raises(ProviderQuotaExhausted):
        synthesizer.synthesize(_identity(), _evidence())

    assert len(agent.calls) == 1


def test_retired_model_id_names_the_model(monkeypatch: pytest.MonkeyPatch, patched_agent):
    # The configured id is pinned here rather than read from the ambient
    # settings: a developer's own .env would otherwise decide what this asserts.
    from app.core.config import Settings

    monkeypatch.setattr(
        synthesizer, "get_settings", lambda: Settings(_env_file=None, model_name="some-retired-id")
    )
    patched_agent([RuntimeError("model_not_found: it has been decommissioned")])

    with pytest.raises(ModelUnavailable) as caught:
        synthesizer.synthesize(_identity(), _evidence())

    assert "some-retired-id" in str(caught.value)


def test_unknown_provider_error_is_wrapped(patched_agent):
    patched_agent([RuntimeError("connection reset"), RuntimeError("connection reset")])

    with pytest.raises(SynthesisError):
        synthesizer.synthesize(_identity(), _evidence())


def test_token_exhaustion_names_the_real_cause(monkeypatch: pytest.MonkeyPatch, patched_agent):
    # Seen live: gpt-oss-120b is a reasoning model, so it spent the whole output
    # allowance thinking and returned json_validate_failed without emitting any
    # JSON. The generic message points the reader at their prompt; the fix is the
    # model id or the cap (E-37).
    from app.core.config import Settings

    monkeypatch.setattr(
        synthesizer,
        "get_settings",
        lambda: Settings(_env_file=None, model_name="openai/gpt-oss-120b", max_tokens=2500),
    )
    error = RuntimeError(
        "Error code: 400 - {'code': 'json_validate_failed', 'failed_generation': "
        "'max completion tokens reached before generating a valid document'}"
    )
    patched_agent([error, error])

    with pytest.raises(SynthesisError) as caught:
        synthesizer.synthesize(_identity(), _evidence())

    message = str(caught.value)
    assert "gpt-oss-120b" in message
    assert "MAX_TOKENS" in message and "2500" in message


def test_over_limit_request_is_retried_with_less_evidence(patched_agent):
    # Seen live twice: free-tier TPM ceilings count the output reservation as
    # well as the prompt, so a request can be too large before it is ever slow.
    # Waiting does not help; sending less does.
    too_large = RuntimeError(
        "Error code: 413 - Request too large for model on tokens per minute (TPM): Limit 6000"
    )
    agent = patched_agent([too_large, _valid_reply()])

    bulky = tuple(
        section.model_copy(
            update={
                "items": tuple(
                    item.model_copy(update={"content": "word " * 400}) for item in section.items
                )
            }
        )
        for section in _evidence()
    )
    _, failed = synthesizer.synthesize(_identity(), bulky)

    assert failed is False
    assert len(agent.calls) == 2
    # The retry is a smaller prompt, not the same one sent again.
    assert len(agent.calls[1]) < len(agent.calls[0])


def test_over_limit_twice_is_not_swallowed(patched_agent):
    too_large = RuntimeError("Error code: 413 - Request too large on tokens per minute (TPM)")
    patched_agent([too_large, too_large])

    with pytest.raises(ProviderRequestTooLarge):
        synthesizer.synthesize(_identity(), _evidence())


def test_a_model_without_structured_output_is_retried_in_prose(patched_agent):
    # Seen live on OpenRouter: structured output is a model capability, not a
    # provider one. The prompt asks for JSON in words too, so dropping the flag
    # costs the provider-side guarantee rather than the feature.
    unsupported = RuntimeError(
        "Error code: 400 - model: some/model does not support feature: structured-outputs"
    )
    agent = patched_agent([unsupported, _valid_reply()])

    _, failed = synthesizer.synthesize(_identity(), _evidence())

    assert failed is False
    assert len(agent.calls) == 2


def test_non_text_parts_are_skipped(monkeypatch: pytest.MonkeyPatch):
    # Seen live: a reasoning block arrives as a content part whose `text` is
    # None, and joining it raises a TypeError that looks nothing like the
    # provider difference it actually is.
    agent = FakeAgent([[None, _valid_reply()]])
    monkeypatch.setattr(synthesizer, "_build_agent", lambda: agent)
    monkeypatch.setattr(synthesizer, "build_options", lambda **_: {})

    _, failed = synthesizer.synthesize(_identity(), _evidence())
    assert failed is False


def test_a_nested_section_is_flattened_not_rejected(patched_agent):
    # Seen live: asked for six strings, the model returned strengths_weaknesses
    # as {"strengths": [...], "weaknesses": [...]}. That is the right answer in
    # the wrong shape, and rejecting it throws away a good brief.
    payload = json.loads(_valid_reply())
    payload["strengths_weaknesses"] = {
        "strengths": ["Fast builds", "Good DX"],
        "weaknesses": ["Pricing at scale"],
    }
    patched_agent([json.dumps(payload)])

    result, failed = synthesizer.synthesize(_identity(), _evidence())
    body = result.section(SectionKey.STRENGTHS_WEAKNESSES)

    assert failed is False
    assert "**Strengths**" in body
    assert "- Fast builds" in body
    assert "- Pricing at scale" in body


def test_a_list_section_becomes_bullets(patched_agent):
    payload = json.loads(_valid_reply())
    payload["recent_moves"] = ["2026-01-02: raised a round", "2026-03-04: shipped an API"]
    patched_agent([json.dumps(payload)])

    result, _ = synthesizer.synthesize(_identity(), _evidence())
    assert result.section(SectionKey.RECENT_MOVES).startswith("- 2026-01-02")
