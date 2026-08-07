"""The single synthesis call.

One agent, one run, one JSON reply validated into
:class:`~app.models.schemas.SynthesisResult`. Everything interesting here is
about what happens when that does not work:

* **Invalid JSON** gets one repair retry, then falls back to an all-"Not found"
  report (E-34, E-38). A structurally valid report whose sources are real beats
  an error page — the search half of the work still has value.
* **A 429 is two different situations behind one status code** (E-35). A
  per-minute rate limit is worth one backoff retry; an exhausted daily token cap
  is not, since retrying spends a minute to fail identically.
* **A retired model id** is reported with the id named, because free-tier ids are
  withdrawn without notice and the fix is a config change the user can make
  (E-36).

The sync wrappers exist because Streamlit's script body is synchronous; they
route through the shared event loop rather than starting a new one per call.
"""

import asyncio
import json
import re
from typing import Any

from agent_framework import Agent
from pydantic import ValidationError

from app.agents.client import build_options, get_chat_client
from app.agents.prompts import REPAIR_INSTRUCTION, build_user_message, system_instructions
from app.core.config import get_settings
from app.core.exceptions import (
    ModelUnavailable,
    ProviderQuotaExhausted,
    ProviderRateLimited,
    ProviderRequestTooLarge,
    StructuredOutputUnsupported,
    SynthesisError,
)
from app.core.logging import get_logger
from app.models.schemas import CompanyIdentity, SectionEvidence, SynthesisResult
from app.services.async_bridge import run_sync

logger = get_logger(__name__)

#: Backoff before the single retry of a per-minute rate limit. Long enough to
#: clear a per-minute window, short enough that the run still finishes inside its
#: deadline.
RATE_LIMIT_BACKOFF_SECONDS = 8.0

#: A fenced code block wrapping the JSON. Models add these even when asked not
#: to, and unwrapping one is cheaper than spending a retry on it.
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

#: The outermost JSON object in a reply that carried prose around it.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Args:
        text: The raw reply.

    Returns:
        The decoded object.

    Raises:
        ValueError: No JSON object could be decoded.
    """
    candidate = text.strip()

    fenced = _CODE_FENCE.match(candidate)
    if fenced:
        candidate = fenced.group(1)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT.search(candidate)
    if not match:
        raise ValueError("no JSON object in reply")
    return json.loads(match.group(0))


def _classify(exc: Exception) -> SynthesisError:
    """Turn a provider exception into the app's own error type.

    The provider signals a per-minute rate limit and an exhausted daily budget
    with the same status code, and the difference decides whether a retry is
    worth eight seconds or is guaranteed waste (E-35).

    Args:
        exc: Whatever the client raised.

    Returns:
        The typed error to raise in its place.
    """
    text = str(exc).lower()

    # 413: the prompt plus the output reservation exceeded the per-minute token
    # allowance. Distinct from a rate limit -- waiting does not help, only
    # sending less does, which is why it has its own class and its own retry.
    if "does not support feature: structured-outputs" in text or (
        "response_format" in text and "not support" in text
    ):
        return StructuredOutputUnsupported("this model has no structured-output mode")

    if "413" in text or "request too large" in text:
        return ProviderRequestTooLarge("the request exceeded the provider's per-minute token limit")

    if "429" in text or "rate limit" in text or "too many requests" in text:
        daily = any(
            marker in text
            for marker in ("per day", "daily", "tokens per day", "tpd", "requests per day", "rpd")
        )
        if daily:
            return ProviderQuotaExhausted(
                "the model provider's daily free-tier budget is exhausted — "
                "try again tomorrow or switch MODEL_NAME"
            )
        return ProviderRateLimited("the model provider is rate limiting this key")

    # The provider gave up mid-JSON because the token budget ran out. On Groq
    # this is what a *reasoning* model looks like from the outside: the hidden
    # reasoning is drawn from the same allowance as the visible reply, so the
    # budget can be spent before a single character of JSON is emitted (E-37).
    # Worth naming, because the generic message sends the reader to look at their
    # prompt when the fix is the model id or the cap.
    if "json_validate_failed" in text or "max completion tokens reached" in text:
        settings = get_settings()
        return SynthesisError(
            f"{settings.model_name!r} ran out of output tokens before finishing the brief. "
            "Reasoning models spend this budget on hidden reasoning first — either raise "
            f"MAX_TOKENS well above {settings.max_tokens} or set MODEL_NAME to a "
            "non-reasoning instruct model."
        )

    if "model_not_found" in text or "does not exist" in text or "decommissioned" in text:
        return ModelUnavailable(
            f"the configured model {get_settings().model_name!r} was rejected by the "
            "provider — free-tier model ids are retired without notice; try one of the "
            "alternates in .env.example"
        )

    return SynthesisError(f"synthesis failed: {type(exc).__name__}")


def _build_agent() -> Agent:
    """Build the synthesis agent over the cached chat client."""
    return get_chat_client().as_agent(
        name="competitor-analyst",
        instructions=system_instructions(),
    )


async def _run_once(agent: Agent, message: str, *, as_json: bool = True) -> str:
    """Run the agent once and return its raw text.

    Args:
        agent: The synthesis agent.
        message: The full user message.
        as_json: Whether to ask the provider for structured output. Turned off
            after a provider rejects the request for lacking that feature.

    Raises:
        SynthesisError: Or a subclass, classified from the provider's error.
    """
    try:
        response = await agent.run(message, options=build_options(as_json=as_json))
    except Exception as exc:  # noqa: BLE001 - classified, then re-raised as ours
        raise _classify(exc) from exc

    # Concatenate content parts directly. The framework's own text accessor joins
    # them with a space, which corrupts JSON split across parts.
    #
    # `part.text` is present but None on non-text parts -- a reasoning block, a
    # tool-call stub -- so the attribute check alone is not enough, and the join
    # raises a TypeError that looks nothing like the provider difference it is.
    text = "".join(
        part.text
        for part in getattr(response.messages[-1], "contents", [])
        if isinstance(getattr(part, "text", None), str)
    )
    return text or str(response)


async def asynthesize(
    identity: CompanyIdentity,
    sections: tuple[SectionEvidence, ...] | list[SectionEvidence],
) -> tuple[SynthesisResult, bool]:
    """Run synthesis, with one repair retry and a structural fallback.

    Args:
        identity: Which company is being profiled.
        sections: The packed evidence.

    Returns:
        The result, and whether the fallback was used — the UI says so plainly
        rather than presenting six "Not found" sections as though they were
        findings.

    Raises:
        ProviderQuotaExhausted: The daily budget is spent; no retry helps.
        ModelUnavailable: The configured model id was rejected.
        SynthesisError: The provider failed for another reason on both attempts.
    """
    agent = _build_agent()
    message = build_user_message(identity, sections)

    # Two kinds of retry, deliberately counted apart. A *repair* retry is spent
    # when the model produced something unusable, and there is one. An
    # *adaptation* — dropping structured output, halving the evidence, waiting
    # out a rate limit — is the app learning something about the provider, and
    # must not consume the repair budget: a model that has just been asked a
    # different way deserves its first real attempt.
    attempt = 0
    as_json = True
    downgraded = False
    shrunk = False
    waited = False

    while attempt < 2:
        text = message if attempt == 0 else f"{message}\n\n{REPAIR_INSTRUCTION}"

        try:
            raw = await _run_once(agent, text, as_json=as_json)
            return SynthesisResult.model_validate(_extract_json(raw)), False

        except StructuredOutputUnsupported:
            # Structured output is a *model* capability, not a provider one: two
            # models behind the same OpenRouter key disagree about it. The prompt
            # already asks for JSON in words and the parser already tolerates
            # prose around it, so dropping the flag costs the provider-side
            # guarantee rather than the output.
            if downgraded:
                raise
            logger.warning("provider has no structured-output mode; asking in prose instead")
            downgraded, as_json = True, False

        except ProviderRequestTooLarge:
            # Waiting does not help; sending less does. On a free tier this
            # ceiling is an ordinary condition, not an exceptional one.
            if shrunk:
                raise
            logger.warning("prompt over the per-minute limit; retrying with half the evidence")
            shrunk = True
            message = build_user_message(identity, _halve(sections))

        except ProviderRateLimited:
            if waited:
                raise
            logger.warning("rate limited; retrying once in %.0fs", RATE_LIMIT_BACKOFF_SECONDS)
            waited = True
            await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)

        except (ValueError, ValidationError) as exc:
            logger.warning("synthesis reply was unusable (attempt %d): %s", attempt + 1, exc)
            attempt += 1

    # Both real attempts produced something unusable. The sources are still real,
    # so ship the structure and say synthesis failed (E-34).
    logger.error("synthesis failed after the repair retry; falling back")
    return SynthesisResult.all_not_found(), True


def _halve(sections: tuple[SectionEvidence, ...] | list[SectionEvidence]):
    """Return the same sections with each snippet cut in half.

    Used only by the over-limit retry. Halving every section keeps the six-part
    shape intact rather than dropping whole sections, which would silently turn
    "we found nothing" and "we could not afford to look" into the same output.
    """
    return tuple(
        section.model_copy(
            update={
                "items": tuple(
                    item.model_copy(
                        update={"content": item.content[: max(200, len(item.content) // 2)]}
                    )
                    for item in section.items
                )
            }
        )
        for section in sections
    )


def synthesize(
    identity: CompanyIdentity,
    sections: tuple[SectionEvidence, ...] | list[SectionEvidence],
) -> tuple[SynthesisResult, bool]:
    """Synchronous wrapper for :func:`asynthesize`, for the Streamlit script."""
    return run_sync(asynthesize(identity, sections))
