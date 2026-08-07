"""The single gap-analysis call.

One agent, one run, one JSON reply validated into
:class:`~app.models.schemas.GapAnalysisResult`. Ported from the retry/fallback
structure in `ai-competitor-analyzer`'s synthesiser:

* **Invalid JSON** gets one repair retry, then falls back to an empty result
  (`analysis_failed=True`) rather than a crash -- the fetched reviews and
  computed stats still have value even when synthesis does not.
* **A 429 is two different situations behind one status code.** A per-minute
  rate limit is worth one backoff retry; an exhausted daily token cap is not,
  since retrying spends a minute to fail identically.
* **A retired model id** is reported with the id named, because free-tier ids
  are withdrawn without notice and the fix is a config change.

The sync wrapper exists because Streamlit's script body is synchronous; it
routes through the shared event loop rather than starting a new one per call.
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
    AnalysisError,
    ModelUnavailable,
    ProviderQuotaExhausted,
    ProviderRateLimited,
    ProviderRequestTooLarge,
    StructuredOutputUnsupported,
)
from app.core.logging import get_logger
from app.models.schemas import GapAnalysisResult, Review
from app.services.async_bridge import run_sync
from app.services.evidence import format_evidence

logger = get_logger(__name__)

RATE_LIMIT_BACKOFF_SECONDS = 8.0

_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

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


def _classify(exc: Exception) -> AnalysisError:
    """Turn a provider exception into the app's own error type."""
    text = str(exc).lower()

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

    if "json_validate_failed" in text or "max completion tokens reached" in text:
        settings = get_settings()
        return AnalysisError(
            f"{settings.model_name!r} ran out of output tokens before finishing the analysis. "
            "Reasoning models spend this budget on hidden reasoning first — either raise "
            f"MAX_TOKENS well above {settings.max_tokens} or set MODEL_NAME to a "
            "non-reasoning instruct model."
        )

    if "model_not_found" in text or "does not exist" in text or "decommissioned" in text:
        return ModelUnavailable(
            f"the configured model {get_settings().model_name!r} was rejected by the "
            "provider — free-tier model ids are retired without notice; try an alternate "
            "in .env.example"
        )

    return AnalysisError(f"gap analysis failed: {type(exc).__name__}")


def _build_agent() -> Agent:
    """Build the analysis agent over the cached chat client."""
    return get_chat_client().as_agent(name="review-gap-analyst", instructions=system_instructions())


async def _run_once(agent: Agent, message: str, *, as_json: bool = True) -> str:
    """Run the agent once and return its raw text.

    Raises:
        AnalysisError: Or a subclass, classified from the provider's error.
    """
    try:
        response = await agent.run(message, options=build_options(as_json=as_json))
    except Exception as exc:  # noqa: BLE001 - classified, then re-raised as ours
        raise _classify(exc) from exc

    # Concatenate content parts directly rather than using the framework's own
    # text accessor, which joins parts with a space and would corrupt JSON
    # split across parts. `part.text` is present but None on non-text parts (a
    # reasoning block, a tool-call stub), so the isinstance check is load-bearing.
    text = "".join(
        part.text
        for part in getattr(response.messages[-1], "contents", [])
        if isinstance(getattr(part, "text", None), str)
    )
    return text or str(response)


async def aanalyze(app_name: str, reviews: list[Review]) -> tuple[GapAnalysisResult, bool]:
    """Run gap analysis, with one repair retry and an empty-result fallback.

    Args:
        app_name: The app being analyzed.
        reviews: The critical reviews to analyze. Caller has already applied
            :func:`app.services.evidence.select_critical` and the minimum-
            signal gate (`InsufficientSignal`) before this is called.

    Returns:
        The result, and whether the fallback was used -- the UI states this
        plainly rather than presenting zero gaps as though nothing was found.

    Raises:
        ProviderQuotaExhausted: The daily budget is spent; no retry helps.
        ModelUnavailable: The configured model id was rejected.
        AnalysisError: The provider failed for another reason on both attempts.
    """
    agent = _build_agent()
    working_reviews = reviews
    message = build_user_message(app_name, working_reviews, format_evidence(working_reviews, settings=get_settings()))

    attempt = 0
    as_json = True
    downgraded = False
    shrunk = False
    waited = False

    while attempt < 2:
        text = message if attempt == 0 else f"{message}\n\n{REPAIR_INSTRUCTION}"

        try:
            raw = await _run_once(agent, text, as_json=as_json)
            return GapAnalysisResult.model_validate(_extract_json(raw)), False

        except StructuredOutputUnsupported:
            if downgraded:
                raise
            logger.warning("provider has no structured-output mode; asking in prose instead")
            downgraded, as_json = True, False

        except ProviderRequestTooLarge:
            if shrunk:
                raise
            logger.warning("prompt over the per-minute limit; retrying with half the reviews")
            shrunk = True
            working_reviews = working_reviews[: max(1, len(working_reviews) // 2)]
            message = build_user_message(
                app_name, working_reviews, format_evidence(working_reviews, settings=get_settings())
            )

        except ProviderRateLimited:
            if waited:
                raise
            logger.warning("rate limited; retrying once in %.0fs", RATE_LIMIT_BACKOFF_SECONDS)
            waited = True
            await asyncio.sleep(RATE_LIMIT_BACKOFF_SECONDS)

        except (ValueError, ValidationError) as exc:
            logger.warning("analysis reply was unusable (attempt %d): %s", attempt + 1, exc)
            attempt += 1

    logger.error("gap analysis failed after the repair retry; falling back to no gaps")
    return GapAnalysisResult.empty(), True


def analyze(app_name: str, reviews: list[Review]) -> tuple[GapAnalysisResult, bool]:
    """Synchronous wrapper for :func:`aanalyze`, for the Streamlit script."""
    return run_sync(aanalyze(app_name, reviews))
