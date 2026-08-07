"""The one place this app talks to a language model.

Groq, synchronously, through the official SDK. Synchronous on purpose: Streamlit
runs the script on its own thread and every call here is request/response with no
streaming, so there is nothing an event loop would buy that a blocking call and a
spinner do not -- and a background loop painting into Streamlit is a known trap.

Two things in this module are worth reading before changing it:

**Provider errors are classified by status code and message, not by SDK exception
class.** The classes move between SDK versions; `429` does not. More importantly,
Groq returns `429` for two situations that need opposite handling -- a per-minute
rate limit worth retrying, and an exhausted daily token budget where a retry is a
minute spent to fail identically.

**JSON is asked for twice**: through the provider's `response_format` and again
in words, in the prompt. Neither is a guarantee, so the reply is parsed and
validated against a Pydantic model, and a failure gets exactly one repair retry
that shows the model its own broken output. Two failures is a real failure.
"""

import json
import re
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.exceptions import (
    ModelError,
    ModelQuotaExhausted,
    ModelRateLimited,
    ModelRequestTooLarge,
    ModelResponseInvalid,
    ModelUnavailable,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Attempts for a call that fails with a retryable per-minute rate limit. Groq's
#: free-tier window is a minute wide, so three attempts with short waits could
#: exhaust themselves inside a single window and report failure to a user who
#: only had to wait.
_RATE_LIMIT_ATTEMPTS = 5

#: Base seconds for the backoff between those attempts, doubled each time. Used
#: only when the provider does not tell us how long to wait.
_BACKOFF_SECONDS = 4.0

#: Longest we will ever sit waiting on a provider's `retry-after`. Sized to the
#: per-minute window: waiting 60s can genuinely clear a TPM limit, and beyond
#: that the user is better served by an error than by a spinner.
_MAX_BACKOFF_SECONDS = 60.0

#: Groq puts the arithmetic in the 429 body: "Limit 6000, Used 1235, Requested
#: 5632". When the request alone exceeds the ceiling, waiting is pointless -- the
#: call has to get smaller -- so that case is reclassified as too-large even
#: though the status code says rate limit.
_LIMIT_FIGURES = re.compile(r"limit\s+(\d+).*?requested\s+(\d+)", re.IGNORECASE | re.DOTALL)

#: Markers of a *daily* budget in a 429 body, as opposed to a per-minute one.
_DAILY_QUOTA_MARKERS = ("per day", "tokens per day", "tpd", "requests per day", "rpd")

#: A fenced code block wrapping the JSON, which instruct models add despite being
#: asked not to.
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _client() -> Any:
    """Build a Groq client from settings.

    Imported lazily so that importing `app` -- which the offline test suite does
    constantly -- does not require the SDK to be installed or a key to be set.

    Returns:
        A configured `groq.Groq` instance.

    Raises:
        ModelError: The SDK is not installed, or no key is configured.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise ModelError("GROQ_API_KEY is not set.")

    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise ModelError("The `groq` package is not installed. `pip install groq`.") from exc

    return Groq(
        api_key=settings.groq_api_key,
        base_url=settings.groq_base_url,
        timeout=settings.model_timeout_seconds,
        max_retries=0,  # retries are this module's job, so they can be classified
    )


def _classify(exc: Exception) -> ModelError:
    """Map a provider exception onto this app's error vocabulary.

    Args:
        exc: Whatever the SDK raised.

    Returns:
        The matching :class:`~app.core.exceptions.ModelError` subclass, with a
        message written for the person reading the screen rather than for a log.
    """
    status = getattr(exc, "status_code", None)
    body = str(exc)
    lowered = body.lower()
    name = type(exc).__name__

    if status == 401 or "invalid api key" in lowered:
        return ModelError("The Groq API key was rejected. Check GROQ_API_KEY.")

    if status == 404 or "does not exist" in lowered or "decommissioned" in lowered:
        return ModelUnavailable(
            f"The model `{get_settings().model_name}` was rejected by Groq. Free-tier "
            "model ids get retired; set MODEL_NAME to a current one."
        )

    if status == 429:
        if any(marker in lowered for marker in _DAILY_QUOTA_MARKERS):
            return ModelQuotaExhausted(
                "This Groq key has spent its daily token budget for this model. "
                "Wait for the reset, switch MODEL_NAME, or use another key."
            )
        figures = _LIMIT_FIGURES.search(body)
        if figures and int(figures.group(2)) > int(figures.group(1)):
            return ModelRequestTooLarge(
                f"This request needs {figures.group(2)} tokens and the model's per-minute "
                f"limit is {figures.group(1)}. Retrying with a smaller request."
            )
        return ModelRateLimited("Groq is rate-limiting this key. Waiting and retrying.")

    if status == 413 or "too large" in lowered or "context_length" in lowered:
        return ModelRequestTooLarge(
            "The request was too large for this model's limits. Shorten the resume or "
            "the job description, or lower the token caps in settings."
        )

    if "Timeout" in name or "timed out" in lowered:
        return ModelError(
            f"Groq did not respond within {get_settings().model_timeout_seconds:.0f}s."
        )

    if "Connection" in name:
        return ModelError("Could not reach Groq. Check the network and try again.")

    return ModelError(f"The model call failed ({name}).")


def _retry_after_seconds(exc: Exception) -> float | None:
    """Read a `retry-after` hint from a provider exception, if it carries one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return min(float(raw), _MAX_BACKOFF_SECONDS) if raw else None
    except (TypeError, ValueError):
        return None


def complete(
    *,
    system: str,
    user: str,
    max_tokens: int,
    as_json: bool = True,
    temperature: float | None = None,
) -> str:
    """Make one chat completion call and return the reply text.

    Args:
        system: The system prompt.
        user: The user message, with untrusted documents already fenced.
        max_tokens: Output cap for this call.
        as_json: Whether to ask the provider for a JSON object.
        temperature: Override for the configured temperature.

    Returns:
        The assistant's message content, stripped.

    Raises:
        ModelError: Any provider failure, classified. Rate limits are retried
            first and only surface if the retries are also refused.
    """
    settings = get_settings()
    client = _client()

    request: dict[str, Any] = {
        "model": settings.model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": settings.model_temperature if temperature is None else temperature,
        "max_tokens": max_tokens,
    }
    if as_json:
        request["response_format"] = {"type": "json_object"}

    last: ModelError | None = None
    for attempt in range(1, _RATE_LIMIT_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            error = _classify(exc)
            if isinstance(error, ModelRateLimited) and attempt < _RATE_LIMIT_ATTEMPTS:
                delay = _retry_after_seconds(exc) or _BACKOFF_SECONDS * attempt
                logger.warning(
                    "Rate limited by Groq, retrying in %.0fs (attempt %d/%d)",
                    delay,
                    attempt,
                    _RATE_LIMIT_ATTEMPTS,
                )
                time.sleep(delay)
                last = error
                continue
            raise error from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            # An empty reply from a reasoning model means the token budget went
            # on hidden reasoning. Saying so beats "the model returned nothing".
            raise ModelResponseInvalid(
                "The model returned an empty reply. If MODEL_NAME is a reasoning model, "
                "its thinking tokens come out of the same budget as its answer -- raise "
                "the token caps or use an instruct model."
            )
        return content

    raise last or ModelError("The model call failed.")


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply.

    Tolerates the two things instruct models do despite instructions: wrapping
    the object in a fenced code block, and prefixing it with a sentence.

    Args:
        text: The raw reply.

    Returns:
        The decoded object.

    Raises:
        ModelResponseInvalid: No JSON object could be recovered.
    """
    candidate = _CODE_FENCE.sub("", text.strip())
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ModelResponseInvalid("The model's reply was not JSON.") from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelResponseInvalid("The model's reply was not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ModelResponseInvalid("The model returned JSON that was not an object.")
    return parsed


def complete_model(
    *,
    system: str,
    user: str,
    schema: type[T],
    max_tokens: int,
    temperature: float | None = None,
) -> T:
    """Call the model and validate its reply against a Pydantic model.

    One repair retry is allowed, and it shows the model its own output and the
    validation error. A second failure is reported rather than retried: past that
    point the model cannot produce the shape, and looping burns a free-tier
    budget to arrive at the same place.

    Args:
        system: The system prompt.
        user: The user message.
        schema: The model class the reply must satisfy.
        max_tokens: Output cap.
        temperature: Override for the configured temperature.

    Returns:
        A validated instance of `schema`.

    Raises:
        ModelResponseInvalid: The reply failed validation twice.
        ModelError: Any other provider failure.
    """
    raw = complete(system=system, user=user, max_tokens=max_tokens, temperature=temperature)

    try:
        return schema.model_validate(parse_json(raw))
    except (ModelResponseInvalid, ValidationError) as first:
        # Bound to a local: Python unbinds the `except` name at the end of the
        # clause, and the repair prompt below needs the reason.
        reason = str(first)
        logger.warning("First %s parse failed: %s", schema.__name__, type(first).__name__)

    repair = (
        f"{user}\n\n"
        "Your previous reply could not be parsed. It was:\n"
        f"{raw[:1500]}\n\n"
        f"The error was: {reason}\n\n"
        "Reply again with the same content as a single valid JSON object matching the "
        "schema exactly. No prose, no code fence, no trailing commas."
    )
    raw = complete(system=system, user=repair, max_tokens=max_tokens, temperature=0.0)

    try:
        return schema.model_validate(parse_json(raw))
    except (ModelResponseInvalid, ValidationError) as second:
        raise ModelResponseInvalid(
            f"The model could not produce a valid {schema.__name__} after a repair "
            f"attempt: {second}"
        ) from second
