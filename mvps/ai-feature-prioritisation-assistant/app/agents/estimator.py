"""The estimator: one Microsoft Agent Framework agent, one call per backlog.

There is exactly one model call in this application, and this module is it. The
whole backlog goes out in a single request so the features are calibrated
against each other, and what comes back is a factor set per feature — never a
score, never a rank.

Parsing is written to salvage rather than to fail. A twenty-five item reply is a
large JSON object produced by a model under a token cap, and the realistic
failure is not "no JSON at all" but "twenty-three good entries and one where
Impact came back as the string 'high'". Rejecting the whole reply for that would
throw away a working estimate and charge the user another call, so entries are
validated one at a time: the good ones are kept, the bad ones are dropped, and
the features they belonged to are reported as unestimated by the scorer.

Every entry point comes in two flavours. The ``a``-prefixed coroutine is the real
implementation; its sync twin wraps it via :mod:`app.services.async_bridge` and
is what the Streamlit UI calls.
"""

import json
import re
from typing import Any

from agent_framework import Agent

from app.agents.client import build_options, get_chat_client, structured_response_format
from app.agents.prompts import build_estimator_instructions, format_backlog
from app.core.exceptions import EstimateParseError
from app.core.logging import get_logger
from app.models.schemas import BacklogEstimate, BacklogInput, FeatureEstimate
from app.services.async_bridge import run_sync
from app.services.guardrails import sanitize_markdown

logger = get_logger(__name__)

#: Last-resort match for a JSON object anywhere in a reply, used when a model
#: wraps the object in a code fence or a sentence.
_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

#: Output budget per feature, plus a fixed allowance for the wrapper object.
#: One entry is four rationales, an assumptions list, and four numbers -- about
#: 180 tokens of JSON in practice, with headroom here for a wordy model.
#:
#: The budget is computed from the backlog rather than fixed, because providers
#: charge ``max_tokens`` against the per-minute rate limit *as requested*,
#: whether or not the reply uses it. A flat 8k cap made a three-feature backlog
#: cost the same as a twenty-five feature one and pushed a live free-tier run
#: straight into ``413 Request too large`` (limit 8,000/min, requested 10,206).
#: ``MAX_TOKENS`` stays as the ceiling.
#:
#: The base is large because a reasoning model spends a *fixed* chunk of the
#: output budget thinking before it emits a character of JSON, and that chunk
#: does not shrink with the backlog. Sized at 600 it worked for twelve features
#: and failed for nine, with the provider returning
#: ``400 json_validate_failed`` and an empty ``failed_generation`` -- reasoning
#: had consumed the whole allowance. The symptom names JSON and the cause is
#: arithmetic, so it is worth the two constants being explicit.
TOKENS_PER_FEATURE = 220
BASE_OUTPUT_TOKENS = 2000


def _output_budget(feature_count: int) -> int:
    """Size the reply cap for this backlog, never above the configured ceiling."""
    from app.core.config import get_settings

    wanted = BASE_OUTPUT_TOKENS + TOKENS_PER_FEATURE * feature_count
    return min(wanted, get_settings().max_tokens)


#: Fields carrying model-written prose, sanitised before they reach a render or
#: an export.
_PROSE_FIELDS = (
    "reach_rationale",
    "impact_rationale",
    "confidence_rationale",
    "effort_rationale",
)


def _build_estimator_agent(backlog: BacklogInput) -> Agent:
    """Create the estimator agent for one backlog."""
    return get_chat_client().as_agent(
        name="factor-estimator",
        instructions=build_estimator_instructions(backlog),
    )


def _payload_from(response: Any) -> dict[str, Any]:
    """Find the JSON object in whatever the run produced.

    Three strategies, cheapest first. ``response.value`` is already populated by
    the framework -- as the model under ``json_schema`` mode, as a plain dict
    under ``json_object`` -- but neither is guaranteed: some providers get no
    ``response_format`` at all, and some models still wrap the object in a fence.
    Falling through to the outermost ``{...}`` costs nothing and saves a retry
    against a rate-limited free-tier model.

    Raises:
        EstimateParseError: If no strategy found a JSON object.
    """
    value = response.value
    if isinstance(value, BacklogEstimate):
        return value.model_dump()
    if isinstance(value, dict):
        return value

    text = (response.text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        logger.debug("Estimator reply was not bare JSON; trying the embedded-object fallback")

    match = _JSON_BLOCK_PATTERN.search(text)
    if match is None:
        raise EstimateParseError(f"Estimator reply was not JSON: {text[:200]}")
    try:
        return json.loads(match.group(0))
    except ValueError as exc:
        raise EstimateParseError(f"Estimator reply was not valid JSON: {text[:200]}") from exc


def _sanitize(estimate: FeatureEstimate) -> FeatureEstimate:
    """Defang model-written prose before it can reach a render or an export."""
    return estimate.model_copy(
        update={
            **{field: sanitize_markdown(getattr(estimate, field)) for field in _PROSE_FIELDS},
            "assumptions": [sanitize_markdown(item) for item in estimate.assumptions],
        }
    )


def _coerce_estimate(response: Any) -> BacklogEstimate:
    """Turn one estimation reply into a validated :class:`BacklogEstimate`.

    Args:
        response: The ``AgentResponse`` returned by the estimation run.

    Returns:
        The entries that validated, sanitised. Possibly fewer than were asked
        for -- the scorer reconciles that against the ids it sent and reports
        the gap.

    Raises:
        EstimateParseError: If the reply held no JSON object, or held one with
            no usable entry in it at all.
    """
    payload = _payload_from(response)
    raw_entries = payload.get("estimates")
    if not isinstance(raw_entries, list):
        raise EstimateParseError(f"Estimator reply had no 'estimates' list: {str(payload)[:200]}")

    estimates: list[FeatureEstimate] = []
    for entry in raw_entries:
        try:
            estimates.append(_sanitize(FeatureEstimate.model_validate(entry)))
        except ValueError as exc:
            # One malformed row does not invalidate the other twenty-four. The
            # feature it belonged to shows up as unestimated instead.
            logger.warning("Dropping unusable estimate entry: %s", exc)

    if not estimates:
        raise EstimateParseError(
            f"No usable estimates in the reply ({len(raw_entries)} entries, none valid)."
        )

    unit = sanitize_markdown(str(payload.get("reach_unit") or "")).strip()
    return BacklogEstimate(
        # The default lives on the schema; an absent or blank unit falls back to
        # it rather than putting an empty string in a column header.
        **({"reach_unit": unit[:30]} if unit else {}),
        estimates=estimates,
    )


async def aestimate_backlog(backlog: BacklogInput) -> BacklogEstimate:
    """Estimate the RICE factors for a whole backlog in one call.

    Args:
        backlog: The parsed backlog.

    Returns:
        One factor set per feature the model returned a usable entry for.

    Raises:
        EstimateParseError: If the reply could not be parsed at all.
    """
    logger.info(
        "Estimating %d feature(s); product context: %s",
        len(backlog.features),
        "yes" if backlog.product_context else "no",
    )
    response = await _build_estimator_agent(backlog).run(
        format_backlog(backlog),
        options=build_options(
            response_format=structured_response_format(BacklogEstimate),
            max_tokens=_output_budget(len(backlog.features)),
        ),
    )
    estimate = _coerce_estimate(response)
    logger.info("Estimator returned %d usable entries", len(estimate.estimates))
    return estimate


def estimate_backlog(backlog: BacklogInput) -> BacklogEstimate:
    """Blocking twin of :func:`aestimate_backlog`, for the Streamlit UI."""
    return run_sync(aestimate_backlog(backlog))
