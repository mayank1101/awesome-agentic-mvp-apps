"""The PRD pipeline, as two Microsoft Agent Framework agents.

Two agents make up the whole thing:

* **outline agent** -- one run, returns the title plus an ordered section list.
* **section agent** -- one run per section, returns that section's Markdown.

Both are built per call with ``client.as_agent(...)``: an agent is a thin
wrapper over the cached client, and its instructions depend on the brief's
scope, length, and audience.

Every entry point comes in two flavours. The ``a``-prefixed coroutine is the
real implementation; its sync twin wraps it via :mod:`app.services.async_bridge`
and is what the Streamlit UI calls.
"""

import json
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agent_framework import Agent

from app.agents.client import build_options, get_chat_client, outline_response_format
from app.agents.prompts import (
    build_outline_instructions,
    build_section_instructions,
    format_brief,
    format_section_message,
)
from app.core.exceptions import OutlineParseError
from app.core.logging import get_logger
from app.models.schemas import PRDInput, PRDOutline, PRDSection
from app.services.async_bridge import iter_sync, run_sync

logger = get_logger(__name__)

#: The outline reply is a short JSON object; the document-wide cap would let a
#: chatty model ramble past it.
OUTLINE_MAX_TOKENS = 2048

#: Last-resort match for a JSON object anywhere in a reply, used when a model
#: wraps the object in a code fence or a sentence.
_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------
def _build_outline_agent(prd_input: PRDInput) -> Agent:
    """Create the outline agent for one brief."""
    return get_chat_client().as_agent(
        name="prd-outline",
        instructions=build_outline_instructions(prd_input),
    )


def _coerce_outline(response: Any) -> PRDOutline:
    """Turn whatever the outline run produced into a :class:`PRDOutline`.

    Three strategies, cheapest first. ``response.value`` is already populated by
    the framework -- as the model under ``json_schema`` mode, as a plain dict
    under ``json_object`` -- but neither is guaranteed: Anthropic gets no
    ``response_format`` at all, and some models still wrap the object in a
    fence. Falling through to the outermost ``{...}`` costs nothing and saves a
    retry against a rate-limited free-tier model.

    Args:
        response: The ``AgentResponse`` returned by the outline run.

    Returns:
        The parsed outline.

    Raises:
        OutlineParseError: If no strategy found usable JSON in the reply.
    """
    value = response.value
    if isinstance(value, PRDOutline):
        return value
    if isinstance(value, dict):
        return PRDOutline.model_validate(value)

    text = (response.text or "").strip()
    try:
        return PRDOutline.model_validate_json(text)
    except ValueError:
        logger.debug("Outline reply was not bare JSON; trying the embedded-object fallback")

    match = _JSON_BLOCK_PATTERN.search(text)
    if match is None:
        raise OutlineParseError(f"Outline reply was not JSON: {text[:200]}")
    try:
        return PRDOutline.model_validate(json.loads(match.group(0)))
    except ValueError as exc:
        raise OutlineParseError(f"Outline reply was not a valid outline: {text[:200]}") from exc


async def agenerate_outline(prd_input: PRDInput) -> PRDOutline:
    """Plan the PRD: produce a title and an ordered list of sections.

    Args:
        prd_input: The validated brief.

    Returns:
        The outline the section runs will then expand.

    Raises:
        OutlineParseError: If the model's reply could not be parsed.
    """
    logger.info("Generating outline: scope=%s length=%s", prd_input.scope, prd_input.length)
    response = await _build_outline_agent(prd_input).run(
        format_brief(prd_input),
        options=build_options(
            response_format=outline_response_format(PRDOutline),
            max_tokens=OUTLINE_MAX_TOKENS,
        ),
    )
    outline = _coerce_outline(response)
    logger.info("Outline ready: %r, %d sections", outline.title, len(outline.sections))
    return outline


def generate_outline(prd_input: PRDInput) -> PRDOutline:
    """Blocking twin of :func:`agenerate_outline`, for the Streamlit UI."""
    return run_sync(agenerate_outline(prd_input))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _build_section_agent(
    prd_input: PRDInput,
    outline: PRDOutline,
    section_title: str,
    section_summary: str,
) -> Agent:
    """Create the section agent for one section of one outline."""
    return get_chat_client().as_agent(
        name="prd-section",
        instructions=build_section_instructions(prd_input, outline, section_title, section_summary),
    )


async def astream_section(
    prd_input: PRDInput,
    outline: PRDOutline,
    section_title: str,
    section_summary: str,
) -> AsyncIterator[str]:
    """Write one section, yielding text deltas as they arrive.

    Args:
        prd_input: The validated brief.
        outline: The full outline, sent along so the section stays consistent
            with its siblings.
        section_title: Title of the section to write.
        section_summary: The outline's one-line brief for that section.

    Yields:
        Non-empty text fragments, in order.
    """
    logger.info("Streaming section %r", section_title)
    agent = _build_section_agent(prd_input, outline, section_title, section_summary)
    stream = agent.run(
        format_section_message(prd_input, outline),
        stream=True,
        options=build_options(),
    )
    async for update in stream:
        if update.text:
            yield update.text


def stream_section(
    prd_input: PRDInput,
    outline: PRDOutline,
    section_title: str,
    section_summary: str,
) -> Iterator[str]:
    """Blocking twin of :func:`astream_section`, ready for ``st.write_stream``."""
    return iter_sync(lambda: astream_section(prd_input, outline, section_title, section_summary))


async def agenerate_section(
    prd_input: PRDInput,
    outline: PRDOutline,
    section_title: str,
    section_summary: str,
) -> PRDSection:
    """Write one section and return it whole, without streaming."""
    agent = _build_section_agent(prd_input, outline, section_title, section_summary)
    response = await agent.run(
        format_section_message(prd_input, outline),
        options=build_options(),
    )
    return PRDSection(title=section_title, content=(response.text or "").strip())


def generate_section(
    prd_input: PRDInput,
    outline: PRDOutline,
    section_title: str,
    section_summary: str,
) -> PRDSection:
    """Blocking twin of :func:`agenerate_section`."""
    return run_sync(agenerate_section(prd_input, outline, section_title, section_summary))
