"""The interview pipeline, as two Microsoft Agent Framework agents.

Two agents make up the whole thing:

* **interviewer** -- one run per turn, streams a single question. Carries the
  conversation via the framework's ``AgentSession``, so the history provider loads
  prior messages before the run and persists this turn's after it.
* **evaluator** -- one run at the end, returns a structured
  :class:`~app.models.schemas.FeedbackReport`. Deliberately given **no** session.

The interviewer agent is rebuilt on every call. That is not an oversight to
optimise away: the agent carries the instructions and the *session* carries the
history, so rebuilding per turn is what re-asserts the four persona rules on every
question while the conversation accumulates untouched.

Every entry point comes in two flavours. The ``a``-prefixed coroutine is the real
implementation; its sync twin wraps it via :mod:`app.services.async_bridge` and is
what the Streamlit UI calls.
"""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from agent_framework import AgentSession, InMemoryHistoryProvider
from pydantic import ValidationError

from app.agents.client import build_options, get_chat_client, structured_response_format
from app.agents.prompts import (
    build_evaluator_instructions,
    build_interviewer_instructions,
    format_answer_message,
    format_opening_request,
    format_transcript_for_evaluation,
)
from app.core.config import get_settings
from app.core.exceptions import EmptyInterviewError, ReportParseError
from app.core.logging import get_logger
from app.models.schemas import FeedbackReport, Transcript
from app.services.async_bridge import iter_sync, run_sync
from app.services.guardrails import sanitize_markdown
from app.services.transcript import HISTORY_SOURCE_ID

logger = get_logger(__name__)

#: Last-resort match for a JSON object anywhere in a reply, used when a model wraps
#: the object in a code fence or a sentence.
_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def new_agent_session() -> AgentSession:
    """Create the session that will hold one interview's conversation.

    The session is just an id plus a ``state`` dict; the history provider is
    attached to the *agent* (see :func:`_build_interviewer`) and writes into
    ``session.state[HISTORY_SOURCE_ID]["messages"]`` -- the framework scopes each
    provider's storage to its own sub-dict. Since the provider keeps no instance
    state of its own, the session object is the thing to hold across Streamlit
    reruns and the thing to purge.
    """
    return AgentSession()


# ---------------------------------------------------------------------------
# Interviewer
# ---------------------------------------------------------------------------
def _build_interviewer(config, question_number: int):
    """Create the interviewer agent for one turn.

    Fresh every call, with instructions rendered for this position in the
    interview. The persona rules ride along with it, which is the mechanism that
    keeps them in force on turn six as firmly as on turn one.

    A new ``InMemoryHistoryProvider`` each time is correct rather than wasteful:
    it holds no instance state, since the conversation lives in the session's
    ``state`` dict. Agent per turn, session for the whole interview.
    """
    return get_chat_client().as_agent(
        name="pm-interviewer",
        instructions=build_interviewer_instructions(config, question_number=question_number),
        context_providers=[InMemoryHistoryProvider(source_id=HISTORY_SOURCE_ID)],
    )


async def _astream_question(
    session: Any,
    message: str,
    question_number: int,
) -> AsyncIterator[str]:
    """Stream one question, bounded by the configured timeout.

    The timeout is not belt-and-braces. ``after_run`` -- and therefore persistence
    of both this turn's answer and its question -- fires only when the stream
    finalises, so a stream that never finalises would leave the session marked busy
    forever, blocking the candidate *and* the lock that is supposed to clear
    up after them.
    """
    settings = get_settings()
    agent = _build_interviewer(session.config, question_number)

    stream = agent.run(
        message,
        stream=True,
        session=session.agent_session,
        options=build_options(max_tokens=settings.interviewer_max_tokens),
    )

    emitted = False
    try:
        async with asyncio.timeout(settings.stream_timeout_seconds):
            async for update in stream:
                if update.text:
                    emitted = True
                    yield sanitize_markdown(update.text)
    except TimeoutError:
        logger.warning(
            "Interviewer stream exceeded %ss on question %s",
            settings.stream_timeout_seconds,
            question_number,
        )
        raise

    if not emitted:
        # An empty assistant message would open a turn with no question and
        # corrupt the projection's pairing, so it is a failure rather than an
        # empty question.
        raise ReportParseError("The interviewer returned an empty question.")


async def aopen_interview(session: Any) -> AsyncIterator[str]:
    """Ask the opening question, yielding text as it arrives."""
    logger.info(
        "Opening interview: type=%s seniority=%s archetype=%s",
        session.config.interview_type,
        session.config.seniority,
        session.config.archetype,
    )
    async for chunk in _astream_question(session, format_opening_request(session.config), 1):
        yield chunk


async def aask_followup(session: Any, answer: str) -> AsyncIterator[str]:
    """Record an answer and ask the next question, yielding text as it arrives.

    The answer is fenced by :func:`format_answer_message` before it is sent, so the
    fence is what gets persisted and is present on every later turn that loads the
    history.

    Args:
        session: The live session, whose ``agent_session`` carries the history.
        answer: The candidate's raw answer.

    Yields:
        Text fragments of the next question.
    """
    question_number = len(session.transcript().turns) + 1
    logger.info("Asking follow-up %s", question_number)
    async for chunk in _astream_question(session, format_answer_message(answer), question_number):
        yield chunk


def open_interview(session: Any) -> Iterator[str]:
    """Blocking twin of :func:`aopen_interview`, ready for ``st.write_stream``."""
    return iter_sync(lambda: aopen_interview(session))


def ask_followup(session: Any, answer: str) -> Iterator[str]:
    """Blocking twin of :func:`aask_followup`, ready for ``st.write_stream``."""
    return iter_sync(lambda: aask_followup(session, answer))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
def _build_evaluator(transcript: Transcript):
    """Create the evaluator agent for one finished interview."""
    return get_chat_client().as_agent(
        name="pm-evaluator",
        instructions=build_evaluator_instructions(transcript),
    )


def _coerce_report(response: Any) -> FeedbackReport:
    """Turn whatever the evaluator produced into a :class:`FeedbackReport`.

    Four strategies, cheapest first. ``response.value`` is already populated by the
    framework -- as the model under ``json_schema`` mode, as a plain dict under
    ``json_object`` -- but neither is guaranteed: Anthropic and Gemini take a
    different path, and free-tier models wrap the object in a code fence often
    enough that falling through to the outermost ``{...}`` costs nothing and saves
    a retry against a rate-limited model.

    Args:
        response: The ``AgentResponse`` returned by the evaluator run.

    Returns:
        The parsed report.

    Raises:
        ReportParseError: If no strategy found usable JSON.
        ValidationError: If JSON was found but does not satisfy the schema. Left
            unwrapped on purpose -- the caller uses the validation message to build
            a repair request.
    """
    value = getattr(response, "value", None)
    if isinstance(value, FeedbackReport):
        return value
    if isinstance(value, dict):
        return FeedbackReport.model_validate(value)

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ReportParseError("The evaluator returned an empty reply.")

    try:
        return FeedbackReport.model_validate_json(text)
    except ValidationError:
        # Distinguish "not JSON at all" from "JSON that failed the schema": only
        # the first should fall through to the regex.
        try:
            json.loads(text)
        except ValueError:
            pass
        else:
            raise

    match = _JSON_BLOCK_PATTERN.search(text)
    if match is None:
        raise ReportParseError(f"Evaluator reply was not JSON: {text[:200]}")
    try:
        parsed = json.loads(match.group(0))
    except ValueError as exc:
        raise ReportParseError(f"Evaluator reply was not valid JSON: {text[:200]}") from exc
    return FeedbackReport.model_validate(parsed)


def _repair_request(error: ValidationError) -> str:
    """Build the correction message for a report that failed the schema.

    A near-miss -- one dimension omitted, one `evidence` left empty -- is worth
    repairing rather than regenerating. It is cheaper, likelier to succeed, and it
    does not spend the user-visible retry that the grace window exists for.
    """
    problems = "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()
    )
    return (
        "Your previous reply was valid JSON but did not satisfy the required shape. "
        f"Problems: {problems}. "
        "Reply again with the corrected JSON object and nothing else. Keep every "
        "score and quotation you already justified; only fix what is listed."
    )


async def agenerate_report(transcript: Transcript) -> FeedbackReport:
    """Grade a finished interview.

    One call, whole transcript, structured output. Not streamed: the report has to
    validate before it renders, so there is nothing useful to show mid-flight.

    No ``session=`` argument, and that is deliberate twice over. An agent grading a
    conversation it believes it conducted rates it generously; and passing the
    session would have ``after_run`` persist the grading call into the interview
    history.

    Args:
        transcript: The finished conversation.

    Returns:
        The validated report.

    Raises:
        EmptyInterviewError: If nothing was answered. Grading an empty transcript
            would have the evaluator invent five scores from no evidence.
        ReportParseError: If the reply could not be parsed, or a schema failure
            survived the repair attempt.
    """
    if not transcript.is_gradable:
        raise EmptyInterviewError(
            "The interview ended before any question was answered, so there is nothing to grade."
        )

    settings = get_settings()
    agent = _build_evaluator(transcript)
    document = format_transcript_for_evaluation(transcript)
    options = build_options(
        response_format=structured_response_format(FeedbackReport),
        max_tokens=settings.report_max_tokens,
    )

    logger.info("Grading interview: %d answered turn(s)", transcript.answered_turns)
    response = await agent.run(document, options=options)

    try:
        report = _coerce_report(response)
    except ValidationError as exc:
        logger.info("Report failed validation; attempting one repair")
        repair = await agent.run(f"{document}\n\n{_repair_request(exc)}", options=options)
        try:
            report = _coerce_report(repair)
        except ValidationError as repair_exc:
            raise ReportParseError(
                f"The evaluator's report did not match the required shape, even after "
                f"a correction attempt: {repair_exc}"
            ) from repair_exc

    _warn_on_unquotable_evidence(report, transcript)
    logger.info(
        "Report ready: %s",
        ", ".join(f"{score.dimension}={score.score}" for score in report.scores),
    )
    return report


def generate_report(transcript: Transcript) -> FeedbackReport:
    """Blocking twin of :func:`agenerate_report`."""
    return run_sync(agenerate_report(transcript))


def _warn_on_unquotable_evidence(report: FeedbackReport, transcript: Transcript) -> None:
    """Log when a score's evidence does not appear in the transcript.

    Log only, by decision (E-21). A substring check catches a fabricated quote but
    false-positives on a legitimate near-quote or a paraphrase, and a wrong
    "unverified" badge on a real quote would damage trust more than the rare
    fabrication it caught.
    """
    answers = " ".join(turn.answer or "" for turn in transcript.turns).lower()
    if not answers:
        return
    for score in report.scores:
        probe = score.evidence.strip().strip('"').lower()[:60]
        if probe and probe not in answers:
            logger.warning(
                "Evidence for %s may be paraphrased or fabricated: %r",
                score.dimension,
                score.evidence[:80],
            )
