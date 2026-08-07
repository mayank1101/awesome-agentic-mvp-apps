"""Projection from stored messages to the domain transcript.

The framework owns the conversation: it lives in ``AgentSession.state["messages"]``
as a flat list of ``Message`` objects, written by the history provider's
``after_run`` hook. This module is the only place that turns that list into
question-and-answer :class:`~app.models.schemas.Turn` pairs.

Pure, and deliberately defensive. It is the seam between a structure we do not
control and every consumer that does -- the chat rendering, the turn counter, the
budget check, and the document the evaluator grades. A misalignment here would
mispair every quote in the report, silently, so the ragged cases get explicit
handling rather than an assumption.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.core.logging import get_logger
from app.models.schemas import InterviewConfig, Transcript, Turn
from app.services.guardrails import unfence

logger = get_logger(__name__)

#: Roles that carry conversation. Anything else in the history -- a system or tool
#: message -- is not part of the interview and would corrupt the pairing.
_ASSISTANT = "assistant"
_USER = "user"

#: The ``source_id`` our history provider is constructed with, and therefore the
#: key its messages live under inside ``AgentSession.state``.
#:
#: The framework scopes each context provider's storage to its own sub-dict --
#: ``state.setdefault(provider.source_id, {})`` -- so the conversation is at
#: ``state[HISTORY_SOURCE_ID]["messages"]`` and *not* at ``state["messages"]``. We
#: set the id explicitly rather than relying on the provider's default so the key
#: is greppable and cannot change under us.
HISTORY_SOURCE_ID = "interview_history"


def messages_from_state(state: Mapping[str, Any] | None) -> list[Any]:
    """Read the stored conversation out of an ``AgentSession.state``.

    The single place that knows the storage layout, so the session store, the
    projection, and any future reader cannot disagree about it.

    Args:
        state: The session's state mapping, or ``None``.

    Returns:
        The messages in order, or an empty list if nothing has been stored yet.
    """
    if not isinstance(state, Mapping):
        return []
    scoped = state.get(HISTORY_SOURCE_ID)
    if not isinstance(scoped, Mapping):
        return []
    messages = scoped.get("messages")
    return list(messages) if isinstance(messages, list) else []


def append_answer_to_state(state: Any, fenced_answer: str) -> bool:
    """Append a candidate answer to stored history directly, without a model call.

    Needed for exactly one case: the **final** answer of an interview. Every other
    answer reaches history as the input of the next follow-up call, persisted by the
    framework's ``after_run``. The last answer has no follow-up to ride along with —
    the interview is over — so without this it would be graded from nowhere, or a
    wasted question would have to be asked purely to store it.

    Args:
        state: The session's state mapping.
        fenced_answer: The answer, already fenced. Fencing stays the caller's job so
            the stored form is identical whichever path wrote it.

    Returns:
        Whether the answer was stored.
    """
    from agent_framework import Message

    if not isinstance(state, dict):
        return False
    scoped = state.setdefault(HISTORY_SOURCE_ID, {})
    if not isinstance(scoped, dict):
        return False
    messages = scoped.setdefault("messages", [])
    if not isinstance(messages, list):
        return False
    messages.append(Message(role="user", contents=[fenced_answer]))
    return True


def _role_of(message: Any) -> str:
    """Read a message's role as a plain string.

    The framework types it as ``RoleLiteral | str`` and some paths carry an enum,
    so this normalises rather than assuming either.
    """
    role = getattr(message, "role", "")
    return str(getattr(role, "value", role))


def _text_of(message: Any) -> str:
    """Concatenate a message's text parts, with no separator inserted.

    Deliberately *not* ``Message.text``, which joins parts with a single space.
    A streamed question can be stored as one content part per delta, and those
    deltas already carry their own leading spaces -- so a space-joining read
    turns "Hel" + "lo" into "Hel lo". Concatenating without a separator is
    lossless whether the framework coalesced the deltas or not.
    """
    contents: Iterable[Any] = getattr(message, "contents", None) or []
    parts = [
        content.text
        for content in contents
        if getattr(content, "type", None) == "text" and getattr(content, "text", None)
    ]
    if parts:
        return "".join(parts)
    # Fall back for any message shape without typed contents.
    return str(getattr(message, "text", "") or "")


def to_transcript(config: InterviewConfig, messages: Sequence[Any]) -> Transcript:
    """Build the domain transcript from the framework's stored messages.

    Pairing rule: each assistant message opens a turn, and the next user message
    closes it. Three ragged cases are handled explicitly rather than assumed away:

    * **Consecutive assistant messages** are concatenated into one question. This
      should not happen, but a retry that partially persisted could produce it,
      and the alternative -- treating the second as a new turn -- would shift
      every later answer onto the wrong question and corrupt every quote in the
      report.
    * **A user message before any question** is ignored. It cannot occur through
      the UI, so it is not trusted either.
    * **A trailing question with no answer** yields a turn with ``answer=None``,
      which is the state the interview sits in for most of its life.

    Args:
        config: The configuration the session was started with. Not derivable
            from the messages, so it is passed in.
        messages: ``AgentSession.state["messages"]``, in order.

    Returns:
        The transcript. Answers come back unfenced, since the fence is a
        prompt-level construct and no consumer of this object wants to see it.
    """
    turns: list[Turn] = []
    orphaned_answers = 0

    for message in messages:
        role = _role_of(message)
        text = _text_of(message)
        if not text.strip():
            continue

        if role == _ASSISTANT:
            if turns and turns[-1].answer is None:
                # Two questions in a row, with nothing between them.
                turns[-1] = turns[-1].model_copy(
                    update={"question": f"{turns[-1].question}\n\n{text}"}
                )
                logger.warning("Concatenated consecutive assistant messages into one question")
            else:
                turns.append(Turn(index=len(turns), question=text))
        elif role == _USER:
            if not turns:
                orphaned_answers += 1
                continue
            if turns[-1].answer is None:
                turns[-1] = turns[-1].model_copy(update={"answer": unfence(text)})
            else:
                # An answer with no question of its own. Appending to the previous
                # answer keeps the text rather than dropping a candidate's words.
                merged = f"{turns[-1].answer}\n\n{unfence(text)}"
                turns[-1] = turns[-1].model_copy(update={"answer": merged})
                logger.warning("Merged a second consecutive answer into the open turn")

    if orphaned_answers:
        # Debug rather than warning, because one of these is *expected*: the message
        # that kicks the interview off ("ask your opening question") is a user
        # message, and the framework persists run inputs, so it sits in history
        # ahead of the first question. Logging it as a warning meant a line on every
        # single render.
        logger.debug("Ignored %d message(s) that preceded any question", orphaned_answers)

    return Transcript(config=config, turns=turns)
