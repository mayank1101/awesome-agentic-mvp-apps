"""Instruction templates and message formatting for both agents.

Prompt text lives apart from the agent plumbing so it can be read and tuned
without scrolling past client construction.

Three rules hold across everything here:

* **The candidate's words travel in a user message, fenced.** Instructions never
  interpolate an answer.
* **The few user-derived values that must appear in instructions** -- only
  ``focus_area`` -- are neutralised by :func:`_defang` first, because instructions
  are the one place the fence cannot protect.
* **The interviewer's rules are rebuilt on every turn.** The agent is constructed
  per call, so the four rules and the calibration are re-sent with each question
  rather than set once at the start of the conversation. That is the whole
  mechanism preventing persona drift across six turns.
"""

from app.agents import rubric
from app.agents.presets import (
    ARCHETYPE_PRESETS,
    INTERVIEW_TYPE_PRESETS,
    SENIORITY_PRESETS,
    primary_dimensions,
)
from app.models.schemas import InterviewConfig, Transcript
from app.services.guardrails import UNTRUSTED_DATA_NOTICE, fence

#: Longest user-derived value allowed inside instructions. Well above any honest
#: focus area, short enough that a pasted instruction block gets cut off rather
#: than reasoned about.
_MAX_INLINE_LENGTH = 120


def _defang(value: str) -> str:
    """Make a user-derived string safe to interpolate into instructions.

    Collapses newlines, drops the characters that start a chat-template role
    marker, and truncates. A blunt instrument on purpose: the only value it
    guards is a short phrase, so nothing legitimate survives it differently.
    """
    flattened = " ".join(str(value).split())
    stripped = flattened.replace("<", "").replace(">", "").replace("|", "")
    return stripped[:_MAX_INLINE_LENGTH]


# --- interviewer --------------------------------------------------------------

#: The four rules, verbatim on every turn. Rule 1's carve-out is not a softening:
#: a real interviewer answers scope questions, and an interviewer that refuses to
#: makes the session useless rather than rigorous.
_INTERVIEWER_RULES = """Rules you follow without exception:

1. ASK, NEVER ANSWER. Do not offer a framework, do not suggest an approach, do not
   solve any part of the question, and do not hint. The one exception: if the
   candidate asks a scope or constraint question ("consumers or businesses?",
   "which market?"), answer it in one short sentence and then re-ask your
   question. Clarifying scope is legitimate; supplying reasoning is not.
2. GROUND EVERY FOLLOW-UP. Each follow-up must attach to something specific the
   candidate actually said -- the assumption underneath it, the trade-off they
   skipped, or the number they did not give. If your question would make sense
   against any answer, it is the wrong question.
3. ONE QUESTION PER TURN. Never stack two questions; it lets the candidate answer
   the easier one and ignore the other.
4. NO PRAISE, NO GRADING, NO SUMMARY. Do not say "good", do not evaluate, do not
   recap what they said back to them. Feedback comes later from someone else, and
   encouragement mid-interview teaches nothing.

Output only the question itself. No preamble, no numbering, no stage directions."""


def build_interviewer_instructions(
    config: InterviewConfig,
    *,
    question_number: int,
) -> str:
    """Render the interviewer's instructions for one turn.

    Args:
        config: The session's configuration.
        question_number: 1-based position of the question about to be asked.
            Included so the interviewer knows where it is in the interview and
            does not wrap up early or drift into a summary.

    Returns:
        The full instruction string, ending with the untrusted-data notice.
    """
    interview = INTERVIEW_TYPE_PRESETS[config.interview_type]
    seniority = SENIORITY_PRESETS[config.seniority]
    archetype = ARCHETYPE_PRESETS[config.archetype]

    focus = _defang(config.focus_area) if config.focus_area else ""
    focus_line = (
        f"\nAnchor the interview in this domain: {focus}."
        if focus
        else "\nPick the domain yourself; do not ask the candidate to choose one."
    )

    probes = "\n".join(f"  - {angle}" for angle in interview.probe_angles)
    dimensions = ", ".join(
        rubric.dimension(key).label for key in primary_dimensions(config.interview_type)
    )

    return f"""You are an experienced product manager conducting a {interview.label} interview \
for a {seniority.label} role at a {archetype.label} company.

Company context: {archetype.context}
At this company a good answer optimises for {archetype.optimises_for}.
The bar for this level: {seniority.bar}{focus_line}

This is question {question_number} of {config.total_questions}. Ask only this question and stop.

What the opening question should look like: {interview.question_brief}

When following up, press on one of these:
{probes}

You are probing for: {dimensions}. Choose the follow-up that gives the candidate \
the clearest opportunity to show one of those, or exposes its absence.

{_INTERVIEWER_RULES}{UNTRUSTED_DATA_NOTICE}"""


def format_opening_request(config: InterviewConfig) -> str:
    """The user message that starts the interview.

    Carries no candidate text, so it needs no fence. Phrased as a request rather
    than left empty, because an empty first message makes some models produce a
    greeting instead of a question.
    """
    interview = INTERVIEW_TYPE_PRESETS[config.interview_type]
    return (
        f"Begin the interview. Ask your opening {interview.label} question now. "
        "Do not greet me, do not explain the format, and do not tell me how long "
        "we have. Just the question."
    )


def format_answer_message(answer: str) -> str:
    """Wrap a candidate's answer for sending, and therefore for storage.

    The returned string is what the history provider persists, so the fence
    travels with the message and is present on every later turn that loads it.
    There is no later opportunity to add it -- the prompt is assembled by the
    framework, not by us.
    """
    return fence(answer)


# --- evaluator ----------------------------------------------------------------

#: Spelled out here rather than left to the provider, because
#: STRUCTURED_OUTPUT_MODE defaults to plain json_object -- many free models reject
#: a strict json_schema response_format.
_REPORT_JSON_SHAPE = """{
  "headline": "one sentence naming the main strength and the main gap",
  "scores": [
    {"dimension": "structure", "score": 1-4, "justification": "one line", "evidence": "quoted from the transcript"},
    {"dimension": "user_insight", "score": 1-4, "justification": "...", "evidence": "..."},
    {"dimension": "prioritization", "score": 1-4, "justification": "...", "evidence": "..."},
    {"dimension": "metrics", "score": 1-4, "justification": "...", "evidence": "..."},
    {"dimension": "communication", "score": 1-4, "justification": "...", "evidence": "..."}
  ],
  "what_worked": ["specific moment, quoted", "another"],
  "what_cost_points": [{"moment": "what happened", "stronger_move": "what to have done instead"}],
  "rewritten_answer": {"question": "the question being re-answered", "rewrite": "the stronger answer in full", "why_better": "what it does that the original did not"},
  "next_focus": "one dimension to practise and a concrete drill"
}"""

_CALIBRATION = """Calibration, which matters more than anything else in this task:

- The modal candidate scores 2 or 3 on most dimensions. A 4 is genuinely strong,
  not merely adequate. If you find yourself awarding 4s across the board, you have
  misread the rubric -- go back and apply the level descriptions literally.
- A 1 is not an insult and you must use it when it fits. A dimension the candidate
  never addressed at all is a 1, and the evidence for it is what they said instead.
- The scale has no middle. Every dimension is either below the bar (1-2) or at or
  above it (3-4), so each score is a decision you are making rather than a hedge.
- Every score needs `evidence`: a specific moment quoted or closely paraphrased
  from the transcript. A score you cannot attach a moment to is a score you have
  not justified, and it will be rejected.
- Judge only what is in the transcript. Do not credit or penalise anything the
  candidate did not say."""


def build_evaluator_instructions(transcript: Transcript) -> str:
    """Render the evaluator's instructions for one finished interview.

    The evaluator is a fresh reader: it is never given the interview session, so
    it has no memory of having asked the questions. An agent grading a
    conversation it believes it conducted rates it generously.

    Args:
        transcript: The finished conversation. Used for its configuration and its
            answer count -- the transcript text itself travels in the user
            message, fenced.

    Returns:
        The full instruction string, ending with the untrusted-data notice.
    """
    config = transcript.config
    interview = INTERVIEW_TYPE_PRESETS[config.interview_type]
    seniority = SENIORITY_PRESETS[config.seniority]
    archetype = ARCHETYPE_PRESETS[config.archetype]

    primary = primary_dimensions(config.interview_type)
    primary_labels = ", ".join(rubric.dimension(key).label for key in primary)

    answered = transcript.answered_turns
    sample_note = (
        f"The candidate answered {answered} of {config.total_questions} questions. "
        "That is a short sample: grade what exists, and say so in the headline "
        "rather than implying a verdict on the candidate."
        if answered < 3
        else f"The candidate answered {answered} of {config.total_questions} questions."
    )

    return f"""You are a hiring manager grading a {interview.label} interview for a \
{seniority.label} role at a {archetype.label} company.

You did not conduct this interview. You are reading a transcript.

At this company a good answer optimises for {archetype.optimises_for}.
The bar for this level: {seniority.bar}
{sample_note}

Score all five dimensions against this rubric, using the level descriptions literally:

{rubric.format_for_instructions()}

{_CALIBRATION}

This was a {interview.label} interview, which primarily tests: {primary_labels}. Score \
every dimension, but `next_focus` must name one of those primary dimensions -- never \
send a candidate away to practise something this interview format could not exercise. \
Refer to it by that display name, not by an internal key like "user_insight".

`rewritten_answer` takes the candidate's weakest answer and rewrites it at the level \
that would clear the bar, so the difference is visible rather than described.

Reply with one JSON object and nothing else -- no prose, no code fence -- in exactly \
this shape:
{_REPORT_JSON_SHAPE}{UNTRUSTED_DATA_NOTICE}"""


def format_transcript_for_evaluation(transcript: Transcript) -> str:
    """Render the conversation as the document the evaluator grades.

    Each answer is re-fenced. The projection stripped the stored fence to recover
    readable text, and this is a *different* prompt being assembled, so the
    protection has to be reapplied rather than assumed -- the candidate's words are
    no less untrusted for having been through a projection.

    Unanswered questions are included and marked, so the evaluator can see that a
    question was asked and abandoned rather than silently grading a shorter
    interview.
    """
    blocks: list[str] = []
    for turn in transcript.turns:
        blocks.append(f"INTERVIEWER (question {turn.index + 1}):\n{turn.question}")
        if turn.is_answered:
            blocks.append(f"CANDIDATE:\n{fence(turn.answer or '')}")
        else:
            blocks.append("CANDIDATE:\n(no answer -- the interview ended here)")
    return "\n\n".join(blocks)
