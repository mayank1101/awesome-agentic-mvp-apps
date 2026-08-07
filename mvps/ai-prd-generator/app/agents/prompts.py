"""Instruction templates and the brief-to-prompt formatting.

Prompt text lives apart from the agent plumbing so it can be read and tuned
without scrolling past client construction. Every template is a `str.format`
template; the placeholders are filled in by :mod:`app.agents.prd_agents`.

Two rules hold across all of them:

* The brief travels in the **user message** built by :func:`format_brief`, and
  is wrapped in the guardrail fence so the model is told it is data.
* The few user-derived values that must appear in instructions -- `audience`,
  and the section title and summary the outline produced -- are neutralised
  first by :func:`_defang`, because instructions are the one place the fence
  cannot protect.
"""

from app.agents.presets import LENGTH_PRESETS, MIN_SECTION_WORDS, SCOPE_PRESETS
from app.models.schemas import PRDInput, PRDOutline
from app.services.guardrails import UNTRUSTED_DATA_NOTICE, fence

AUDIENCE_GUIDANCE = """Write for a {audience} audience: match their vocabulary and the \
depth they need. For engineering-leaning readers lead with architecture, data flow, \
interfaces, and failure modes; for business-leaning readers lead with user impact, \
outcomes, and trade-offs, keeping technical detail brief and plain."""

# The JSON shape is spelled out here rather than left to the provider, because
# structured_output_mode defaults to plain json_object -- many free OpenRouter
# models reject a strict json_schema response_format. Braces are doubled to
# survive str.format.
OUTLINE_INSTRUCTIONS = (
    """You are a senior product management partner who writes crisp, well-structured PRDs."""
    + AUDIENCE_GUIDANCE
    + """Given the {subject} context you are sent, produce a PRD outline: an overall title \
and an ordered list of sections tailored to this specific {subject}.

Target length: about {total_words} words total across {section_count} sections. Pick the \
sections that matter most -- do not pad the outline with sections just to hit a count.

{sections_hint}

Reply with one JSON object and nothing else -- no prose, no code fence -- in exactly this shape:
{{"title": "overall PRD title", "sections": [{{"title": "section title", "summary": "1-2 \
sentence brief for this section"}}]}}"""
)

SECTION_INSTRUCTIONS = (
    """You are a senior product management partner writing one section of a PRD about a {subject}."""
    + AUDIENCE_GUIDANCE
    + """Write the content for the section "{section_title}" ({section_summary}) in clean Markdown. \
Be specific and concrete, use bullet points and sub-headings where useful, and stay \
consistent with the rest of the PRD outline. Do not repeat the section title as a heading \
-- start directly with the content.

Keep this section to roughly {word_budget} words, be concise and avoid filler; do not \
pad to reach the target."""
)


#: Longest user-derived value allowed inside instructions. Well above any honest
#: audience or section title, short enough that a pasted instruction block gets
#: cut off rather than reasoned about.
_MAX_INLINE_LENGTH = 120


def _defang(value: str) -> str:
    """Make a user- or model-derived string safe to interpolate into instructions.

    Collapses newlines, drops the characters that start a chat-template role
    marker, and truncates. This is a blunt instrument on purpose: the values it
    guards -- `audience`, section titles -- are short phrases, so nothing
    legitimate survives it differently.
    """
    flattened = " ".join(str(value).split())
    stripped = flattened.replace("<", "").replace(">", "").replace("|", "")
    return stripped[:_MAX_INLINE_LENGTH]


def build_outline_instructions(prd_input: PRDInput) -> str:
    """Render the outline agent's instructions for one brief."""
    scope = SCOPE_PRESETS[prd_input.scope]
    length = LENGTH_PRESETS[prd_input.length]
    return (
        OUTLINE_INSTRUCTIONS.format(
            audience=_defang(prd_input.audience),
            subject=scope["subject"],
            sections_hint=scope["sections_hint"],
            total_words=length["total_words"],
            section_count=length["section_count"],
        )
        + UNTRUSTED_DATA_NOTICE
    )


def build_section_instructions(
    prd_input: PRDInput,
    outline: PRDOutline,
    section_title: str,
    section_summary: str,
) -> str:
    """Render the section agent's instructions for one section of one outline.

    The word budget is the document total split evenly across the sections the
    outline actually produced, floored at :data:`MIN_SECTION_WORDS`.

    The title and summary come from the outline the *model* wrote, which the
    brief influenced -- so they are defanged like any other untrusted value
    before landing in instructions.
    """
    return (
        SECTION_INSTRUCTIONS.format(
            audience=_defang(prd_input.audience),
            subject=SCOPE_PRESETS[prd_input.scope]["subject"],
            section_title=_defang(section_title),
            section_summary=_defang(section_summary),
            word_budget=section_word_budget(prd_input, outline),
        )
        + UNTRUSTED_DATA_NOTICE
    )


def section_word_budget(prd_input: PRDInput, outline: PRDOutline) -> int:
    """Words to allot one section, given the document budget and section count."""
    total_words = LENGTH_PRESETS[prd_input.length]["total_words"]
    section_count = max(len(outline.sections), 1)
    return max(total_words // section_count, MIN_SECTION_WORDS)


def format_brief(prd_input: PRDInput) -> str:
    """Flatten a brief into the user message both agents receive.

    Optional fields are omitted rather than sent empty, so the model is not
    handed blank headings to fill in. The whole thing is fenced, pairing with
    :data:`~app.services.guardrails.UNTRUSTED_DATA_NOTICE` in the instructions.
    """
    subject = SCOPE_PRESETS[prd_input.scope]["subject"]
    lines = [f"{subject.capitalize()} name: {prd_input.product_name or '(unnamed)'}"]

    if prd_input.scope == "feature":
        parent = prd_input.parent_product or "(not specified -- infer conservatively)"
        lines.append(f"Existing product this feature is being added to:\n{parent}")

    lines += [
        f"One-liner: {prd_input.one_liner}",
        f"Problem statement: {prd_input.problem_statement}",
        f"Target users: {prd_input.target_users}",
    ]
    if prd_input.goals:
        lines.append("Goals:\n" + "\n".join(f"- {goal}" for goal in prd_input.goals))
    if prd_input.context_notes:
        lines.append(f"Additional context / notes:\n{prd_input.context_notes}")

    return fence("\n\n".join(lines))


def format_section_message(prd_input: PRDInput, outline: PRDOutline) -> str:
    """Build the section agent's user message: the brief plus the full outline.

    Every section run sees the whole outline, which is what keeps sections from
    repeating each other or contradicting a sibling.
    """
    outline_str = "\n".join(f"- {s.title}: {s.summary}" for s in outline.sections)
    return (
        f"Full PRD context:\n{format_brief(prd_input)}\n\n"
        f"Full outline (for consistency):\n{outline_str}"
    )
