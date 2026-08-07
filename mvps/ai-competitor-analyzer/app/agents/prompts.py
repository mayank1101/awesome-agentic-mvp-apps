"""The synthesis prompt.

One call, one prompt. Its whole job is to turn packed evidence into six sections
of prose, and the two things it must be talked out of are the two failures that
make this app worse than useless:

* **Writing from memory.** The model knows things about Notion. Everything it
  knows is stale, unsourced, and indistinguishable in tone from what the
  evidence says. So "Not found in public sources" is presented as a *correct*
  answer with a worked example, not as a permitted fallback -- a model that has
  seen the phrase used approvingly reaches for it; one that has only seen it
  allowed does not.
* **Following instructions it read on a web page.** The fence notice is in
  :mod:`app.services.guardrails`, and it names the likely attacker outright: the
  pages are written by the company being profiled.

The reply is JSON with exactly six keys. Markdown comes later, from the renderer,
which is what keeps the sources list and the URLs out of the model's reach.
"""

from app.models.schemas import NOT_FOUND, CompanyIdentity, SectionEvidence, SectionKey
from app.services.guardrails import UNTRUSTED_DATA_NOTICE, fence

#: What each section must contain, in the model's own working order. Written as
#: instructions to a researcher rather than as field descriptions, because the
#: failure mode is tonal: a model told to "describe pricing" writes marketing
#: copy, one told to "report the tiers as published" reports tiers.
SECTION_BRIEFS: dict[SectionKey, str] = {
    SectionKey.SNAPSHOT: (
        "What the company sells, in two sentences, then founding year, headquarters, "
        "approximate size, and ownership or funding status. Only what the evidence states."
    ),
    SectionKey.PRODUCT: (
        "The main products and the capabilities the company leads with. Group related "
        "features; do not list every checkbox mentioned in a comparison table."
    ),
    SectionKey.PRICING: (
        "Tiers and pricing model as published, with the figures and the units "
        "(per seat, per month, annual). If pricing is gated behind a sales conversation, "
        "say exactly that instead of estimating. Never infer a price from a competitor's."
    ),
    SectionKey.POSITIONING: (
        "Who the company says it is for, and the claim it makes about itself. Quote its "
        "own words where the evidence carries them, in quotation marks."
    ),
    SectionKey.RECENT_MOVES: (
        "Dated events from the last twelve months: launches, funding, acquisitions, "
        "leadership changes. One bullet each, each starting with the date as given in the "
        "evidence. Omit anything undated -- an undated claim is worse than no claim."
    ),
    SectionKey.STRENGTHS_WEAKNESSES: (
        "Two short lists: strengths, then weaknesses. This is the one section that "
        "interprets rather than reports, so every point must trace to something in the "
        "evidence -- praise and complaints that users actually published, capabilities "
        "the other sections established. No speculation about strategy or finances."
    ),
}

_SYSTEM = f"""You are a competitive-research analyst. You write one brief about one
company, using only the evidence supplied to you.

Your entire source of truth is the retrieved evidence in this message. You have
background knowledge about many companies; it is out of date, it is unsourced, and
it is not usable here. If the evidence does not support a statement, you do not
make it.

When a section has no supporting evidence, its value is exactly: "{NOT_FOUND}"

That is a correct, expected answer and a good outcome. A reader can act on "not
published"; a reader cannot act on a plausible guess, and will not know it was one.
Example of the right behaviour: if the pricing evidence contains only third-party
blog commentary and no published figures, the pricing value is
"{NOT_FOUND}" rather than an approximation assembled from context.

Style: plain, specific, and short. Markdown for structure inside a section
(paragraphs, `-` bullets, `**bold**` for tier names) but never a heading -- headings
are added later. Do not include URLs, links, or citation markers of any kind; sources
are attached separately. Do not refer to "the evidence", "the sources", or "the
snippets"; write the finding itself.

Reply with a single JSON object and nothing else. Exactly these six keys, all
strings, no others:
{", ".join(key.value for key in SectionKey)}
{UNTRUSTED_DATA_NOTICE}"""


def system_instructions() -> str:
    """Return the synthesiser's system instructions."""
    return _SYSTEM


def format_evidence(sections: tuple[SectionEvidence, ...] | list[SectionEvidence]) -> str:
    """Render packed evidence as the fenced block the model reads.

    Each item is labelled with its id and title but **not its URL**: a model that
    never sees a link cannot reproduce one, which is what turns "no invented
    links" from an instruction into a property (SC-1).

    Args:
        sections: Packed evidence, in render order.

    Returns:
        The fenced evidence block.
    """
    blocks: list[str] = []

    for section in sections:
        blocks.append(f"## Evidence for: {section.section.value}")
        if section.is_empty:
            blocks.append(f"(no results for the query “{section.query}”)")
            continue

        for item in section.items:
            dated = f" [published {item.published_date}]" if item.published_date else ""
            blocks.append(f"[{item.id}] {item.title}{dated}\n{item.content}")

    return fence("\n\n".join(blocks))


def build_user_message(
    identity: CompanyIdentity,
    sections: tuple[SectionEvidence, ...] | list[SectionEvidence],
) -> str:
    """Assemble the single user message for the synthesis call.

    Args:
        identity: Which company is being profiled.
        sections: The packed evidence.

    Returns:
        The message text: who the subject is, what each section needs, then the
        fenced evidence.
    """
    subject = identity.name
    if identity.domain:
        subject += f" ({identity.domain})"

    briefs = "\n".join(f"- **{key.value}**: {brief}" for key, brief in SECTION_BRIEFS.items())

    return (
        f"Write the competitor brief for: {subject}\n\n"
        f"Sections to fill:\n{briefs}\n\n"
        f"Evidence follows. Every section must be written only from the evidence "
        f"labelled for it.\n\n{format_evidence(sections)}"
    )


REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON with exactly the six required keys. "
    "Reply again with only the JSON object: no prose before or after it, no code "
    "fence, no extra keys. Every value is a string."
)
