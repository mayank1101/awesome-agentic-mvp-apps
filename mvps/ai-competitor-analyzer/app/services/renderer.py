"""Turning a synthesis result and its evidence into the finished document.

One Markdown string feeds both the screen and the download, so they cannot drift
(SC-9). Everything the model is not trusted with happens here:

* **URLs are attached from the app's own records**, never from the reply.
* **Every URL in the output is checked against what was retrieved** (SC-1).
* **Evidence ids and fence lookalikes are stripped** from the prose (E-40).
* **Retrieved titles are sanitised** before they land in the sources list — the
  one place third-party text reaches the page verbatim (E-43).
* **Undated items in recent moves are dropped**, here rather than in the prompt,
  because a rule enforced in code holds and a rule requested of a model mostly
  holds (E-23, Q-4).
"""

import re
from datetime import date

from app.models.schemas import (
    NOT_FOUND,
    SECTION_TITLES,
    CompanyIdentity,
    SectionEvidence,
    SectionKey,
    SynthesisResult,
)
from app.services.evidence import known_urls
from app.services.guardrails import (
    sanitize_markdown,
    sanitize_title,
    strip_evidence_ids,
    strip_unknown_urls,
)

#: A bullet in recent moves that opens with something date-shaped: an ISO date,
#: a month name, or a bare year. Anything else is undated and dropped.
_DATED_BULLET = re.compile(
    r"^\s*[-*]\s*\*{0,2}\(?\s*("
    r"\d{4}-\d{2}(-\d{2})?"
    r"|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}"
    r"|\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}"
    r"|q[1-4]\s+\d{4}"
    r"|\d{4}"
    r")",
    re.IGNORECASE,
)


def drop_undated_items(text: str) -> str:
    """Remove undated bullets from the recent-moves section (E-23).

    Args:
        text: The section body as the model wrote it.

    Returns:
        The body with undated bullets removed, or :data:`NOT_FOUND` if that
        leaves nothing — an undated competitive claim is worse than no claim, and
        an empty section stated plainly is a result.
    """
    if text.strip() == NOT_FOUND:
        return text

    lines = text.splitlines()
    kept = [
        line
        for line in lines
        if not line.lstrip().startswith(("-", "*")) or _DATED_BULLET.match(line)
    ]

    body = "\n".join(kept).strip()
    has_bullet = any(line.lstrip().startswith(("-", "*")) for line in kept)
    return body if (body and has_bullet) else NOT_FOUND


def _render_section(
    section: SectionKey,
    result: SynthesisResult,
    evidence: SectionEvidence | None,
) -> str:
    """Render one section: heading, body, and its sources."""
    body = strip_evidence_ids(result.section(section)).strip()

    if section is SectionKey.RECENT_MOVES:
        body = drop_undated_items(body)

    if evidence is None or evidence.is_empty:
        # Say what was searched, so "nothing found" is a statement about the web
        # rather than an unexplained blank.
        body = NOT_FOUND
        if evidence is not None:
            body += f"\n\n*Searched: “{evidence.query}”*"

    lines = [f"## {SECTION_TITLES[section]}", "", body]

    # A section that found nothing must not carry a source list. Sources under a
    # "Not found" heading read as though they support something, and the live run
    # that prompted this showed four unrelated news links sitting under exactly
    # that (E-58).
    says_nothing = body.startswith(NOT_FOUND)

    if evidence is not None and not evidence.is_empty and not says_nothing:
        lines += ["", "**Sources**", ""]
        lines += [f"- [{sanitize_title(item.title)}]({item.url})" for item in evidence.items]

    return "\n".join(lines)


def render_report(
    identity: CompanyIdentity,
    result: SynthesisResult,
    evidence: tuple[SectionEvidence, ...] | list[SectionEvidence],
    *,
    generated_on: date,
    synthesis_failed: bool = False,
) -> str:
    """Assemble the finished Markdown document.

    Args:
        identity: Which company was profiled. Rendered at the top so a wrong
            resolution is visible immediately (Q-1).
        result: The model's six sections.
        evidence: The packed evidence, in render order.
        generated_on: The date to stamp on the report.
        synthesis_failed: Whether the all-"Not found" fallback was used. Stated in
            a banner rather than left to look like six genuine findings.

    Returns:
        The complete document: one string, for both the screen and the download.
    """
    by_section = {section.section: section for section in evidence}

    header = [f"# Competitor brief: {identity.name}", ""]

    facts = [f"**Domain:** {identity.domain}"] if identity.domain else []
    facts.append(f"**Generated:** {generated_on.isoformat()}")
    if not identity.supplied_by_user and identity.domain:
        facts.append(
            "*Company identified from search — check the domain above is the one you meant.*"
        )
    header += [" · ".join(facts[:2]), ""]
    if len(facts) > 2:
        header += [facts[2], ""]

    if identity.description:
        header += [f"> {sanitize_title(identity.description)}", ""]

    if synthesis_failed:
        header += [
            "> **Synthesis failed.** The model did not return a usable brief, so the "
            "sections below are empty — but the sources listed are real and were "
            "retrieved for this company.",
            "",
        ]

    body = [_render_section(section, result, by_section.get(section)) for section in SectionKey]

    document = "\n".join(header + ["\n\n".join(body)])

    # Order matters: sanitise first so a hostile construct cannot be reassembled
    # by a later step, then allowlist URLs -- including the ones this renderer
    # just added, which is free, since they came from the retrieved set.
    document = sanitize_markdown(document)
    return strip_unknown_urls(document, known_urls(evidence))


def download_filename(identity: CompanyIdentity, generated_on: date) -> str:
    """Build the download filename for a report."""
    from app.services.normalize import slugify

    return f"{slugify(identity.name)}-brief-{generated_on.isoformat()}.md"
