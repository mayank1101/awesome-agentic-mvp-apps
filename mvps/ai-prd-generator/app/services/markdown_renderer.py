"""Rendering a structured PRD to Markdown.

Kept apart from the models so other output formats -- docx, PDF -- can sit
beside it later without the schema growing rendering concerns.
"""

from app.models.schemas import PRDDocument


def render_markdown(prd: PRDDocument) -> str:
    """Assemble a document into one Markdown string.

    The title becomes the sole ``#`` heading and each section a ``##``, so the
    result has exactly one top-level heading and imports cleanly into editors
    that build a table of contents from heading levels.

    Args:
        prd: The finished document.

    Returns:
        Markdown text, ready to download.
    """
    parts = [f"# {prd.title}\n"]
    for section in prd.sections:
        parts.append(f"## {section.title}\n\n{section.content.strip()}\n")
    return "\n".join(parts)


def count_words(markdown: str) -> int:
    """Rough word count of rendered Markdown, used for the size estimate."""
    return len(markdown.split())
