"""Reading a pasted backlog into structured features.

The input is deliberately a paste box rather than a form with four numeric
fields per row, because the form is the thing this app replaces. That puts the
burden here: notes arrive as bullets, as numbered lists, as paragraphs separated
by blank lines, and as all three in the same paste, because people build a
backlog by pasting from three places.

The rules are stated rather than clever, so that a user who gets an unexpected
split can look at their text and see why.
"""

import re

from app.core.config import get_settings
from app.core.exceptions import BacklogParseError
from app.core.logging import get_logger
from app.models.schemas import BacklogInput, FeatureIdea

logger = get_logger(__name__)

#: Bullet and numbered-list markers, stripped from the front of a line.
_BULLET = re.compile(r"^\s*(?:[-*•+]|\(?\d{1,2}[.)])\s+")

#: Two or more newlines: a paragraph break, which is one way people separate
#: features.
_BLANK_LINE = re.compile(r"\n\s*\n")

#: Separators between a feature's name and its notes on a single line. Ordered
#: longest-first so an em dash is not mistaken for a hyphen inside a word.
_TITLE_SPLIT = re.compile(r"\s+—\s+|\s+–\s+|\s+-\s+|:\s+")

#: Where a title is cut when the user wrote one long unpunctuated line.
_TITLE_SOFT_CAP = 120


def _strip_bullet(line: str) -> str:
    """Remove a leading bullet or list number from one line."""
    return _BULLET.sub("", line).strip()


def _is_bulleted(line: str) -> bool:
    """Whether a line opens with a bullet or list marker."""
    return bool(_BULLET.match(line))


def _split_blocks(text: str) -> list[list[str]]:
    """Group the paste into one block of lines per feature.

    Two shapes, in order of precedence:

    1. **Blank-line separated.** Each paragraph is one feature: first line its
       name, the rest its notes. This is the shape that supports multi-line
       notes, so it wins when blank lines are present.
    2. **One per line.** No blank lines, so every non-empty line is a feature.

    The awkward middle case is a paste where paragraphs *contain* bullet lists --
    someone's notes doc. A paragraph whose every line is bulleted is treated as a
    run of features rather than one feature with a bulleted body, because a
    bulleted run is overwhelmingly a list of ideas.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    if _BLANK_LINE.search(text):
        blocks: list[list[str]] = []
        for chunk in _BLANK_LINE.split(text):
            lines = [line for line in chunk.split("\n") if line.strip()]
            if not lines:
                continue
            if len(lines) > 1 and all(_is_bulleted(line) for line in lines):
                blocks.extend([line] for line in lines)
            else:
                blocks.append(lines)
        return blocks

    return [[line] for line in text.split("\n") if line.strip()]


def _split_title_and_notes(lines: list[str], max_chars: int) -> tuple[str, str]:
    """Turn one block of lines into a title and the notes that follow it.

    A single line carrying both -- ``"Bulk CSV export — sales asks weekly"`` --
    is split on the first dash or colon. A line with neither is the title, and
    if it is very long it is cut at a word boundary with the tail kept as notes,
    so nothing the user wrote is silently dropped.
    """
    head = _strip_bullet(lines[0])
    rest = " ".join(_strip_bullet(line) for line in lines[1:]).strip()

    parts = _TITLE_SPLIT.split(head, maxsplit=1)
    if len(parts) == 2 and parts[0].strip():
        title, inline_notes = parts[0].strip(), parts[1].strip()
    elif len(head) > _TITLE_SOFT_CAP:
        cut = head.rfind(" ", 0, _TITLE_SOFT_CAP)
        cut = cut if cut > 0 else _TITLE_SOFT_CAP
        title, inline_notes = head[:cut].strip(), head[cut:].strip()
    else:
        title, inline_notes = head, ""

    notes = " ".join(part for part in (inline_notes, rest) if part).strip()

    # The cap is on the feature as a whole, since it is a token budget: a long
    # title spends the same as long notes.
    budget = max(max_chars - len(title), 0)
    if len(notes) > budget:
        notes = notes[:budget].rstrip()
    return title[:_TITLE_SOFT_CAP], notes


def parse_backlog(raw_text: str, product_context: str = "") -> BacklogInput:
    """Parse a pasted backlog into features with stable ids.

    Args:
        raw_text: Whatever the user pasted.
        product_context: The optional one-or-two-line product description.

    Returns:
        The parsed backlog, ids assigned in input order as ``F1``, ``F2``, ...

    Raises:
        BacklogParseError: If nothing usable was found, or the list is longer
            than ``MAX_FEATURES``. The cap is refused rather than truncated:
            silently dropping the tail of someone's backlog and then ranking
            what is left is the worst available behaviour.
    """
    settings = get_settings()
    blocks = _split_blocks(raw_text)
    if not blocks:
        raise BacklogParseError(
            "No features found. Add one feature per line, or one per paragraph."
        )

    if len(blocks) > settings.max_features:
        raise BacklogParseError(
            f"{len(blocks)} features found, and the limit is {settings.max_features}. "
            "The whole list is estimated in one call so the features are calibrated against "
            "each other, which is what caps the length. Split the backlog and rank it in two passes."
        )

    features: list[FeatureIdea] = []
    for index, block in enumerate(blocks, start=1):
        title, notes = _split_title_and_notes(block, settings.max_feature_chars)
        if not title:
            continue
        features.append(FeatureIdea(id=f"F{index}", title=title, notes=notes))

    if not features:
        raise BacklogParseError(
            "No features found. Add one feature per line, or one per paragraph."
        )

    logger.info("Parsed %d feature(s) from %d characters", len(features), len(raw_text))
    return BacklogInput(
        features=features,
        product_context=product_context.strip()[: settings.max_context_chars],
    )
