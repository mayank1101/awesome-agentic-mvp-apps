"""Input normalisation for the one free-text field this app has: the app search query.

Runs before anything else sees the string, including the injection scan --
folding first is what makes a literal comparison downstream mean what it says.
"""

import re
import unicodedata

#: Invisible or bidirectional-override characters that change how text is read
#: without being readable themselves.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")

_ALLOWED_CONTROL = {"\t", "\n", "\r"}
_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_query(raw: str, *, max_chars: int = 200) -> str:
    """Fold, clean, and cap a search query.

    Args:
        raw: Exactly what the user typed.
        max_chars: Longer input is truncated rather than rejected.

    Returns:
        The normalised query, possibly empty if nothing usable was in it.
    """
    folded = unicodedata.normalize("NFKC", raw)
    folded = _INVISIBLE.sub("", folded)
    folded = "".join(
        ch for ch in folded if ch in _ALLOWED_CONTROL or unicodedata.category(ch)[0] != "C"
    )
    collapsed = _WHITESPACE_RUN.sub(" ", folded).strip()
    return collapsed[:max_chars].strip()


def is_meaningful_query(query: str) -> bool:
    """Whether a normalised query could plausibly identify an app."""
    return any(ch.isalnum() for ch in query)
