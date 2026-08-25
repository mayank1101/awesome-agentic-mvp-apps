"""A small heuristic scan for prompt-injection phrasing.

Two fields reach the model as user-controlled text: the destination and the
free-text interests field. Neither is likely to carry an attack in ordinary
use, but both are exactly the kind of short free-text box this pattern shows
up in when it does, and the check is cheap enough that skipping it saves
nothing. The fetched search evidence is a second source of untrusted text and
is handled differently -- fenced and labelled for the model rather than
scanned for blocking, the same choice this repo's other search apps make,
because refusing to plan a trip over a phrase in someone's travel blog would
be a worse failure mode than a stray sentence reaching a fenced prompt.
"""

import re

from app.core.config import get_settings
from app.core.exceptions import DestinationBlocked
from app.models.schemas import GuardrailFinding

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all|any|previous|prior|above) instructions",
        r"disregard (all|any|previous|prior|above) instructions",
        r"forget (all|any|previous|prior) instructions",
        r"you are now",
        r"new instructions?:",
        r"system prompt",
        r"reveal (your|the) (system )?prompt",
        r"act as (if|though|an?)\b",
        r"do anything now",
        r"pretend (you|to) (have no|are not)",
        r"override (your|the) (rules|instructions|guidelines)",
    )
)


def scan(texts: dict[str, str]) -> list[GuardrailFinding]:
    """Scan a field-name-to-text mapping for injection phrasing.

    Args:
        texts: Text to scan, keyed by field name.

    Returns:
        One finding per matched phrase. Empty when nothing matched.
    """
    findings: list[GuardrailFinding] = []
    for field, text in texts.items():
        if not text:
            continue
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                findings.append(GuardrailFinding(field=field, pattern=pattern.pattern))
    return findings


def guard(destination: str, interests: str) -> None:
    """Raise if scanning the destination or interests finds a high-severity hit.

    Args:
        destination: The requested destination.
        interests: The free-text interests field.

    Raises:
        DestinationBlocked: A pattern matched and `block_flagged_input` is set.
    """
    settings = get_settings()
    if not settings.guardrails_enabled:
        return

    findings = scan({"destination": destination, "interests": interests})
    if not findings:
        return

    fields = ", ".join(sorted({f.field for f in findings}))
    if settings.block_flagged_input:
        raise DestinationBlocked(
            f"Text that looks like an instruction to the assistant was found in: {fields}.",
            findings=findings,
        )
