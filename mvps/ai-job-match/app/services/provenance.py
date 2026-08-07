"""The fabrication guard: checking a rewrite against the original resume.

The rewrite prompt tells the model not to invent facts. This module is why that
sentence is worth anything. An instruction is a preference; this is a check, and
it is the one hard guarantee the app makes.

Three classes of fact are checkable mechanically, and they happen to be the three
that do real damage when invented:

* **Numbers.** "Reduced latency by 40%" on a resume that never mentioned latency
  or 40 is the classic tailoring failure, and it is the one an interviewer finds
  in the first five minutes.
* **Named things.** Employers, tools, certifications, universities. A token that
  is capitalised mid-sentence, or that carries internal case or punctuation
  (`k8s`, `C++`, `Node.js`, `PostgreSQL`), is a name -- and a name that is not in
  the original resume was invented.
* **Contact details.** An email or URL that the candidate never wrote.

What this cannot check is *prose*: "led the migration" where the original said
"contributed to the migration" is a real exaggeration and no string comparison
will catch it. That limit is stated in the README rather than papered over. The
mitigation is that the original and the rewrite are shown side by side, so the
person whose resume it is reviews the diff before sending it.

The check is deliberately biased toward false positives over false negatives. A
flagged line the user glances at and approves costs a second; an invented
employer that ships costs an interview.
"""

import re
from dataclasses import dataclass
from typing import Literal

from app.core.logging import get_logger

logger = get_logger(__name__)

ViolationKind = Literal["number", "name", "contact"]

#: Markdown syntax stripped before analysis, so `**Go**` is compared as `Go`.
_MARKDOWN_SYNTAX = re.compile(r"[*_`#>|]+")

#: A number, with the punctuation that decorates one: `40%`, `1,200`, `$3.5M`.
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?[kKmMbB]?")

#: A word-ish token, keeping the punctuation that lives inside real names:
#: `Node.js`, `C++`, `scikit-learn`, `CI/CD`.
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#./&_-]*")

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s)\]]+", re.IGNORECASE)

#: Sentence-ish boundaries, used to decide whether a capitalised word is
#: capitalised because it is a name or because it starts a sentence.
_SENTENCE_BREAK = re.compile(r"[.!?:;]\s+$|^\s*$")

#: List markers and leading punctuation. Stripped when deciding whether a word is
#: sentence-initial: the first word of a bullet is capitalised for the same
#: grammatical reason as the first word of a sentence, and treating "- Delivered"
#: as a name flags every bullet in the document.
_LEADING_PUNCTUATION = " \t-*+•>."

#: Words that are capitalised in ordinary resume prose and are not names. Short
#: on purpose -- the sentence-position rule below does most of the work, and a
#: long allowlist is exactly how a guard stops guarding.
_ALLOWED = frozenset(
    # One string, split at import time: written as prose because a formatter
    # renders a literal list of 70 words as 70 lines, which nobody reads.
    """
    i a an and or the of in on at to for with by from as is are was were be am
    jan feb mar apr may jun jul aug sep sept oct nov dec
    january february march april june july august september october november december
    present current now today ongoing
    summary skills experience education projects certifications profile contact
    professional technical achievements highlights awards publications languages
    interests references volunteer leadership objective about
    name email phone location links headline
    """.split()
)


@dataclass(frozen=True)
class Violation:
    """One fact in the rewrite that is absent from the original resume.

    Attributes:
        kind: Which check caught it.
        text: The offending token, as it appears in the rewrite.
        context: The line it appeared on, trimmed, so the user can see it in
            place rather than as a bare word.
    """

    kind: ViolationKind
    text: str
    context: str


def _strip_markdown(text: str) -> str:
    """Remove Markdown syntax characters without changing word boundaries."""
    return _MARKDOWN_SYNTAX.sub(" ", text)


def _normalize_number(token: str) -> str:
    """Reduce a number to a comparable form: `$1,200.00` and `1200` both to `1200`."""
    cleaned = token.strip("$%").replace(",", "").lower().rstrip("kmb")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned.rstrip(".")


def _original_numbers(text: str) -> set[str]:
    """Every number in the original, normalised for comparison."""
    return {_normalize_number(token) for token in _NUMBER.findall(text)}


def _original_words(text: str) -> set[str]:
    """Every word in the original, lowercased, for name comparison."""
    return {token.lower().strip(".-/") for token in _WORD.findall(text)}


def squash(text: str) -> str:
    """Reduce text to lowercase alphanumerics, dropping everything else.

    PDF text extraction splits words at kerning pairs -- a real resume yielded
    ``Indian Institute of T echnology`` -- and glues template icon names onto the
    values they decorate. Both make exact token matching report a word the
    candidate plainly wrote as invented, and in strict mode that refuses their
    rewrite over the extractor's spacing.

    Squashing both sides and testing containment survives every artefact of that
    shape, because it removes exactly the thing the extractor gets wrong.
    """
    return re.sub(r"[^a-z0-9]+", "", text.lower())


#: Shortest token allowed to clear the check by squashed containment. Below this,
#: containment is nearly meaningless -- "go" appears inside "google" -- so short
#: tokens must match a real word in the original or be flagged.
_MIN_SQUASH_LENGTH = 5


def _is_name_like(token: str, *, sentence_start: bool) -> bool:
    """Whether a token should be treated as a name that needs provenance.

    Args:
        token: The token as written in the rewrite.
        sentence_start: Whether it is the first word of a line or sentence, where
            capitalisation carries no information.

    Returns:
        True for tokens that look like proper nouns, products, or technologies.
    """
    if len(token) < 2 or token.lower() in _ALLOWED:
        return False

    has_internal_case = any(character.isupper() for character in token[1:])
    has_symbol_or_digit = any(character in "+#./&_-" or character.isdigit() for character in token)

    if has_internal_case or has_symbol_or_digit:
        return True
    return token[0].isupper() and not sentence_start


def check(original: str, tailored: str) -> list[Violation]:
    """Find every fact in `tailored` that is not present in `original`.

    Args:
        original: The resume text as extracted from the uploaded PDF, verbatim.
        tailored: The rewritten resume, as Markdown.

    Returns:
        Violations in the order they appear in the rewrite, deduplicated by the
        offending token. Empty means the rewrite introduced no new checkable
        fact.
    """
    known_numbers = _original_numbers(original)
    known_words = _original_words(original)
    known_emails = {match.lower() for match in _EMAIL.findall(original)}
    known_urls = {match.lower().rstrip("/") for match in _URL.findall(original)}
    # The fallback for everything PDF extraction mangles: see :func:`squash`.
    squashed_original = squash(original)

    def is_known(token: str, *, exact: set[str], minimum: int = 1) -> bool:
        """Whether `token` appears in the original, allowing for extraction damage."""
        cleaned = token.lower().strip(".-/")
        if cleaned in exact:
            return True
        squashed = squash(cleaned)
        return len(squashed) >= minimum and squashed in squashed_original

    violations: list[Violation] = []
    seen: set[tuple[str, str]] = set()

    def record(kind: ViolationKind, token: str, line: str) -> None:
        key = (kind, token.lower())
        if key in seen:
            return
        seen.add(key)
        violations.append(Violation(kind=kind, text=token, context=line.strip()[:160]))

    for raw_line in tailored.splitlines():
        line = _strip_markdown(raw_line)
        if not line.strip():
            continue

        for email in _EMAIL.findall(line):
            if not is_known(email, exact=known_emails):
                record("contact", email, raw_line)

        for url in _URL.findall(line):
            if not is_known(url.rstrip("/"), exact=known_urls):
                record("contact", url, raw_line)

        for number in _NUMBER.findall(line):
            if _normalize_number(number) and _normalize_number(number) not in known_numbers:
                record("number", number, raw_line)

        # Contact details are checked whole above; strip them before the word
        # pass so their internal tokens are not re-reported as names.
        prose = _URL.sub(" ", _EMAIL.sub(" ", line))
        for match in _WORD.finditer(prose):
            # Edge punctuation off first. The token pattern deliberately keeps
            # the characters that live inside real names (`Node.js`, `CI/CD`),
            # which means a word ending a sentence arrives as "applications." --
            # and a trailing full stop would otherwise read as exactly the
            # evidence that a token is a technical name. That misfire refused a
            # real rewrite over the word "applications".
            token = match.group(0).strip(".,;:!?()[]-/&_")
            if not token:
                continue
            preceding = prose[: match.start()]
            sentence_start = bool(_SENTENCE_BREAK.search(preceding)) or not preceding.strip(
                _LEADING_PUNCTUATION
            )
            if not _is_name_like(token, sentence_start=sentence_start):
                continue
            if not is_known(token, exact=known_words, minimum=_MIN_SQUASH_LENGTH):
                record("name", token, raw_line)

    if violations:
        logger.warning(
            "Fabrication guard found %d unsupported fragment(s): %s",
            len(violations),
            ", ".join(sorted({v.kind for v in violations})),
        )
    return violations


def describe(violations: list[Violation]) -> list[str]:
    """Render violations as lines a user can read on screen."""
    labels = {
        "number": "number not in your resume",
        "name": "name or tool not in your resume",
        "contact": "contact detail not in your resume",
    }
    return [f"“{v.text}” — {labels[v.kind]} (in: {v.context})" for v in violations]
