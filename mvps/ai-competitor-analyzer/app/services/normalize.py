"""Input normalisation: the app's entire boundary with what a user typed.

Three jobs, all pure functions:

* :func:`normalize_name` — NFKC-fold and strip the characters that are invisible
  or that impersonate other characters (E-02, E-03).
* :func:`normalize_domain` — reduce anything domain-shaped to a bare host, and
  refuse anything that is not `http(s)` (E-05).
* :func:`slugify` — a filename for the download that survives every alphabet.

Normalisation runs before *anything* else sees the string, including the
injection scan and the fence-marker strip. A Cyrillic "о" in "Nоtion" both
defeats the search (no engine matches it) and would walk straight around a
literal comparison in the fence stripper, so folding first is what makes those
later checks mean what they say.

NFKC only: no transliteration, no ASCII coercion. `日本電気` is a legitimate
company name, and an app that mangles it into `Ri Ben Dian Qi` is broken for
every non-Latin market.
"""

import re
import unicodedata
from urllib.parse import urlsplit

#: Characters that are invisible or that change how text is read without being
#: readable themselves: zero-width joiners, bidirectional overrides, BOM.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")

#: Everything in Unicode category C (control, format, surrogate, unassigned)
#: except the whitespace that ordinary text legitimately contains.
_ALLOWED_CONTROL = {"\t", "\n", "\r"}

_WHITESPACE_RUN = re.compile(r"\s+")

#: Schemes the domain field accepts. Anything else -- `javascript:`, `file:`,
#: `data:` -- is rejected outright rather than coerced into something harmless,
#: because a value that had to be sanitised to be safe is a value the user should
#: be told about.
_ALLOWED_SCHEMES = frozenset({"http", "https", ""})

#: A hostname's labels may contain hyphens but may not begin or end with one, and
#: the final label is alphabetic. The lookarounds are what reject `-.com`.
_HOST_PATTERN = re.compile(r"^(?=.{1,253}$)((?!-)[a-z0-9-]{1,63}(?<!-)\.)+[a-z]{2,63}$")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


class InvalidDomainError(ValueError):
    """The domain field held something that is not a usable web host."""


def normalize_name(raw: str, *, max_chars: int) -> str:
    """Fold, clean, and cap a company name.

    Args:
        raw: Exactly what the user typed.
        max_chars: Cap from settings. Longer names are truncated rather than
            rejected -- someone pasting a full legal entity name should still get
            a report (E-04).

    Returns:
        The normalised name, possibly empty if the input held nothing usable.
    """
    folded = unicodedata.normalize("NFKC", raw)
    folded = _INVISIBLE.sub("", folded)
    folded = "".join(
        ch for ch in folded if ch in _ALLOWED_CONTROL or unicodedata.category(ch)[0] != "C"
    )
    collapsed = _WHITESPACE_RUN.sub(" ", folded).strip()
    return collapsed[:max_chars].strip()


def is_meaningful_name(name: str) -> bool:
    """Whether a normalised name could plausibly identify a company.

    A name of pure punctuation resolves to nothing, so the run is refused before
    a credit is spent (E-01).

    Args:
        name: An already-normalised name.

    Returns:
        True when at least one letter or digit is present.
    """
    return any(ch.isalnum() for ch in name)


def normalize_domain(raw: str | None) -> str | None:
    """Reduce a domain, URL, or host to a bare registrable host.

    Accepts `example.com`, `www.Example.com/pricing?x=1`, `https://example.com`,
    and internationalised hosts. Rejects any other scheme (E-05).

    Args:
        raw: Whatever was in the domain field; may be `None` or blank.

    Returns:
        The lowercased, IDN-encoded host without `www.`, or `None` when the field
        was empty.

    Raises:
        InvalidDomainError: The value carries a disallowed scheme, or does not
            look like a host at all.
    """
    if raw is None:
        return None

    candidate = unicodedata.normalize("NFKC", raw).strip()
    candidate = _INVISIBLE.sub("", candidate)
    if not candidate:
        return None

    # urlsplit only recognises a scheme when `//` follows, so a bare host and a
    # `javascript:` payload both need the explicit check below.
    scheme, _, rest = candidate.partition(":")
    if rest and scheme.lower() not in _ALLOWED_SCHEMES:
        raise InvalidDomainError(
            f"{scheme}: links are not accepted here — enter a domain like example.com"
        )

    if "//" not in candidate:
        candidate = f"//{candidate}"

    host = urlsplit(candidate).netloc or ""
    host = host.split("@")[-1].split(":")[0]  # drop credentials and port
    host = host.lower().removeprefix("www.").rstrip(".")

    if not host:
        raise InvalidDomainError("that does not look like a domain — try example.com")

    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise InvalidDomainError("that domain could not be interpreted") from exc

    if not _HOST_PATTERN.match(host):
        raise InvalidDomainError("that does not look like a domain — try example.com")

    return host


def registrable_host(url: str) -> str:
    """Return the registrable domain of a URL, for comparing hits to a company.

    Deliberately naive -- the last two labels. A public-suffix list would be more
    correct for `example.co.uk`, and is a dependency this app does not need: the
    comparison is used to prefer a company's own pages, never to make a security
    decision.

    Args:
        url: Any absolute URL.

    Returns:
        The lowercased host without `www.`, reduced to its last two labels.
    """
    netloc = urlsplit(url).netloc.lower().removeprefix("www.").split(":")[0]
    parts = [part for part in netloc.split(".") if part]
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


def slugify(text: str, *, fallback: str = "competitor") -> str:
    """Build a filename-safe slug.

    Args:
        text: Any text, typically the company name.
        fallback: Used when nothing survives — a non-Latin name slugs to nothing,
            and a download called `-.md` helps no one (E-55).

    Returns:
        A lowercase ASCII slug with single hyphens and no leading or trailing
        separator.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_only).strip("-")
    return slug or fallback
