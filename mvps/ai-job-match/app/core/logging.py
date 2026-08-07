"""Logging setup, with secret and PII redaction applied at the handler.

Streamlit reruns the entry-point script on every interaction, so configuration
has to be idempotent: :func:`configure_logging` is safe to call on each rerun and
only touches handlers the first time.

Redaction lives in a logging filter rather than at each call site because the
call sites that leak are the ones nobody thought about -- a provider error whose
message quotes the request body, a debug line dumping a parsed profile.

This app is different from the repo's others in one way that matters here: its
input is a **real person's resume**. Names, emails, and phone numbers pass
through every layer. Nothing in `app/` logs resume text, and the filter below
catches the email and phone shapes anyway, because "nothing logs it" is a claim
about code that will be edited later.
"""

import logging
import re

from app.core.config import get_settings

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

#: Credential shapes this app handles: Groq keys are ``gsk_``, Mistral keys are
#: a bare 32-char token so they cannot be matched by shape alone -- the generic
#: ``api_key=...`` pattern is what covers those when a third-party error string
#: quotes the request.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:gsk_|sk-)[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|authorization)\b\s*[:=]\s*(?:bearer\s+)?\S+"),
)

#: Contact details that would otherwise reach a log line via an error message
#: quoting resume text back at us.
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?<!\d)(?:\+\d{1,3}[\s\-.]?)?(?:\(\d{2,4}\)[\s\-.]?)?\d{3,5}[\s\-.]\d{4,6}(?!\d)"),
)

_REDACTED = "[redacted]"


def redact(text: str) -> str:
    """Replace credentials and contact details with a placeholder.

    Args:
        text: Arbitrary text, typically a log message or an error string.

    Returns:
        The text with credential-shaped and contact-shaped substrings replaced.
        Never raises -- a redactor that can fail is a redactor that will fail at
        the worst moment.
    """
    redacted = _SECRET_PATTERNS[0].sub(_REDACTED, text)
    redacted = _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}={_REDACTED}", redacted)
    for pattern in _PII_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


class _RedactingFilter(logging.Filter):
    """Rewrites every record's rendered message through :func:`redact`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the record in place and always allow it through.

        Args:
            record: The record about to be emitted.

        Returns:
            Always ``True``: this filter redacts, it does not drop.
        """
        record.msg = redact(str(record.getMessage()))
        record.args = ()
        return True


def configure_logging() -> None:
    """Attach a console handler and apply the configured level, once per process.

    Third-party loggers are pinned to WARNING: the HTTP stack emits one INFO line
    per request, which drowns out this application's own logs at DEBUG.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format=_FORMAT)

    redactor = _RedactingFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(redactor)

    logging.getLogger("app").setLevel(settings.log_level)
    for noisy in ("httpx", "httpcore", "groq", "guardrails", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for `name`, the conventional `__name__` value."""
    return logging.getLogger(name)
