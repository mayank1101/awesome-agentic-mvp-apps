"""Logging setup, with secret redaction applied at the handler.

Streamlit reruns the entry-point script on every interaction, so configuration
has to be idempotent: :func:`configure_logging` is safe to call on each rerun
and only touches handlers the first time.

Redaction lives in a logging filter rather than at each call site because the
call sites that leak are the ones nobody thought about -- a provider error
whose message quotes the request body.
"""

import logging
import re

from app.core.config import get_settings

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

#: Groq keys are `gsk_`; Tavily keys are `tvly-`; the generic `api_key=...`
#: pattern covers other providers' shapes when an error string quotes the
#: request.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:gsk_|sk-|tvly-)[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|authorization)\b\s*[:=]\s*(?:bearer\s+)?\S+"),
)

_REDACTED = "[redacted]"


def redact(text: str) -> str:
    """Replace credential-shaped substrings with a placeholder.

    Args:
        text: Arbitrary text, typically a log message or an error string.

    Returns:
        The text with credential-shaped substrings replaced. Never raises -- a
        redactor that can fail is a redactor that will fail at the worst moment.
    """
    redacted = _SECRET_PATTERNS[0].sub(_REDACTED, text)
    return _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}={_REDACTED}", redacted)


class _RedactingFilter(logging.Filter):
    """Rewrites every record's rendered message through :func:`redact`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the record in place and always allow it through."""
        record.msg = redact(str(record.getMessage()))
        record.args = ()
        return True


def configure_logging() -> None:
    """Attach a console handler and apply the configured level, once per process.

    Third-party loggers are pinned to WARNING: the HTTP stack emits one INFO
    line per request, which drowns out this application's own logs.
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
    for noisy in ("httpx", "httpcore", "groq", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for `name`, the conventional `__name__` value."""
    return logging.getLogger(name)
