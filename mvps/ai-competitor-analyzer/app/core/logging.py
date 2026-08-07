"""Logging setup, with secret redaction applied at the handler.

Streamlit reruns the entry-point script on every interaction, so configuration
has to be idempotent: :func:`configure_logging` is safe to call on each rerun and
only touches handlers the first time.

Redaction lives in a logging filter rather than at each call site because the
call sites that leak keys are the ones nobody thought about -- a provider error
whose message quotes the request, a debug line dumping settings (E-47). A filter
covers those too.
"""

import logging
import re

from app.core.config import get_settings

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

#: Patterns for the credential shapes this app handles. Groq keys are ``gsk_``,
#: OpenAI and OpenRouter are ``sk-``, and Tavily is ``tvly-``; the generic
#: fallback catches ``api_key=...`` style leaks from third-party error strings.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:gsk_|sk-|tvly-)[A-Za-z0-9_\-]{8,}"),
    # ``Bearer`` is matched explicitly: without it the value pattern stops at the
    # scheme word and leaves the token itself in the log, which is the whole
    # thing this is here to prevent.
    re.compile(r"(?i)\b(api[_-]?key|token|secret|authorization)\b\s*[:=]\s*(?:bearer\s+)?\S+"),
)

_REDACTED = "[redacted]"


def redact_secrets(text: str) -> str:
    """Replace anything that looks like a credential with a placeholder.

    Args:
        text: Arbitrary text, typically a log message or an error string.

    Returns:
        The text with credential-shaped substrings replaced. Never raises -- a
        redactor that can fail is a redactor that will fail at the worst moment.
    """
    redacted = _SECRET_PATTERNS[0].sub(_REDACTED, text)
    return _SECRET_PATTERNS[1].sub(
        lambda m: f"{m.group(1)}={_REDACTED}",
        redacted,
    )


class _RedactingFilter(logging.Filter):
    """Rewrites every record's rendered message through :func:`redact_secrets`."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the record in place and always allow it through.

        Args:
            record: The record about to be emitted.

        Returns:
            Always ``True``: this filter redacts, it does not drop.
        """
        record.msg = redact_secrets(str(record.getMessage()))
        record.args = ()
        return True


def configure_logging() -> None:
    """Attach a console handler and apply the configured level, once per process.

    Third-party loggers are pinned to WARNING: the HTTP stack under the agent
    emits one INFO line per request, which drowns out this application's own logs
    at DEBUG.
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
    for noisy in ("httpx", "httpcore", "openai", "tavily"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for `name`, the conventional `__name__` value."""
    return logging.getLogger(name)
