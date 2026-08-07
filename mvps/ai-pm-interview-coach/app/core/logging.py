"""Logging setup.

Streamlit reruns the entry-point script on every interaction, so configuration
has to be idempotent: `configure_logging()` is safe to call on each rerun and
only touches handlers the first time.
"""

import logging

from app.core.config import get_settings

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def configure_logging() -> None:
    """Attach a console handler and apply the configured level, once per process.

    Third-party loggers are pinned to WARNING: the HTTP stack under the agent
    emits one INFO line per request, which drowns out this application's own
    logs at DEBUG.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format=_FORMAT)
    logging.getLogger("app").setLevel(settings.log_level)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for `name`, the conventional `__name__` value."""
    return logging.getLogger(name)
