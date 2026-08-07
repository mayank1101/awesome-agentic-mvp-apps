"""Sync/async bridge.

Agent Framework is async-only; a Streamlit script body is synchronous. Rather
than starting a fresh event loop per call, everything runs on one long-lived
loop in a daemon thread.

That is not incidental: the HTTP client underneath the cached chat client
keeps a connection pool bound to the loop that created it. A per-call
``asyncio.run()`` would work for the first call and fail on the second, once
the loop that owns the pool has been closed.

The loop is a process-wide singleton. Streamlit reruns the script on every
interaction, but module state survives reruns, so the loop is created once and
reused for the life of the server. Ported as-is from `ai-competitor-analyzer`.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_LOOP_THREAD_NAME = "review-tracker-agent-loop"

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the shared event loop, starting its thread on first use."""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            logger.debug("Starting background event loop")
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name=_LOOP_THREAD_NAME, daemon=True).start()
        return _loop


def run_sync(coro: Coroutine[object, object, T]) -> T:
    """Run one coroutine on the shared loop and block until it returns.

    Args:
        coro: The coroutine to execute.

    Returns:
        Whatever the coroutine returned.

    Raises:
        Exception: Anything the coroutine raised, re-raised on the caller's
            thread.
    """
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()
