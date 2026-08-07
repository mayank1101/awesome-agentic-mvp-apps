"""Sync/async bridge.

Agent Framework is async-only; a Streamlit script body is synchronous. Rather
than starting a fresh event loop per call, everything runs on one long-lived loop
in a daemon thread.

That is not incidental: the HTTP client underneath the cached chat client keeps a
connection pool bound to the loop that created it. A per-call ``asyncio.run()``
would work for the opening question and fail on the first follow-up, once the
loop that owns the pool has been closed.

The loop is a process-wide singleton. Streamlit reruns the script on every
interaction, but module state survives reruns, so the loop is created once and
reused for the life of the server.
"""

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from queue import Queue
from typing import TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_LOOP_THREAD_NAME = "interview-agent-loop"

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()

#: Sentinel pushed onto the queue when the producer is finished. A dedicated
#: object is used so that a legitimate ``None`` item could never end the stream.
_DONE = object()


def get_loop() -> asyncio.AbstractEventLoop:
    """Return the shared event loop, starting its thread on first use.

    Public because it is the process-wide place for work that outlives a single
    script run, and because tests need a handle on it.
    """
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
    return asyncio.run_coroutine_threadsafe(coro, get_loop()).result()


def iter_sync(agen_factory: Callable[[], AsyncIterator[T]]) -> Iterator[T]:
    """Drain an async iterator into a sync generator, item by item.

    The factory is called on the loop thread rather than taking a ready-made
    iterator, so the async generator is created and consumed on the same loop.

    The queue is deliberately unbounded: a bounded one would have the producer
    block inside the event loop thread, stalling every other task on that loop.
    An interview question is a few hundred bytes, so the memory ceiling is not a
    concern.

    Args:
        agen_factory: Zero-argument callable returning the async iterator.

    Yields:
        Items in the order the async iterator produced them.

    Raises:
        Exception: Anything the async iterator raised, re-raised on the caller's
            thread once the preceding items have been yielded.
    """
    queue: Queue = Queue()

    async def pump() -> None:
        try:
            async for item in agen_factory():
                queue.put(item)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller verbatim
            queue.put(exc)
        finally:
            queue.put(_DONE)

    asyncio.run_coroutine_threadsafe(pump(), get_loop())

    while True:
        item = queue.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item
