"""The sync/async bridge.

The properties worth pinning are the ones that would fail *intermittently* if
broken: that a second call reuses the same loop (a fresh loop per call breaks the
connection pool on the second turn), that stream items arrive in order, and that
an exception surfaces on the caller's thread only after the items before it.
"""

import asyncio
import threading

import pytest

from app.services.async_bridge import get_loop, iter_sync, run_sync


def test_run_sync_returns_the_result():
    async def work() -> int:
        return 21 * 2

    assert run_sync(work()) == 42


def test_run_sync_reraises_on_the_calling_thread():
    class Boom(Exception):
        pass

    async def work() -> None:
        raise Boom("from the loop")

    with pytest.raises(Boom, match="from the loop"):
        run_sync(work())


def test_loop_is_reused_across_calls():
    # The whole reason this module exists: the HTTP connection pool under the
    # cached chat client is bound to the loop that created it, so a per-call loop
    # works for the opening question and fails on the first follow-up.
    async def work() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    assert run_sync(work()) is run_sync(work())


def test_loop_runs_off_the_calling_thread():
    async def work() -> int:
        return threading.get_ident()

    assert run_sync(work()) != threading.get_ident()


def test_iter_sync_yields_in_order():
    async def stream():
        for chunk in ["a", "b", "c"]:
            yield chunk

    assert list(iter_sync(stream)) == ["a", "b", "c"]


def test_iter_sync_handles_an_empty_stream():
    async def stream():
        return
        yield  # pragma: no cover - makes this an async generator

    assert list(iter_sync(stream)) == []


def test_iter_sync_yields_preceding_items_before_raising():
    class Boom(Exception):
        pass

    async def stream():
        yield "first"
        yield "second"
        raise Boom("late failure")

    received: list[str] = []
    with pytest.raises(Boom, match="late failure"):
        for chunk in iter_sync(stream):
            received.append(chunk)

    # A streamed question that fails mid-flight must not lose the text already
    # rendered to the candidate.
    assert received == ["first", "second"]


def test_iter_sync_creates_the_generator_on_the_loop_thread():
    # The factory is called on the loop thread rather than taking a ready-made
    # iterator, so creation and consumption share one loop.
    seen: list[int] = []

    async def stream():
        seen.append(threading.get_ident())
        yield "x"

    list(iter_sync(stream))

    assert seen and seen[0] != threading.get_ident()


def test_get_loop_is_running():
    assert get_loop().is_running()
