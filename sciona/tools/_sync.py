"""Sync wrapper for async coroutines used by agent-callable tools."""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

_T = TypeVar("_T")


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine synchronously.

    Uses asyncio.run() when no event loop is running. Falls back to a
    thread-pool executor if called from within an existing loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
