"""Checkpointer factory for the decomposition graph.

Returns AsyncPostgresSaver when a PostgreSQL URI is available,
falls back to MemorySaver otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


@asynccontextmanager
async def create_checkpointer(
    postgres_uri: str | None = None,
) -> AsyncIterator[BaseCheckpointSaver]:
    """Yield a checkpoint saver, preferring Postgres when a URI is given.

    Falls back to MemorySaver on any failure or when URI is empty/None.
    Never raises — always yields a usable checkpointer.
    """
    if postgres_uri:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except Exception:
            logger.warning(
                "Failed to connect to PostgreSQL at %s; falling back to MemorySaver",
                postgres_uri,
                exc_info=True,
            )
        else:
            # Only fallback on connection/setup failures. Exceptions raised by
            # the caller while using the yielded saver must propagate normally.
            stack = AsyncExitStack()
            try:
                saver = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(postgres_uri)
                )
            except Exception:
                await stack.aclose()
                logger.warning(
                    "Failed to connect to PostgreSQL at %s; falling back to MemorySaver",
                    postgres_uri,
                    exc_info=True,
                )
            else:
                try:
                    await saver.setup()
                except Exception:
                    await stack.aclose()
                    logger.warning(
                        "Failed to connect to PostgreSQL at %s; falling back to MemorySaver",
                        postgres_uri,
                        exc_info=True,
                    )
                else:
                    try:
                        yield saver
                        return
                    finally:
                        await stack.aclose()

    yield MemorySaver()
