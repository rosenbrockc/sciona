"""Checkpointer factory for the decomposition graph.

Returns AsyncPostgresSaver when a PostgreSQL URI is available,
falls back to MemorySaver otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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

            saver = AsyncPostgresSaver.from_conn_string(postgres_uri)
            await saver.setup()
            yield saver
            return
        except Exception:
            logger.warning(
                "Failed to connect to PostgreSQL at %s; falling back to MemorySaver",
                postgres_uri,
                exc_info=True,
            )

    yield MemorySaver()
