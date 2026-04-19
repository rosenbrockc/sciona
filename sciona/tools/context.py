"""Synchronous facade for the shared context store.

Wraps the in-memory shared context store for use by agents that
call tools synchronously.
"""

from __future__ import annotations

from typing import Any

from sciona.shared_context import ContextRecord
from sciona.tools._sync import run_sync


class SyncContextStore:
    """Sync wrapper around an async SharedContextStore.

    Provides search, put, and recent operations for agent use.
    Uses the in-memory implementation by default.
    """

    def __init__(self) -> None:
        from sciona.shared_context import InMemorySharedContextStore

        self._store = InMemorySharedContextStore()

    def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        """Search for matching context records."""
        return run_sync(self._store.search(namespace, query, limit=limit))

    def put(
        self,
        namespace: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a context record."""
        run_sync(self._store.put(namespace, text, metadata=metadata))

    def recent(
        self,
        namespace: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        """Return the most recent context records."""
        return run_sync(self._store.recent(namespace, limit=limit))


__all__ = ["SyncContextStore", "ContextRecord"]
