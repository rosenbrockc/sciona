"""Shared context primitives for cross-agent prompt augmentation."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class ContextRecord:
    """A single memory record used for prompt-time retrieval."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SharedContextStore(Protocol):
    """Protocol for shared context stores."""

    async def put(
        self,
        namespace: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a context record into a namespace."""
        ...

    async def recent(self, namespace: str, *, limit: int = 5) -> list[ContextRecord]:
        """Return the most recent records for a namespace."""
        ...

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        """Return the best-matching records for a query within a namespace."""
        ...


class InMemorySharedContextStore:
    """Simple append-only in-memory context store.

    This is intended as a default local backend and test utility.
    """

    def __init__(self, *, max_records_per_namespace: int = 500) -> None:
        self._data: dict[str, list[ContextRecord]] = {}
        self._lock = asyncio.Lock()
        self._max = max_records_per_namespace

    async def put(
        self,
        namespace: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not namespace.strip() or not text.strip():
            return
        rec = ContextRecord(text=text.strip(), metadata=dict(metadata or {}))
        async with self._lock:
            bucket = self._data.setdefault(namespace, [])
            bucket.append(rec)
            if len(bucket) > self._max:
                self._data[namespace] = bucket[-self._max :]

    async def recent(self, namespace: str, *, limit: int = 5) -> list[ContextRecord]:
        async with self._lock:
            bucket = list(self._data.get(namespace, []))
        if limit <= 0:
            return []
        return list(reversed(bucket[-limit:]))

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        if limit <= 0:
            return []
        q_tokens = _tokenize(query)
        async with self._lock:
            bucket = list(self._data.get(namespace, []))
        if not bucket:
            return []
        if not q_tokens:
            return list(reversed(bucket[-limit:]))

        scored: list[tuple[float, ContextRecord]] = []
        n = len(bucket)
        for idx, rec in enumerate(bucket):
            base = _overlap_score(q_tokens, _tokenize(rec.text))
            if base <= 0.0:
                continue
            # Slight recency bias: more recent entries rank higher on ties.
            recency = (idx + 1) / max(n, 1)
            scored.append((base + (0.01 * recency), rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [rec for _, rec in scored[:limit]]


def format_context_block(
    title: str,
    records: list[ContextRecord],
    *,
    max_chars: int = 900,
) -> str:
    """Render shared context records into a compact prompt block."""
    if not records:
        return ""
    lines = [f"## {title}"]
    for rec in records:
        text = " ".join(rec.text.split())
        if len(text) > 220:
            text = text[:217] + "..."
        lines.append(f"- {text}")
    block = "\n".join(lines)
    if len(block) > max_chars:
        return block[: max_chars - 3] + "..."
    return block


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap_score(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)
