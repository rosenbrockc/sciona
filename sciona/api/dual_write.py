"""Dual-write helpers for the Phase 5 cutover period."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_METRICS: dict[str, int] = {
    "dual_write_attempts": 0,
    "dual_write_failures": 0,
    "dual_write_latency_ms_total": 0,
    "dual_write_mutating_queries": 0,
}

_MUTATING_SQL = re.compile(
    r"^\s*(?:WITH\b.*?\b(?:INSERT|UPDATE|DELETE|MERGE|TRUNCATE)\b|"
    r"INSERT|UPDATE|DELETE|MERGE|TRUNCATE)\b",
    re.IGNORECASE | re.DOTALL,
)


def get_dual_write_metrics() -> dict[str, int]:
    """Return a copy of the in-memory dual-write counters."""
    return dict(_METRICS)


def reset_dual_write_metrics() -> None:
    """Reset the in-memory counters. Primarily useful in tests."""
    for key in _METRICS:
        _METRICS[key] = 0


def is_mutating_sql(query: str) -> bool:
    """Return True when a SQL string is likely to mutate state."""
    return bool(_MUTATING_SQL.match(query or ""))


class DualWriteConnection:
    """Proxy an asyncpg connection and mirror mutating queries to a second DB."""

    def __init__(self, primary_conn: Any, supabase_conn: Any) -> None:
        self._primary = primary_conn
        self._supabase = supabase_conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)

    async def _mirror_to_supabase(
        self,
        method: str,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any | None:
        _METRICS["dual_write_attempts"] += 1
        started = time.monotonic()
        try:
            fn = getattr(self._supabase, method)
            return await fn(query, *args, **kwargs)
        except Exception:
            _METRICS["dual_write_failures"] += 1
            logger.exception(
                "dual-write mirror failed for %s | query_prefix=%s",
                method,
                (query or "")[:120],
            )
            return None
        finally:
            _METRICS["dual_write_latency_ms_total"] += int(
                (time.monotonic() - started) * 1000
            )

    async def fetch(self, query: str, *args: Any, **kwargs: Any) -> Any:
        result = await self._primary.fetch(query, *args, **kwargs)
        if is_mutating_sql(query):
            _METRICS["dual_write_mutating_queries"] += 1
            await self._mirror_to_supabase("fetch", query, *args, **kwargs)
        return result

    async def fetchrow(self, query: str, *args: Any, **kwargs: Any) -> Any:
        result = await self._primary.fetchrow(query, *args, **kwargs)
        if is_mutating_sql(query):
            _METRICS["dual_write_mutating_queries"] += 1
            await self._mirror_to_supabase("fetchrow", query, *args, **kwargs)
        return result

    async def fetchval(self, query: str, *args: Any, **kwargs: Any) -> Any:
        result = await self._primary.fetchval(query, *args, **kwargs)
        if is_mutating_sql(query):
            _METRICS["dual_write_mutating_queries"] += 1
            await self._mirror_to_supabase("fetchval", query, *args, **kwargs)
        return result

    async def execute(self, query: str, *args: Any, **kwargs: Any) -> Any:
        result = await self._primary.execute(query, *args, **kwargs)
        if is_mutating_sql(query):
            _METRICS["dual_write_mutating_queries"] += 1
            await self._mirror_to_supabase("execute", query, *args, **kwargs)
        return result
