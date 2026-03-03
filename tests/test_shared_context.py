"""Tests for shared-context stores, factory, and metrics."""

from __future__ import annotations

import pytest

from ageom.shared_context import (
    InMemorySharedContextStore,
    InstrumentedSharedContextStore,
    PostgresSharedContextStore,
    SharedContextMetrics,
    create_shared_context_store,
    format_context_block,
)


class TestSharedContextMetrics:
    @pytest.mark.asyncio
    async def test_instrumented_store_records_search_put_and_injection(self):
        metrics = SharedContextMetrics()
        store = InstrumentedSharedContextStore(InMemorySharedContextStore(), metrics)

        await store.put("ns/a", "alpha beta gamma", metadata={"k": "v"})
        rows = await store.search("ns/a", "alpha", limit=3)
        block = format_context_block("Shared Context", rows, metrics=metrics)

        assert "alpha beta gamma" in block
        snap = metrics.snapshot()
        assert snap["puts_total"] == 1
        assert snap["searches_total"] == 1
        assert snap["search_hits"] == 1
        assert snap["injected_blocks"] == 1
        assert snap["injected_chars"] > 0

    @pytest.mark.asyncio
    async def test_search_miss_is_counted(self):
        metrics = SharedContextMetrics()
        store = InstrumentedSharedContextStore(InMemorySharedContextStore(), metrics)
        await store.put("ns/a", "alpha beta gamma")
        rows = await store.search("ns/a", "delta", limit=3)
        assert rows == []
        snap = metrics.snapshot()
        assert snap["searches_total"] == 1
        assert snap["search_misses"] == 1
        assert float(snap["search_hit_rate"]) == 0.0


class TestSharedContextFactory:
    @pytest.mark.asyncio
    async def test_auto_without_postgres_uses_memory(self):
        metrics = SharedContextMetrics()
        store = await create_shared_context_store(
            enabled=True,
            backend="auto",
            postgres_uri="",
            metrics=metrics,
        )
        assert store is not None
        assert metrics.backend == "memory"
        await store.put("ns/auto", "value")
        rows = await store.recent("ns/auto", limit=1)
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_postgres_failure_falls_back_to_memory(self, monkeypatch):
        async def _boom(self) -> None:
            raise RuntimeError("db down")

        monkeypatch.setattr(PostgresSharedContextStore, "setup", _boom)
        metrics = SharedContextMetrics()
        store = await create_shared_context_store(
            enabled=True,
            backend="postgres",
            postgres_uri="postgresql://user:pass@localhost:5432/db",
            metrics=metrics,
        )
        assert store is not None
        assert metrics.backend == "memory"
        await store.put("ns/fallback", "record")
        rows = await store.search("ns/fallback", "record", limit=1)
        assert len(rows) == 1

