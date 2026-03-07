"""Tests for shared-context stores, factory, and metrics."""

from __future__ import annotations

import pytest

from ageom.shared_context import (
    ContextRecord,
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
        assert snap["template_searches_total"] == 0
        assert snap["template_puts_total"] == 0

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

    def test_duplicate_suppression_and_provenance_labels(self):
        metrics = SharedContextMetrics()
        records = [
            # duplicate text should be suppressed
            ContextRecord(
                text="Alpha Beta",
                metadata={"source_channel": "success", "confidence": 0.9},
            ),
            ContextRecord(
                text="Alpha   Beta",
                metadata={"source_channel": "success", "confidence": 0.9},
            ),
        ]
        block = format_context_block("Shared Context", records, metrics=metrics)
        assert "ch:success" in block
        snap = metrics.snapshot()
        assert snap["duplicate_candidates"] == 2
        assert snap["duplicates_suppressed"] == 1
        assert float(snap["duplicate_suppression_rate"]) == pytest.approx(0.5)

    def test_template_metrics_are_tracked_separately(self):
        metrics = SharedContextMetrics()

        metrics.record_template_search(hits=2)
        metrics.record_template_search(hits=0)
        metrics.record_template_put()
        metrics.record_template_injection(chars=42, records=2)

        snap = metrics.snapshot()
        assert snap["template_searches_total"] == 2
        assert snap["template_search_hits"] == 1
        assert float(snap["template_hit_rate"]) == pytest.approx(0.5)
        assert snap["template_puts_total"] == 1
        assert snap["template_injected_blocks"] == 1
        assert snap["template_injected_records"] == 2
        assert snap["template_injected_chars"] == 42


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

    @pytest.mark.asyncio
    async def test_promotion_to_repo_namespace(self):
        metrics = SharedContextMetrics()
        store = await create_shared_context_store(
            enabled=True,
            backend="memory",
            promotion_enabled=True,
            promotion_min_confidence=0.9,
            repo_namespace="repo/testrepo",
            metrics=metrics,
        )
        assert store is not None
        # confidence inferred as high from /success namespace
        await store.put("hunter/run1/success", "Matched Nat.add_comm")
        promoted = await store.recent("repo/testrepo/success", limit=5)
        assert promoted
        snap = metrics.snapshot()
        assert snap["promotions_total"] >= 1
