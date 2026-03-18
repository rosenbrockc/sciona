"""Tests for PostgresTelemetryStore and TelemetryDrain with mocked Postgres."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ageom.telemetry_store import (
    PostgresTelemetryStore,
    TelemetryDrain,
    _DDL_PIPELINE_EVENTS,
    _DDL_PIPELINE_EVENTS_IDX1,
    _DDL_PIPELINE_EVENTS_IDX2,
    _DDL_PIPELINE_RUNS,
    _DDL_PIPELINE_RUNS_IDX,
)


class _AsyncCtx:
    """Simple async context manager wrapper for mocks."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *args):
        return False


def _make_mock_pool():
    """Create a mock AsyncConnectionPool with cursor support."""
    cur = AsyncMock()
    cur.description = [
        ("run_id",), ("pipeline",), ("label",), ("status",),
        ("started_at",), ("ended_at",), ("last_update_at",),
        ("metadata",), ("error",), ("events_count",),
        ("prompt_dispatches",), ("prompt_successes",),
        ("prompt_failures",), ("prompt_inflight",),
        ("prompt_by_key",), ("stages",), ("stage_order",),
        ("inflight_prompts",),
    ]

    conn = MagicMock()
    conn.cursor.return_value = _AsyncCtx(cur)
    conn.commit = AsyncMock()

    pool = MagicMock()
    pool.connection.return_value = _AsyncCtx(conn)
    pool.open = AsyncMock()
    pool.close = AsyncMock()

    return pool, conn, cur


def _sample_run(run_id: str = "r1") -> dict:
    return {
        "run_id": run_id,
        "pipeline": "test",
        "label": "test-label",
        "status": "running",
        "started_at": 1000.0,
        "ended_at": None,
        "last_update_at": 1000.0,
        "metadata": {"key": "val"},
        "error": "",
        "events_count": 0,
        "prompt_dispatches": 0,
        "prompt_successes": 0,
        "prompt_failures": 0,
        "prompt_inflight": 0,
        "prompt_by_key": {},
        "stages": {},
        "stage_order": [],
        "inflight_prompts": {},
    }


def _sample_event(run_id: str = "r1", event_type: str = "TEST_EVENT") -> dict:
    return {
        "run_id": run_id,
        "timestamp": 1000.0,
        "round": "test",
        "phase": "test",
        "event_type": event_type,
        "node_id": "",
        "payload": {},
        "duration_ms": None,
        "stage": "",
        "prompt_key": "",
        "provider": "",
        "model": "",
        "dispatch_id": "",
    }


@pytest.mark.asyncio
async def test_setup_creates_tables():
    pool, conn, cur = _make_mock_pool()

    with patch("psycopg_pool.AsyncConnectionPool", return_value=pool):
        store = PostgresTelemetryStore("postgresql://test")
        await store.setup()

    executed = [call.args[0] for call in cur.execute.call_args_list]
    assert _DDL_PIPELINE_RUNS in executed
    assert _DDL_PIPELINE_RUNS_IDX in executed
    assert _DDL_PIPELINE_EVENTS in executed
    assert _DDL_PIPELINE_EVENTS_IDX1 in executed
    assert _DDL_PIPELINE_EVENTS_IDX2 in executed
    conn.commit.assert_awaited()


@pytest.mark.asyncio
async def test_upsert_run_insert_and_update():
    pool, conn, cur = _make_mock_pool()

    store = PostgresTelemetryStore("postgresql://test")
    store._pool = pool

    run = _sample_run()
    await store.upsert_run(run)
    assert cur.execute.await_count == 1
    sql_text = cur.execute.call_args_list[0].args[0]
    assert "ON CONFLICT (run_id) DO UPDATE" in sql_text

    # Update
    run["status"] = "completed"
    await store.upsert_run(run)
    assert cur.execute.await_count == 2


@pytest.mark.asyncio
async def test_insert_events_batch():
    pool, conn, cur = _make_mock_pool()

    store = PostgresTelemetryStore("postgresql://test")
    store._pool = pool

    events = [_sample_event(event_type=f"EV_{i}") for i in range(5)]
    await store.insert_events(events)

    cur.executemany.assert_awaited_once()
    params = cur.executemany.call_args.args[1]
    assert len(params) == 5


@pytest.mark.asyncio
async def test_list_runs_with_status_filter():
    pool, conn, cur = _make_mock_pool()
    cur.fetchall = AsyncMock(return_value=[])

    store = PostgresTelemetryStore("postgresql://test")
    store._pool = pool

    await store.list_runs(limit=10, status="completed")
    sql_text = cur.execute.call_args_list[0].args[0]
    assert "WHERE status = %s" in sql_text

    cur.execute.reset_mock()
    await store.list_runs(limit=10)
    sql_text = cur.execute.call_args_list[0].args[0]
    assert "WHERE status" not in sql_text


@pytest.mark.asyncio
async def test_list_events_with_filters():
    pool, conn, cur = _make_mock_pool()
    cur.fetchone = AsyncMock(return_value=(0,))
    cur.fetchall = AsyncMock(return_value=[])
    # Override description for event columns
    cur.description = [
        ("id",), ("run_id",), ("timestamp",), ("round",), ("phase",),
        ("event_type",), ("node_id",), ("payload",), ("duration_ms",),
        ("stage",), ("prompt_key",), ("provider",), ("model",), ("dispatch_id",),
    ]

    store = PostgresTelemetryStore("postgresql://test")
    store._pool = pool

    await store.list_events("r1", phase="test", event_type="X", has_error=True)
    # Should have count query + data query
    assert cur.execute.await_count == 2
    count_sql = cur.execute.call_args_list[0].args[0]
    assert "phase = %s" in count_sql
    assert "event_type = %s" in count_sql
    assert "ERROR" in count_sql  # has_error filter


@pytest.mark.asyncio
async def test_drain_batches_events():
    store = AsyncMock(spec=PostgresTelemetryStore)
    store.insert_events = AsyncMock()
    store.upsert_run = AsyncMock()

    drain = TelemetryDrain(store, flush_interval=0.05)
    await drain.start()

    for i in range(20):
        drain.enqueue_event(_sample_event(event_type=f"EV_{i}"))

    # Wait for at least one flush
    await asyncio.sleep(0.2)
    await drain.stop()

    # All 20 events should have been flushed in one or more batches
    total_events = sum(
        len(call.args[0]) for call in store.insert_events.call_args_list
    )
    assert total_events == 20


@pytest.mark.asyncio
async def test_drain_last_writer_wins_for_runs():
    store = AsyncMock(spec=PostgresTelemetryStore)
    store.insert_events = AsyncMock()
    store.upsert_run = AsyncMock()

    drain = TelemetryDrain(store, flush_interval=0.05)

    # Enqueue two snapshots for the same run_id before any flush
    run1 = _sample_run("r1")
    run1["events_count"] = 1
    drain.enqueue_run_snapshot(run1)

    run2 = _sample_run("r1")
    run2["events_count"] = 5
    drain.enqueue_run_snapshot(run2)

    await drain.start()
    await asyncio.sleep(0.2)
    await drain.stop()

    # Should have upserted only the last snapshot
    upserted = [call.args[0] for call in store.upsert_run.call_args_list]
    assert len(upserted) == 1
    assert upserted[0]["events_count"] == 5


@pytest.mark.asyncio
async def test_drain_silent_failure():
    store = AsyncMock(spec=PostgresTelemetryStore)
    store.insert_events = AsyncMock(side_effect=RuntimeError("db down"))
    store.upsert_run = AsyncMock()

    drain = TelemetryDrain(store, flush_interval=0.05)
    drain.enqueue_event(_sample_event())
    await drain.start()
    await asyncio.sleep(0.2)
    # Should not raise
    await drain.stop()


@pytest.mark.asyncio
async def test_configure_postgres_telemetry_wires_drain():
    from unittest.mock import patch as _patch

    from ageom.telemetry import configure_postgres_telemetry, log_event, reset_telemetry_runtime

    mock_drain = MagicMock()
    mock_drain.enqueue_event = MagicMock()
    mock_store = MagicMock()

    try:
        configure_postgres_telemetry(mock_store, mock_drain)
        log_event("test", "test", "TEST_EVENT")
        mock_drain.enqueue_event.assert_called_once()
        event_dict = mock_drain.enqueue_event.call_args.args[0]
        assert event_dict["event_type"] == "TEST_EVENT"
    finally:
        reset_telemetry_runtime()
