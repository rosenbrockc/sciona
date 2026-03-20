"""Postgres-backed telemetry persistence with async connection pooling.

Provides ``PostgresTelemetryStore`` for durable event/run storage and
``TelemetryDrain`` for bridging sync producers to async Postgres writes.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_DDL_PIPELINE_RUNS = """\
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           TEXT PRIMARY KEY,
    pipeline         TEXT NOT NULL,
    label            TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'running',
    started_at       DOUBLE PRECISION NOT NULL,
    ended_at         DOUBLE PRECISION,
    last_update_at   DOUBLE PRECISION NOT NULL,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    error            TEXT NOT NULL DEFAULT '',
    events_count     INTEGER NOT NULL DEFAULT 0,
    prompt_dispatches INTEGER NOT NULL DEFAULT 0,
    prompt_successes  INTEGER NOT NULL DEFAULT 0,
    prompt_failures   INTEGER NOT NULL DEFAULT 0,
    prompt_inflight   INTEGER NOT NULL DEFAULT 0,
    prompt_by_key    JSONB NOT NULL DEFAULT '{}'::jsonb,
    stages           JSONB NOT NULL DEFAULT '{}'::jsonb,
    stage_order      JSONB NOT NULL DEFAULT '[]'::jsonb,
    inflight_prompts JSONB NOT NULL DEFAULT '{}'::jsonb
)"""

_DDL_PIPELINE_RUNS_IDX = """\
CREATE INDEX IF NOT EXISTS pipeline_runs_status_started_idx
    ON pipeline_runs (status, started_at DESC)"""

_DDL_PIPELINE_EVENTS = """\
CREATE TABLE IF NOT EXISTS pipeline_events (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL,
    timestamp    DOUBLE PRECISION NOT NULL,
    round        TEXT NOT NULL,
    phase        TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    node_id      TEXT NOT NULL DEFAULT '',
    payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
    duration_ms  DOUBLE PRECISION,
    stage        TEXT NOT NULL DEFAULT '',
    prompt_key   TEXT NOT NULL DEFAULT '',
    provider     TEXT NOT NULL DEFAULT '',
    model        TEXT NOT NULL DEFAULT '',
    dispatch_id  TEXT NOT NULL DEFAULT ''
)"""

_DDL_PIPELINE_EVENTS_IDX1 = """\
CREATE INDEX IF NOT EXISTS pipeline_events_run_type_idx
    ON pipeline_events (run_id, event_type)"""

_DDL_PIPELINE_EVENTS_IDX2 = """\
CREATE INDEX IF NOT EXISTS pipeline_events_type_ts_idx
    ON pipeline_events (event_type, timestamp)"""


class PostgresTelemetryStore:
    """Async Postgres persistence with connection pooling."""

    def __init__(self, postgres_uri: str, *, min_size: int = 1, max_size: int = 4) -> None:
        self._postgres_uri = postgres_uri
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None  # AsyncConnectionPool

    async def setup(self) -> None:
        """Create tables/indexes and open the connection pool."""
        from psycopg_pool import AsyncConnectionPool

        self._pool = AsyncConnectionPool(
            self._postgres_uri,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
        )
        await self._pool.open()

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_DDL_PIPELINE_RUNS)
                await cur.execute(_DDL_PIPELINE_RUNS_IDX)
                await cur.execute(_DDL_PIPELINE_EVENTS)
                await cur.execute(_DDL_PIPELINE_EVENTS_IDX1)
                await cur.execute(_DDL_PIPELINE_EVENTS_IDX2)
            await conn.commit()

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def upsert_run(self, run_dict: dict[str, Any]) -> None:
        """INSERT ... ON CONFLICT (run_id) DO UPDATE for a run snapshot."""
        if self._pool is None:
            return
        rd = run_dict
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """\
                    INSERT INTO pipeline_runs (
                        run_id, pipeline, label, status, started_at, ended_at,
                        last_update_at, metadata, error, events_count,
                        prompt_dispatches, prompt_successes, prompt_failures,
                        prompt_inflight, prompt_by_key, stages, stage_order,
                        inflight_prompts
                    ) VALUES (
                        %(run_id)s, %(pipeline)s, %(label)s, %(status)s,
                        %(started_at)s, %(ended_at)s, %(last_update_at)s,
                        %(metadata)s, %(error)s, %(events_count)s,
                        %(prompt_dispatches)s, %(prompt_successes)s,
                        %(prompt_failures)s, %(prompt_inflight)s,
                        %(prompt_by_key)s, %(stages)s, %(stage_order)s,
                        %(inflight_prompts)s
                    ) ON CONFLICT (run_id) DO UPDATE SET
                        pipeline = EXCLUDED.pipeline,
                        label = EXCLUDED.label,
                        status = EXCLUDED.status,
                        started_at = EXCLUDED.started_at,
                        ended_at = EXCLUDED.ended_at,
                        last_update_at = EXCLUDED.last_update_at,
                        metadata = EXCLUDED.metadata,
                        error = EXCLUDED.error,
                        events_count = EXCLUDED.events_count,
                        prompt_dispatches = EXCLUDED.prompt_dispatches,
                        prompt_successes = EXCLUDED.prompt_successes,
                        prompt_failures = EXCLUDED.prompt_failures,
                        prompt_inflight = EXCLUDED.prompt_inflight,
                        prompt_by_key = EXCLUDED.prompt_by_key,
                        stages = EXCLUDED.stages,
                        stage_order = EXCLUDED.stage_order,
                        inflight_prompts = EXCLUDED.inflight_prompts
                    """,
                    {
                        "run_id": rd.get("run_id", ""),
                        "pipeline": rd.get("pipeline", ""),
                        "label": rd.get("label", ""),
                        "status": rd.get("status", "running"),
                        "started_at": rd.get("started_at", 0.0),
                        "ended_at": rd.get("ended_at"),
                        "last_update_at": rd.get("last_update_at", 0.0),
                        "metadata": json.dumps(rd.get("metadata", {}), default=str),
                        "error": rd.get("error", ""),
                        "events_count": rd.get("events_count", 0),
                        "prompt_dispatches": rd.get("prompt_dispatches", 0),
                        "prompt_successes": rd.get("prompt_successes", 0),
                        "prompt_failures": rd.get("prompt_failures", 0),
                        "prompt_inflight": rd.get("prompt_inflight", 0),
                        "prompt_by_key": json.dumps(rd.get("prompt_by_key", {}), default=str),
                        "stages": json.dumps(rd.get("stages", {}), default=str),
                        "stage_order": json.dumps(rd.get("stage_order", []), default=str),
                        "inflight_prompts": json.dumps(
                            rd.get("inflight_prompts", {}), default=str
                        ),
                    },
                )
            await conn.commit()

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """SELECT one run by primary key."""
        if self._pool is None:
            return None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM pipeline_runs WHERE run_id = %s", (run_id,)
                )
                row = await cur.fetchone()
                if row is None:
                    return None
                cols = [desc[0] for desc in cur.description]
                return _row_to_run_dict(cols, row)

    async def list_runs(
        self, *, limit: int = 50, status: str | None = None
    ) -> list[dict[str, Any]]:
        """SELECT runs with optional status filter, newest first."""
        if self._pool is None:
            return []
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                if status:
                    await cur.execute(
                        "SELECT * FROM pipeline_runs WHERE status = %s "
                        "ORDER BY last_update_at DESC LIMIT %s",
                        (status, limit),
                    )
                else:
                    await cur.execute(
                        "SELECT * FROM pipeline_runs "
                        "ORDER BY last_update_at DESC LIMIT %s",
                        (limit,),
                    )
                rows = await cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                return [_row_to_run_dict(cols, r) for r in rows]

    async def insert_events(self, events: list[dict[str, Any]]) -> None:
        """Batch INSERT events via executemany."""
        if self._pool is None or not events:
            return
        params = [
            (
                ev.get("run_id", ""),
                ev.get("timestamp", 0.0),
                ev.get("round", ""),
                ev.get("phase", ""),
                ev.get("event_type", ""),
                ev.get("node_id", ""),
                json.dumps(ev.get("payload", {}), default=str),
                ev.get("duration_ms"),
                ev.get("stage", ""),
                ev.get("prompt_key", ""),
                ev.get("provider", ""),
                ev.get("model", ""),
                ev.get("dispatch_id", ""),
            )
            for ev in events
        ]
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """\
                    INSERT INTO pipeline_events (
                        run_id, timestamp, round, phase, event_type, node_id,
                        payload, duration_ms, stage, prompt_key, provider,
                        model, dispatch_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    params,
                )
            await conn.commit()

    async def list_events(
        self,
        run_id: str,
        *,
        offset: int = 0,
        limit: int = 200,
        phase: str | None = None,
        event_type: str | None = None,
        prompt_key: str | None = None,
        round_name: str | None = None,
        has_error: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """SELECT events with server-side filtering + pagination.

        Returns (events, total_count).
        """
        if self._pool is None:
            return [], 0

        where = ["run_id = %s"]
        params: list[Any] = [run_id]

        if phase:
            where.append("phase = %s")
            params.append(phase)
        if event_type:
            where.append("event_type = %s")
            params.append(event_type)
        if prompt_key:
            where.append("prompt_key = %s")
            params.append(prompt_key)
        if round_name:
            where.append("round = %s")
            params.append(round_name)
        if has_error is True:
            where.append(
                "(event_type LIKE '%%ERROR%%' OR event_type LIKE '%%FAIL%%' "
                "OR payload::text LIKE '%%\"error\"%%')"
            )
        elif has_error is False:
            where.append(
                "event_type NOT LIKE '%%ERROR%%' AND event_type NOT LIKE '%%FAIL%%' "
                "AND payload::text NOT LIKE '%%\"error\"%%'"
            )

        where_sql = " AND ".join(where)

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT count(*) FROM pipeline_events WHERE {where_sql}",
                    params,
                )
                total = (await cur.fetchone())[0]

                await cur.execute(
                    f"SELECT * FROM pipeline_events WHERE {where_sql} "
                    f"ORDER BY timestamp, id LIMIT %s OFFSET %s",
                    params + [limit, offset],
                )
                rows = await cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                events = [_row_to_event_dict(cols, r) for r in rows]
        return events, total

    async def list_events_for_coverage(
        self, run_id: str
    ) -> list[dict[str, Any]]:
        """SELECT events relevant to prompt coverage analysis."""
        if self._pool is None:
            return []
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM pipeline_events "
                    "WHERE run_id = %s AND event_type IN ('PROMPT_DISPATCH_DONE', 'PROMPT_DISPATCH_ERROR') "
                    "ORDER BY timestamp, id",
                    (run_id,),
                )
                rows = await cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                return [_row_to_event_dict(cols, r) for r in rows]

    async def list_events_for_errors(
        self, run_id: str
    ) -> list[dict[str, Any]]:
        """SELECT events with error indicators."""
        if self._pool is None:
            return []
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM pipeline_events "
                    "WHERE run_id = %s AND ("
                    "  event_type LIKE '%%ERROR%%' OR event_type LIKE '%%FAIL%%' "
                    "  OR payload::text LIKE '%%\"error\"%%'"
                    ") ORDER BY timestamp, id",
                    (run_id,),
                )
                rows = await cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                return [_row_to_event_dict(cols, r) for r in rows]


def _row_to_run_dict(cols: list[str], row: tuple) -> dict[str, Any]:
    """Convert a DB row to a run dict, deserializing JSONB columns."""
    d = dict(zip(cols, row))
    for key in ("metadata", "prompt_by_key", "stages", "stage_order", "inflight_prompts"):
        val = d.get(key)
        if isinstance(val, str):
            try:
                d[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _row_to_event_dict(cols: list[str], row: tuple) -> dict[str, Any]:
    """Convert a DB row to an event dict."""
    d = dict(zip(cols, row))
    val = d.get("payload")
    if isinstance(val, str):
        try:
            d["payload"] = json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return d


# ---------------------------------------------------------------------------
# TelemetryDrain — bridges sync producers → async Postgres writes
# ---------------------------------------------------------------------------


class TelemetryDrain:
    """Buffers events and run snapshots, flushing to Postgres periodically."""

    def __init__(
        self,
        store: PostgresTelemetryStore,
        *,
        flush_interval: float = 0.5,
        max_events: int = 10_000,
    ) -> None:
        self._store = store
        self._flush_interval = flush_interval
        self._event_buf: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=max_events
        )
        self._run_buf: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._task: asyncio.Task | None = None
        self._stop = False

    def enqueue_event(self, event_dict: dict[str, Any]) -> None:
        """Thread-safe append to the event buffer."""
        with self._lock:
            self._event_buf.append(event_dict)

    def enqueue_run_snapshot(self, run_dict: dict[str, Any]) -> None:
        """Thread-safe last-writer-wins per run_id."""
        run_id = run_dict.get("run_id", "")
        if not run_id:
            return
        with self._lock:
            self._run_buf[run_id] = run_dict

    async def start(self) -> None:
        """Launch the background flush loop."""
        self._stop = False
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Signal stop, await final flush."""
        self._stop = True
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Final flush
        await self._flush()

    async def _loop(self) -> None:
        while not self._stop:
            await asyncio.sleep(self._flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Batch insert buffered events, upsert buffered runs."""
        with self._lock:
            events = list(self._event_buf)
            self._event_buf.clear()
            runs = dict(self._run_buf)
            self._run_buf.clear()

        try:
            if events:
                await self._store.insert_events(events)
        except Exception:
            logger.debug("telemetry drain: failed to insert events", exc_info=True)

        for run_dict in runs.values():
            try:
                await self._store.upsert_run(run_dict)
            except Exception:
                logger.debug("telemetry drain: failed to upsert run", exc_info=True)
