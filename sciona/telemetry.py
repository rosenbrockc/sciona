"""Pipeline telemetry and dashboard-friendly runtime snapshots.

The module keeps:
- append-only structured events (``EventLog``),
- active/past run snapshots (stage progress + prompt dispatch counters),
- optional persisted run JSON files for cross-process dashboarding.

Existing ``log_event`` APIs remain compatible.
"""

from __future__ import annotations

import contextvars
import json
import queue
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineEvent:
    """A single timestamped event from the pipeline."""

    timestamp: float
    round: str  # "architect", "hunter", "synthesizer"
    phase: str  # e.g. "decompose", "search", "compile"
    event_type: str  # e.g. "DECOMPOSITION_START", "MATCH_FOUND"
    node_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    run_id: str = ""
    stage: str = ""
    prompt_key: str = ""
    provider: str = ""
    model: str = ""
    dispatch_id: str = ""


@dataclass
class _Subscriber:
    """Internal subscriber for live event streaming."""

    sub_queue: queue.SimpleQueue
    run_id: str  # empty string means "all runs"


class EventLog:
    """Append-only event log with JSONL export and live subscribers."""

    _MAX_SUBSCRIBER_BACKLOG = 10_000

    def __init__(self) -> None:
        self._events: list[PipelineEvent] = []
        self._live_path: Path | None = None
        self._lock = threading.Lock()
        self._subscribers: dict[str, _Subscriber] = {}
        self._sub_lock = threading.Lock()

    def append(self, event: PipelineEvent) -> None:
        payload = json.dumps(asdict(event), default=str)
        with self._lock:
            self._events.append(event)
            if self._live_path is not None:
                self._live_path.parent.mkdir(parents=True, exist_ok=True)
                with self._live_path.open("a", encoding="utf-8") as fh:
                    fh.write(payload)
                    fh.write("\n")
        # Enqueue to Postgres drain if configured
        if _drain is not None:
            _drain.enqueue_event(asdict(event))
        # Notify subscribers (outside main lock to avoid contention)
        event_dict = asdict(event)
        with self._sub_lock:
            stale_ids: list[str] = []
            for sub_id, sub in self._subscribers.items():
                if sub.run_id and event.run_id != sub.run_id:
                    continue
                if sub.sub_queue.qsize() >= self._MAX_SUBSCRIBER_BACKLOG:
                    stale_ids.append(sub_id)
                    continue
                sub.sub_queue.put(event_dict)
            for sub_id in stale_ids:
                self._subscribers.pop(sub_id, None)

    def subscribe(self, run_id: str = "") -> tuple[str, queue.SimpleQueue]:
        """Subscribe to live events. Returns (subscriber_id, queue)."""
        sub_id = uuid.uuid4().hex
        q: queue.SimpleQueue = queue.SimpleQueue()
        with self._sub_lock:
            self._subscribers[sub_id] = _Subscriber(sub_queue=q, run_id=run_id)
        return sub_id, q

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber."""
        with self._sub_lock:
            self._subscribers.pop(subscriber_id, None)

    def events_for_run(self, run_id: str) -> list[PipelineEvent]:
        """Return events matching a given run_id."""
        with self._lock:
            return [ev for ev in self._events if ev.run_id == run_id]

    @property
    def events(self) -> list[PipelineEvent]:
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def to_jsonl(self) -> str:
        """Serialize all events to a JSONL string."""
        lines: list[str] = []
        for ev in self._events:
            lines.append(json.dumps(asdict(ev), default=str))
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Write the event log to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_jsonl())

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            if self._live_path is not None:
                self._live_path.parent.mkdir(parents=True, exist_ok=True)
                self._live_path.write_text("")

    def configure_live_output(self, path: str | Path | None) -> None:
        """Enable or disable incremental JSONL writes on append."""
        with self._lock:
            if path is None:
                self._live_path = None
                return
            live_path = Path(path)
            live_path.parent.mkdir(parents=True, exist_ok=True)
            live_path.write_text("")
            self._live_path = live_path


# Module-level singleton
_global_log = EventLog()
_current_run_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sciona_telemetry_run_id", default=""
)
_current_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sciona_telemetry_stage", default=""
)


@dataclass
class PromptDispatchSnapshot:
    """Single in-flight prompt dispatch."""

    dispatch_id: str
    prompt_key: str
    provider: str
    model: str
    stage: str
    started_at: float


@dataclass
class FinishedPromptSnapshot:
    """Completed prompt dispatch metadata for event emission."""

    dispatch_id: str
    run_id: str
    prompt_key: str
    provider: str
    model: str
    stage: str
    latency_ms: float
    ok: bool
    error: str = ""


@dataclass
class StageSnapshot:
    """Per-stage progress and heartbeat."""

    name: str
    status: str = "running"  # running | completed | failed
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    last_heartbeat_at: float = field(default_factory=time.time)
    message: str = ""
    completed: int = 0
    total: int = 0


@dataclass
class PromptAggregate:
    """Aggregated counters for a prompt key."""

    prompt_key: str
    provider: str = ""
    model: str = ""
    dispatched: int = 0
    succeeded: int = 0
    failed: int = 0
    inflight: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    _latency_total_ms: float = 0.0
    _latency_count: int = 0

    def record_latency(self, latency_ms: float) -> None:
        if latency_ms < 0:
            return
        self._latency_total_ms += latency_ms
        self._latency_count += 1
        self.avg_latency_ms = self._latency_total_ms / max(1, self._latency_count)
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms


@dataclass
class RunSnapshot:
    """Runtime view for one pipeline run."""

    run_id: str
    pipeline: str
    label: str = ""
    status: str = "running"  # running | completed | failed
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    last_update_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    events_count: int = 0
    prompt_dispatches: int = 0
    prompt_successes: int = 0
    prompt_failures: int = 0
    prompt_inflight: int = 0
    prompt_by_key: dict[str, PromptAggregate] = field(default_factory=dict)
    stages: dict[str, StageSnapshot] = field(default_factory=dict)
    stage_order: list[str] = field(default_factory=list)
    inflight_prompts: dict[str, PromptDispatchSnapshot] = field(default_factory=dict)


class TelemetryRegistry:
    """Tracks active runs and prompt dispatch state."""

    def __init__(self) -> None:
        self._runs: dict[str, RunSnapshot] = {}
        self._dispatch_to_run: dict[str, str] = {}
        self._lock = threading.Lock()
        self._persist_dir: Path | None = None
        self._max_runs_in_memory = 100

    def configure_persist_dir(self, path: str | Path | None) -> None:
        with self._lock:
            if path is None:
                self._persist_dir = None
                return
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            self._persist_dir = directory

    def start_run(
        self,
        pipeline: str,
        *,
        run_id: str | None = None,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        rid = run_id or uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._runs[rid] = RunSnapshot(
                run_id=rid,
                pipeline=pipeline,
                label=label,
                started_at=now,
                last_update_at=now,
                metadata=metadata or {},
            )
            self._trim_runs_if_needed()
            self._persist_locked(rid)
        return rid

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = status
            run.ended_at = now
            run.last_update_at = now
            if error:
                run.error = error
            self._persist_locked(run_id)

    def update_stage(
        self,
        run_id: str,
        stage: str,
        *,
        status: str | None = None,
        message: str | None = None,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            stage_row = run.stages.get(stage)
            if stage_row is None:
                stage_row = StageSnapshot(name=stage, started_at=now, last_heartbeat_at=now)
                run.stages[stage] = stage_row
                run.stage_order.append(stage)
            if status:
                stage_row.status = status
                if status == "running" and stage_row.started_at <= 0:
                    stage_row.started_at = now
                if status in {"completed", "failed"}:
                    stage_row.ended_at = now
            if message is not None:
                stage_row.message = message
            if completed is not None:
                stage_row.completed = max(0, completed)
            if total is not None:
                stage_row.total = max(0, total)
            stage_row.last_heartbeat_at = now
            run.last_update_at = now
            self._persist_locked(run_id)

    def start_prompt(
        self,
        run_id: str,
        *,
        prompt_key: str,
        provider: str,
        model: str,
        stage: str,
    ) -> str:
        dispatch_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return ""
            row = run.prompt_by_key.get(prompt_key)
            if row is None:
                row = PromptAggregate(prompt_key=prompt_key, provider=provider, model=model)
                run.prompt_by_key[prompt_key] = row
            row.dispatched += 1
            row.inflight += 1
            if provider and not row.provider:
                row.provider = provider
            if model and not row.model:
                row.model = model
            run.prompt_dispatches += 1
            run.prompt_inflight += 1
            run.last_update_at = now
            run.inflight_prompts[dispatch_id] = PromptDispatchSnapshot(
                dispatch_id=dispatch_id,
                prompt_key=prompt_key,
                provider=provider,
                model=model,
                stage=stage,
                started_at=now,
            )
            self._dispatch_to_run[dispatch_id] = run_id
            self._persist_locked(run_id)
        return dispatch_id

    def finish_prompt(
        self,
        dispatch_id: str,
        *,
        ok: bool,
        error: str = "",
    ) -> FinishedPromptSnapshot | None:
        now = time.time()
        with self._lock:
            run_id = self._dispatch_to_run.pop(dispatch_id, "")
            if not run_id:
                return None
            run = self._runs.get(run_id)
            if run is None:
                return None
            inflight = run.inflight_prompts.pop(dispatch_id, None)
            if inflight is None:
                return None
            latency_ms = (now - inflight.started_at) * 1000.0
            row = run.prompt_by_key.get(inflight.prompt_key)
            if row is not None:
                row.inflight = max(0, row.inflight - 1)
                row.record_latency(latency_ms)
                if ok:
                    row.succeeded += 1
                else:
                    row.failed += 1
            run.prompt_inflight = max(0, run.prompt_inflight - 1)
            if ok:
                run.prompt_successes += 1
            else:
                run.prompt_failures += 1
                run.error = error or run.error
            run.last_update_at = now
            self._persist_locked(run_id)
            return FinishedPromptSnapshot(
                dispatch_id=dispatch_id,
                run_id=run_id,
                prompt_key=inflight.prompt_key,
                provider=inflight.provider,
                model=inflight.model,
                stage=inflight.stage,
                latency_ms=latency_ms,
                ok=ok,
                error=error,
            )

    def merge_metadata(self, run_id: str, payload: dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            _deep_merge_dict(run.metadata, payload)
            run.last_update_at = now
            self._persist_locked(run_id)

    def increment_metadata_counter(
        self,
        run_id: str,
        path: list[str],
        amount: int = 1,
    ) -> None:
        now = time.time()
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or not path:
                return
            cursor = run.metadata
            for key in path[:-1]:
                child = cursor.get(key)
                if not isinstance(child, dict):
                    child = {}
                    cursor[key] = child
                cursor = child
            leaf = path[-1]
            cursor[leaf] = int(cursor.get(leaf, 0) or 0) + amount
            run.last_update_at = now
            self._persist_locked(run_id)

    def note_event(self, run_id: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.events_count += 1
            run.last_update_at = time.time()
            self._persist_locked(run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return _run_to_dict(run)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [_run_to_dict(r) for r in self._runs.values()]
        rows.sort(key=lambda r: float(r.get("last_update_at", 0.0)), reverse=True)
        return rows

    def reset(self) -> None:
        with self._lock:
            self._runs.clear()
            self._dispatch_to_run.clear()

    def _trim_runs_if_needed(self) -> None:
        if len(self._runs) <= self._max_runs_in_memory:
            return
        ids = sorted(
            self._runs.keys(),
            key=lambda rid: self._runs[rid].last_update_at,
        )
        for rid in ids[:-self._max_runs_in_memory]:
            self._runs.pop(rid, None)

    def _persist_locked(self, run_id: str) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        run_dict = _run_to_dict(run)
        if _drain is not None:
            _drain.enqueue_run_snapshot(run_dict)
        if self._persist_dir is None:
            return
        target = self._persist_dir / f"run_{run_id}.json"
        tmp = self._persist_dir / f".run_{run_id}.tmp"
        tmp.write_text(json.dumps(run_dict, indent=2))
        tmp.replace(target)


def _run_to_dict(run: RunSnapshot) -> dict[str, Any]:
    payload = asdict(run)
    prompt_by_key = payload.get("prompt_by_key", {})
    for row in prompt_by_key.values():
        row.pop("_latency_total_ms", None)
        row.pop("_latency_count", None)
    return payload


def _deep_merge_dict(target: dict[str, Any], payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_dict(target[key], value)
            continue
        target[key] = value


_registry = TelemetryRegistry()

# Postgres drain (set via configure_postgres_telemetry)
_drain: Any = None  # TelemetryDrain | None
_pg_store: Any = None  # PostgresTelemetryStore | None


def get_event_log() -> EventLog:
    """Return the global event log singleton."""
    return _global_log


def subscribe_events(run_id: str = "") -> tuple[str, queue.SimpleQueue]:
    """Subscribe to live pipeline events. Returns (subscriber_id, queue)."""
    return _global_log.subscribe(run_id)


def unsubscribe_events(subscriber_id: str) -> None:
    """Remove an event subscriber."""
    _global_log.unsubscribe(subscriber_id)


def log_event(
    round_name: str,
    phase: str,
    event_type: str,
    *,
    node_id: str = "",
    payload: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    run_id: str = "",
    stage: str = "",
    prompt_key: str = "",
    provider: str = "",
    model: str = "",
    dispatch_id: str = "",
) -> PipelineEvent:
    """Append an event to the global log.

    Args:
        round_name: Pipeline round ("architect", "hunter", "synthesizer").
        phase: Sub-phase within the round.
        event_type: Event type identifier.
        node_id: Optional node identifier.
        payload: Optional dict of additional data.
        duration_ms: Optional timing measurement.

    Returns:
        The created PipelineEvent.
    """
    rid = run_id or _current_run_id.get()
    stage_name = stage or _current_stage.get()
    event = PipelineEvent(
        timestamp=time.time(),
        round=round_name,
        phase=phase,
        event_type=event_type,
        node_id=node_id,
        payload=payload or {},
        duration_ms=duration_ms,
        run_id=rid,
        stage=stage_name,
        prompt_key=prompt_key,
        provider=provider,
        model=model,
        dispatch_id=dispatch_id,
    )
    _global_log.append(event)
    if rid:
        _registry.note_event(rid)
    return event


def get_current_run_id() -> str:
    """Return current telemetry run context."""
    return _current_run_id.get()


def get_current_stage() -> str:
    """Return current telemetry stage context."""
    return _current_stage.get()


@contextmanager
def telemetry_scope(*, run_id: str | None = None, stage: str | None = None):
    """Bind run/stage context for nested telemetry calls."""
    run_token = None
    stage_token = None
    if run_id is not None:
        run_token = _current_run_id.set(run_id)
    if stage is not None:
        stage_token = _current_stage.set(stage)
    try:
        yield
    finally:
        if stage_token is not None:
            _current_stage.reset(stage_token)
        if run_token is not None:
            _current_run_id.reset(run_token)


@contextmanager
def telemetry_stage(
    stage: str,
    *,
    run_id: str | None = None,
    message: str = "",
    total: int | None = None,
):
    """Context manager that marks stage running/completed/failed."""
    rid = run_id or get_current_run_id()
    if rid:
        update_stage(
            stage=stage,
            run_id=rid,
            status="running",
            message=message,
            total=total,
        )
    with telemetry_scope(stage=stage):
        try:
            yield
        except Exception as exc:
            if rid:
                update_stage(
                    stage=stage,
                    run_id=rid,
                    status="failed",
                    message=str(exc),
                )
            raise
        else:
            if rid:
                update_stage(stage=stage, run_id=rid, status="completed")


def configure_dashboard_output(path: str | Path | None) -> None:
    """Set directory where run snapshots are written for dashboard reads."""
    _registry.configure_persist_dir(path)


def start_run(
    pipeline: str,
    *,
    run_id: str | None = None,
    label: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a telemetry run and return its identifier."""
    return _registry.start_run(pipeline, run_id=run_id, label=label, metadata=metadata)


def finish_run(
    run_id: str,
    *,
    status: str = "completed",
    error: str = "",
) -> None:
    """Mark a telemetry run as completed/failed."""
    _registry.finish_run(run_id, status=status, error=error)


def merge_run_metadata(payload: dict[str, Any], *, run_id: str | None = None) -> None:
    """Merge nested metadata into the active run snapshot."""
    rid = run_id or get_current_run_id()
    if not rid or not payload:
        return
    _registry.merge_metadata(rid, payload)


def increment_run_metadata_counter(
    *path: str,
    amount: int = 1,
    run_id: str | None = None,
) -> None:
    """Increment a nested metadata counter on the active run snapshot."""
    rid = run_id or get_current_run_id()
    if not rid or not path:
        return
    _registry.increment_metadata_counter(rid, list(path), amount=amount)


def update_stage(
    *,
    stage: str,
    run_id: str | None = None,
    status: str | None = None,
    message: str | None = None,
    completed: int | None = None,
    total: int | None = None,
) -> None:
    """Update stage heartbeat/progress for the active run."""
    rid = run_id or get_current_run_id()
    if not rid:
        return
    _registry.update_stage(
        rid,
        stage,
        status=status,
        message=message,
        completed=completed,
        total=total,
    )


def _detect_provider_and_model(client: Any) -> tuple[str, str]:
    provider = str(getattr(client, "_telemetry_provider", "") or "").strip()
    model = str(getattr(client, "_telemetry_model", "") or "").strip()
    if not model:
        model = str(getattr(client, "_model", "") or "").strip()
    name = type(client).__name__.lower()
    if provider:
        return provider, model
    if "claude" in name:
        provider = "claude"
    elif "codex" in name:
        provider = "codex"
    elif "gemini" in name:
        provider = "gemini"
    elif "llama" in name:
        provider = "llama_cpp"
    elif "shimpool" in name:
        delegate = getattr(client, "_delegate", None)
        cli = str(getattr(delegate, "_cli", "")).strip()
        if cli:
            provider = f"{cli}_shim"
        model = model or str(getattr(delegate, "_model", "") or "").strip()
    elif "subprocess" in name:
        cli = str(getattr(client, "_cli", "")).strip()
        if cli:
            provider = f"{cli}_cli"
    return provider, model


def start_prompt_dispatch(
    prompt_key: str,
    *,
    client: Any,
    run_id: str | None = None,
) -> str:
    """Register a prompt dispatch for dashboard counters."""
    rid = run_id or get_current_run_id()
    if not rid:
        return ""
    provider, model = _detect_provider_and_model(client)
    stage = get_current_stage()
    dispatch_id = _registry.start_prompt(
        rid,
        prompt_key=prompt_key,
        provider=provider,
        model=model,
        stage=stage,
    )
    if dispatch_id:
        log_event(
            "llm",
            phase=stage or "prompt_dispatch",
            event_type="PROMPT_DISPATCH_START",
            run_id=rid,
            stage=stage,
            prompt_key=prompt_key,
            provider=provider,
            model=model,
            dispatch_id=dispatch_id,
        )
    return dispatch_id


def finish_prompt_dispatch(
    dispatch_id: str,
    *,
    ok: bool,
    error: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    """Finalize prompt dispatch counters."""
    if not dispatch_id:
        return
    finished = _registry.finish_prompt(dispatch_id, ok=ok, error=error)
    if finished is None:
        return
    log_event(
        "llm",
        phase=finished.stage or get_current_stage() or "prompt_dispatch",
        event_type="PROMPT_DISPATCH_DONE" if ok else "PROMPT_DISPATCH_ERROR",
        run_id=finished.run_id or get_current_run_id(),
        stage=finished.stage or get_current_stage(),
        duration_ms=finished.latency_ms,
        prompt_key=finished.prompt_key,
        provider=finished.provider,
        model=finished.model,
        payload=((payload or {}) | ({"error": error} if error else {})),
        dispatch_id=dispatch_id,
    )


def list_runtime_runs() -> list[dict[str, Any]]:
    """List in-memory telemetry runs (newest first)."""
    return _registry.list_runs()


def get_runtime_run(run_id: str) -> dict[str, Any] | None:
    """Get in-memory telemetry run by id."""
    return _registry.get_run(run_id)


def load_persisted_runs(path: str | Path, *, limit: int = 50) -> list[dict[str, Any]]:
    """Load persisted run snapshots from disk."""
    root = Path(path)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for file in sorted(root.glob("run_*.json")):
        try:
            payload = json.loads(file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    rows.sort(key=lambda r: float(r.get("last_update_at", 0.0)), reverse=True)
    return rows[: max(1, limit)]


def get_persisted_run(path: str | Path, run_id: str) -> dict[str, Any] | None:
    """Load one persisted run snapshot from disk."""
    file = Path(path) / f"run_{run_id}.json"
    if not file.exists():
        return None
    try:
        payload = json.loads(file.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def configure_postgres_telemetry(store: Any, drain: Any) -> None:
    """Wire a PostgresTelemetryStore + TelemetryDrain into the global telemetry."""
    global _drain, _pg_store
    _drain = drain
    _pg_store = store


async def load_runs_from_store(
    *, limit: int = 50, status: str | None = None
) -> list[dict[str, Any]] | None:
    """Try loading runs from the Postgres store. Returns None if unavailable."""
    if _pg_store is None:
        return None
    try:
        return await _pg_store.list_runs(limit=limit, status=status)
    except Exception:
        return None


async def load_run_from_store(run_id: str) -> dict[str, Any] | None:
    """Try loading one run from the Postgres store. Returns None if unavailable."""
    if _pg_store is None:
        return None
    try:
        return await _pg_store.get_run(run_id)
    except Exception:
        return None


async def load_events_from_store(
    run_id: str,
    *,
    offset: int = 0,
    limit: int = 200,
    phase: str | None = None,
    event_type: str | None = None,
    prompt_key: str | None = None,
    round_name: str | None = None,
    has_error: bool | None = None,
) -> tuple[list[dict[str, Any]], int] | None:
    """Try loading events from the Postgres store. Returns None if unavailable."""
    if _pg_store is None:
        return None
    try:
        return await _pg_store.list_events(
            run_id,
            offset=offset,
            limit=limit,
            phase=phase,
            event_type=event_type,
            prompt_key=prompt_key,
            round_name=round_name,
            has_error=has_error,
        )
    except Exception:
        return None


def reset_telemetry_runtime() -> None:
    """Reset run registry and event log (for tests/CLI process boundaries)."""
    global _drain, _pg_store
    _registry.reset()
    _global_log.clear()
    _drain = None
    _pg_store = None
