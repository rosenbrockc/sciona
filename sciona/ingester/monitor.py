"""Runtime monitoring and status files for ingestion runs.

This module provides:
- live run status at ``.ingest_status.json``
- completion/failure markers (``COMPLETED.json``, ``FAILED.json``)
- incremental trace event appends to ``trace.jsonl``
- staging helpers for atomic artifact publish
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

STATUS_FILE = ".ingest_status.json"
COMPLETED_FILE = "COMPLETED.json"
FAILED_FILE = "FAILED.json"
PARTIAL_DIR = ".partial"
TRACE_FILE = "trace.jsonl"
STATUS_SCHEMA = "sciona.ingester.monitor.status"
MARKER_SCHEMA = "sciona.ingester.monitor.marker"
SURFACE_SCHEMA = "sciona.ingester.monitor.surface"
MONITOR_SCHEMA_VERSION = 1
OUTPUT_SCOPE_SYMBOL = "symbol"
OUTPUT_SCOPE_FAMILY = "family"
OUTPUT_SCOPE_VALUES = frozenset({OUTPUT_SCOPE_SYMBOL, OUTPUT_SCOPE_FAMILY})
STANDARD_ARTIFACT_SURFACE = (
    "atoms.py",
    "state_models.py",
    "witnesses.py",
    "cdg.json",
    "matches.json",
)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def _now_ts() -> float:
    return time.time()


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _normalize_output_scope(value: Any) -> str:
    scope = str(value or OUTPUT_SCOPE_SYMBOL).strip().lower()
    if scope in OUTPUT_SCOPE_VALUES:
        return scope
    return OUTPUT_SCOPE_SYMBOL


class IngestMonitor:
    """Tracks ingestion progress via durable status and marker files."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        enable_trace: bool = False,
        monitor_stdout: bool = False,
        stale_seconds: int = 120,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_dir / STATUS_FILE
        self.completed_path = self.output_dir / COMPLETED_FILE
        self.failed_path = self.output_dir / FAILED_FILE
        self.partial_dir = self.output_dir / PARTIAL_DIR
        self.trace_path = self.output_dir / TRACE_FILE
        self.enable_trace = enable_trace
        self.monitor_stdout = monitor_stdout
        self.stale_seconds = stale_seconds
        self.run_id = uuid.uuid4().hex
        self._status: dict[str, Any] = {}
        self._last_console_line = ""

    def start(
        self,
        *,
        source_path: str,
        class_name: str,
        procedural: bool,
        llm_provider: str,
        llm_model: str,
        max_depth: int,
        output_scope: str = OUTPUT_SCOPE_SYMBOL,
        output_scope_source: str = "default",
    ) -> None:
        if self.completed_path.exists():
            self.completed_path.unlink()
        if self.failed_path.exists():
            self.failed_path.unlink()
        if self.partial_dir.exists():
            shutil.rmtree(self.partial_dir, ignore_errors=True)
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_trace and self.trace_path.exists():
            self.trace_path.unlink()

        started = _now_ts()
        normalized_scope = _normalize_output_scope(output_scope)
        self._status = {
            "schema": STATUS_SCHEMA,
            "schema_version": MONITOR_SCHEMA_VERSION,
            "run_id": self.run_id,
            "state": "running",
            "phase": "startup",
            "current_step": "init",
            "source_path": source_path,
            "class_name": class_name,
            "procedural": procedural,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "max_depth": max_depth,
            "output_dir": str(self.output_dir),
            "output_scope": normalized_scope,
            "output_scope_source": str(output_scope_source or "default"),
            "publication": {
                "target_dir": str(self.output_dir),
                "target_basename": self.output_dir.name,
                "scope": normalized_scope,
                "expected_artifacts": list(STANDARD_ARTIFACT_SURFACE),
                "published_files": [],
                "missing_artifacts": list(STANDARD_ARTIFACT_SURFACE),
            },
            "started_at": started,
            "last_heartbeat_at": started,
            "llm_call_inflight": None,
            "error": "",
        }
        self._write_status()
        self.trace_event("ingester", "startup", "INGEST_START", payload=self._status)

    def heartbeat(
        self,
        *,
        phase: str | None = None,
        step: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if phase is not None:
            self._status["phase"] = phase
        if step is not None:
            self._status["current_step"] = step
        self._status["last_heartbeat_at"] = _now_ts()
        if payload:
            self._status["payload"] = payload
        self._write_status()

    def phase_start(self, phase: str, *, step: str = "") -> None:
        self.heartbeat(phase=phase, step=step or "running")
        self.trace_event("ingester", phase, "PHASE_START", payload={"step": step})

    def phase_end(self, phase: str, *, step: str = "") -> None:
        self.heartbeat(phase=phase, step=step or "done")
        self.trace_event("ingester", phase, "PHASE_END", payload={"step": step})

    def llm_start(self, prompt_key: str) -> None:
        started = _now_ts()
        self._status["llm_call_inflight"] = {
            "prompt_key": prompt_key,
            "started_at": started,
        }
        self._status["last_heartbeat_at"] = started
        self._write_status()
        self.trace_event(
            "ingester",
            self._status.get("phase", ""),
            "LLM_CALL_START",
            payload={"prompt_key": prompt_key},
        )

    def llm_end(self, prompt_key: str, *, ok: bool, error: str = "") -> None:
        inflight = self._status.get("llm_call_inflight") or {}
        started = float(inflight.get("started_at") or _now_ts())
        dur_ms = max(0.0, (_now_ts() - started) * 1000.0)
        self._status["llm_call_inflight"] = None
        self._status["last_heartbeat_at"] = _now_ts()
        self._write_status()
        self.trace_event(
            "ingester",
            self._status.get("phase", ""),
            "LLM_CALL_END",
            payload={"prompt_key": prompt_key, "ok": ok, "error": error},
            duration_ms=dur_ms,
        )

    def complete(self, *, summary: dict[str, Any]) -> None:
        ended = _now_ts()
        self._status.update(
            {
                "state": "completed",
                "ended_at": ended,
                "last_heartbeat_at": ended,
                "llm_call_inflight": None,
                "summary": summary,
            }
        )
        self._write_status()
        _write_json_atomic(
            self.completed_path,
            {
                "schema": MARKER_SCHEMA,
                "schema_version": MONITOR_SCHEMA_VERSION,
                "state": "completed",
                "run_id": self.run_id,
                "completed_at": ended,
                "phase": str(self._status.get("phase") or ""),
                "output_dir": str(self.output_dir),
                "output_scope": _normalize_output_scope(self._status.get("output_scope")),
                "error": "",
                "summary": summary,
            },
        )
        self.trace_event(
            "ingester",
            self._status.get("phase", ""),
            "INGEST_COMPLETE",
            payload=summary,
        )

    def fail(self, *, error: str, phase: str = "") -> None:
        ended = _now_ts()
        if phase:
            self._status["phase"] = phase
        self._status.update(
            {
                "state": "failed",
                "error": error,
                "ended_at": ended,
                "last_heartbeat_at": ended,
                "llm_call_inflight": None,
            }
        )
        self._write_status()
        failure = {
            "schema": MARKER_SCHEMA,
            "schema_version": MONITOR_SCHEMA_VERSION,
            "state": "failed",
            "run_id": self.run_id,
            "failed_at": ended,
            "phase": self._status.get("phase", ""),
            "output_dir": str(self.output_dir),
            "output_scope": _normalize_output_scope(self._status.get("output_scope")),
            "error": error,
            "summary": {},
        }
        _write_json_atomic(self.failed_path, failure)
        self.trace_event(
            "ingester",
            self._status.get("phase", ""),
            "INGEST_FAILED",
            payload=failure,
        )

    def stage_file(self, name: str, content: str) -> Path:
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        path = self.partial_dir / name
        path.write_text(content)
        return path

    def stage_json(self, name: str, data: Any) -> Path:
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        path = self.partial_dir / name
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def publish_staged(
        self,
        *,
        artifact_surface: tuple[str, ...] | list[str] | None = None,
    ) -> list[str]:
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        published: list[str] = []
        for path in sorted(self.partial_dir.iterdir()):
            if not path.is_file():
                continue
            target = self.output_dir / path.name
            os.replace(path, target)
            published.append(path.name)
        shutil.rmtree(self.partial_dir, ignore_errors=True)
        expected_artifacts = list(artifact_surface or STANDARD_ARTIFACT_SURFACE)
        has_status = bool(self._status)
        publication = {
            "target_dir": str(self.output_dir),
            "target_basename": self.output_dir.name,
            "scope": _normalize_output_scope(self._status.get("output_scope")),
            "expected_artifacts": expected_artifacts,
            "published_files": published,
            "missing_artifacts": [
                name for name in expected_artifacts if name not in published
            ],
        }
        self._status["publication"] = publication
        if has_status:
            self._status["last_heartbeat_at"] = _now_ts()
            self._write_status()
            self.trace_event(
                "ingester",
                self._status.get("phase", ""),
                "PUBLISH_STAGED",
                payload=publication,
            )
        return published

    def publication_summary(self) -> dict[str, Any]:
        publication = self._status.get("publication")
        if isinstance(publication, dict):
            return dict(publication)
        return {}

    def trace_event(
        self,
        round_name: str,
        phase: str,
        event_type: str,
        *,
        node_id: str = "",
        payload: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        if not self.enable_trace:
            return
        event = {
            "timestamp": _now_ts(),
            "round": round_name,
            "phase": phase,
            "event_type": event_type,
            "node_id": node_id,
            "payload": payload or {},
            "duration_ms": duration_ms,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def _write_status(self) -> None:
        _write_json_atomic(self.status_path, self._status)
        if self.monitor_stdout:
            self._emit_console()

    def _emit_console(self) -> None:
        status = self._status
        line = (
            f"[{_fmt_ts(float(status.get('last_heartbeat_at') or _now_ts()))}] "
            f"state={status.get('state')} "
            f"phase={status.get('phase')} "
            f"step={status.get('current_step')}"
        )
        inflight = status.get("llm_call_inflight")
        if isinstance(inflight, dict) and inflight.get("prompt_key"):
            line += f" llm={inflight.get('prompt_key')}"
        if line != self._last_console_line:
            print(line)
            self._last_console_line = line

    @staticmethod
    def read_status(output_dir: str | Path) -> dict[str, Any]:
        path = Path(output_dir) / STATUS_FILE
        return IngestMonitor._read_json_dict(path)

    @staticmethod
    def _read_json_dict(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _as_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_schema_version(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return MONITOR_SCHEMA_VERSION

    @staticmethod
    def _normalize_status(status: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(status, dict) or not status:
            return {}
        summary = status.get("summary")
        return {
            "schema": str(status.get("schema") or STATUS_SCHEMA),
            "schema_version": IngestMonitor._as_schema_version(status.get("schema_version")),
            "run_id": str(status.get("run_id") or ""),
            "state": str(status.get("state") or ""),
            "phase": str(status.get("phase") or ""),
            "current_step": str(status.get("current_step") or ""),
            "source_path": str(status.get("source_path") or ""),
            "class_name": str(status.get("class_name") or ""),
            "procedural": bool(status.get("procedural", False)),
            "llm_provider": str(status.get("llm_provider") or ""),
            "llm_model": str(status.get("llm_model") or ""),
            "max_depth": int(status.get("max_depth") or 0),
            "output_dir": str(status.get("output_dir") or ""),
            "output_scope": _normalize_output_scope(status.get("output_scope")),
            "output_scope_source": str(status.get("output_scope_source") or ""),
            "publication": (
                status.get("publication")
                if isinstance(status.get("publication"), dict)
                else {}
            ),
            "started_at": IngestMonitor._as_float(status.get("started_at")),
            "ended_at": IngestMonitor._as_float(status.get("ended_at")),
            "last_heartbeat_at": IngestMonitor._as_float(status.get("last_heartbeat_at")),
            "llm_call_inflight": status.get("llm_call_inflight"),
            "error": str(status.get("error") or ""),
            "summary": summary if isinstance(summary, dict) else {},
        }

    @staticmethod
    def _normalize_marker(marker: dict[str, Any], *, state: str) -> dict[str, Any]:
        if not isinstance(marker, dict) or not marker:
            return {}
        summary = marker.get("summary")
        return {
            "schema": str(marker.get("schema") or MARKER_SCHEMA),
            "schema_version": IngestMonitor._as_schema_version(marker.get("schema_version")),
            "state": state,
            "run_id": str(marker.get("run_id") or ""),
            "phase": str(marker.get("phase") or ""),
            "output_dir": str(marker.get("output_dir") or ""),
            "output_scope": _normalize_output_scope(marker.get("output_scope")),
            "completed_at": IngestMonitor._as_float(marker.get("completed_at")),
            "failed_at": IngestMonitor._as_float(marker.get("failed_at")),
            "error": str(marker.get("error") or ""),
            "summary": summary if isinstance(summary, dict) else {},
        }

    @staticmethod
    def read_completed_marker(output_dir: str | Path) -> dict[str, Any]:
        path = Path(output_dir) / COMPLETED_FILE
        marker = IngestMonitor._read_json_dict(path)
        return IngestMonitor._normalize_marker(marker, state="completed")

    @staticmethod
    def read_failed_marker(output_dir: str | Path) -> dict[str, Any]:
        path = Path(output_dir) / FAILED_FILE
        marker = IngestMonitor._read_json_dict(path)
        return IngestMonitor._normalize_marker(marker, state="failed")

    @staticmethod
    def read_marker(output_dir: str | Path) -> dict[str, Any]:
        completed = IngestMonitor.read_completed_marker(output_dir)
        if completed:
            return completed
        failed = IngestMonitor.read_failed_marker(output_dir)
        if failed:
            return failed
        return {}

    @staticmethod
    def read_surface(
        output_dir: str | Path, *, stale_seconds: int = 120
    ) -> dict[str, Any]:
        status_raw = IngestMonitor.read_status(output_dir)
        marker = IngestMonitor.read_marker(output_dir)
        derived_state = IngestMonitor.classify_state(status_raw, stale_seconds=stale_seconds)
        marker_state = str(marker.get("state") or "")
        if marker_state in {"completed", "failed"}:
            derived_state = marker_state
        return {
            "schema": SURFACE_SCHEMA,
            "schema_version": MONITOR_SCHEMA_VERSION,
            "derived_state": derived_state,
            "status": IngestMonitor._normalize_status(status_raw),
            "marker": marker,
        }

    @staticmethod
    def classify_state(
        status: dict[str, Any], *, stale_seconds: int = 120
    ) -> str:
        if not status:
            return "missing"
        state = str(status.get("state", "")).strip().lower()
        if state in {"completed", "failed"}:
            return state
        if state == "running":
            last = float(status.get("last_heartbeat_at") or 0.0)
            age = _now_ts() - last if last > 0 else float("inf")
            if age > float(stale_seconds):
                inflight = status.get("llm_call_inflight")
                if isinstance(inflight, dict):
                    started = float(inflight.get("started_at") or 0.0)
                    inflight_age = _now_ts() - started if started > 0 else age
                    # Allow a larger window while an LLM call is in flight.
                    if inflight_age <= float(stale_seconds) * 5.0:
                        return "running"
                return "stalled"
            return "running"
        return state or "unknown"
