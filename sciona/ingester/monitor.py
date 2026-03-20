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


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def _now_ts() -> float:
    return time.time()


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


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
        self._status = {
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
                "run_id": self.run_id,
                "completed_at": ended,
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
            "run_id": self.run_id,
            "failed_at": ended,
            "phase": self._status.get("phase", ""),
            "error": error,
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

    def publish_staged(self) -> list[str]:
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        published: list[str] = []
        for path in sorted(self.partial_dir.iterdir()):
            if not path.is_file():
                continue
            target = self.output_dir / path.name
            os.replace(path, target)
            published.append(path.name)
        shutil.rmtree(self.partial_dir, ignore_errors=True)
        return published

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
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}

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
