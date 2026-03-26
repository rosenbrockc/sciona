"""Tests for ingestion monitor lifecycle and stall classification."""

from __future__ import annotations

import json
import time

from sciona.ingester.monitor import (
    COMPLETED_FILE,
    FAILED_FILE,
    MARKER_SCHEMA,
    MONITOR_SCHEMA_VERSION,
    PARTIAL_DIR,
    STATUS_SCHEMA,
    STATUS_FILE,
    SURFACE_SCHEMA,
    TRACE_FILE,
    IngestMonitor,
)


def test_monitor_complete_writes_status_marker_and_trace(tmp_path):
    mon = IngestMonitor(tmp_path, enable_trace=True)
    mon.start(
        source_path="src/mod.rs",
        class_name="HMC",
        procedural=False,
        llm_provider="codex_cli",
        llm_model="gpt-5.3-codex",
        max_depth=12,
    )
    mon.phase_start("phase1_extract", step="extract")
    mon.phase_end("phase1_extract", step="ok")
    mon.complete(summary={"cdg_nodes": 12, "cdg_edges": 21})

    status = json.loads((tmp_path / STATUS_FILE).read_text())
    assert status["schema"] == STATUS_SCHEMA
    assert status["schema_version"] == MONITOR_SCHEMA_VERSION
    assert status["state"] == "completed"
    assert status["summary"]["cdg_nodes"] == 12
    completed = json.loads((tmp_path / COMPLETED_FILE).read_text())
    assert completed["schema"] == MARKER_SCHEMA
    assert completed["schema_version"] == MONITOR_SCHEMA_VERSION
    assert completed["state"] == "completed"
    assert (tmp_path / COMPLETED_FILE).exists()
    assert not (tmp_path / FAILED_FILE).exists()

    trace_lines = (tmp_path / TRACE_FILE).read_text().strip().splitlines()
    assert len(trace_lines) >= 3


def test_monitor_fail_writes_failed_marker(tmp_path):
    mon = IngestMonitor(tmp_path, enable_trace=False)
    mon.start(
        source_path="src/nuts.rs",
        class_name="NUTS",
        procedural=False,
        llm_provider="codex_cli",
        llm_model="gpt-5.3-codex",
        max_depth=12,
    )
    mon.fail(error="boom", phase="phase2_chunk")

    status = json.loads((tmp_path / STATUS_FILE).read_text())
    assert status["schema"] == STATUS_SCHEMA
    assert status["schema_version"] == MONITOR_SCHEMA_VERSION
    assert status["state"] == "failed"
    assert status["phase"] == "phase2_chunk"

    failed = json.loads((tmp_path / FAILED_FILE).read_text())
    assert failed["schema"] == MARKER_SCHEMA
    assert failed["schema_version"] == MONITOR_SCHEMA_VERSION
    assert failed["state"] == "failed"
    assert failed["error"] == "boom"
    assert not (tmp_path / COMPLETED_FILE).exists()


def test_monitor_stage_and_publish(tmp_path):
    mon = IngestMonitor(tmp_path)
    mon.start(
        source_path="src/lib.rs",
        class_name="Kernel",
        procedural=True,
        llm_provider="codex_cli",
        llm_model="gpt-5.3-codex",
        max_depth=1,
    )

    mon.stage_file("atoms.py", "print('ok')\n")
    mon.stage_json("cdg.json", {"nodes": [], "edges": []})
    published = mon.publish_staged()

    assert published == ["atoms.py", "cdg.json"]
    assert (tmp_path / "atoms.py").exists()
    assert (tmp_path / "cdg.json").exists()
    assert not (tmp_path / PARTIAL_DIR).exists()


def test_classify_state_stalled_without_inflight():
    status = {
        "state": "running",
        "last_heartbeat_at": time.time() - 200.0,
        "llm_call_inflight": None,
    }
    assert IngestMonitor.classify_state(status, stale_seconds=30) == "stalled"


def test_classify_state_running_with_recent_inflight():
    now = time.time()
    status = {
        "state": "running",
        "last_heartbeat_at": now - 40.0,
        "llm_call_inflight": {
            "prompt_key": "ingester_chunk",
            "started_at": now - 50.0,
        },
    }
    assert IngestMonitor.classify_state(status, stale_seconds=15) == "running"


def test_classify_state_stalled_with_very_old_inflight():
    now = time.time()
    status = {
        "state": "running",
        "last_heartbeat_at": now - 200.0,
        "llm_call_inflight": {
            "prompt_key": "ingester_chunk",
            "started_at": now - 200.0,
        },
    }
    assert IngestMonitor.classify_state(status, stale_seconds=20) == "stalled"


def test_read_marker_and_surface_for_completed_run(tmp_path):
    mon = IngestMonitor(tmp_path, enable_trace=False)
    mon.start(
        source_path="src/mod.rs",
        class_name="HMC",
        procedural=False,
        llm_provider="tests",
        llm_model="fixture",
        max_depth=2,
    )
    mon.complete(summary={"cache_state": "hit"})

    marker = IngestMonitor.read_marker(tmp_path)
    surface = IngestMonitor.read_surface(tmp_path, stale_seconds=30)

    assert marker["state"] == "completed"
    assert marker["summary"]["cache_state"] == "hit"
    assert surface["schema"] == SURFACE_SCHEMA
    assert surface["derived_state"] == "completed"
    assert surface["marker"]["state"] == "completed"
    assert surface["status"]["state"] == "completed"


def test_read_marker_returns_failed_when_only_failed_exists(tmp_path):
    mon = IngestMonitor(tmp_path, enable_trace=False)
    mon.start(
        source_path="src/mod.rs",
        class_name="HMC",
        procedural=False,
        llm_provider="tests",
        llm_model="fixture",
        max_depth=2,
    )
    mon.fail(error="type failure", phase="verify_types")

    marker = IngestMonitor.read_marker(tmp_path)
    surface = IngestMonitor.read_surface(tmp_path, stale_seconds=30)

    assert marker["state"] == "failed"
    assert marker["phase"] == "verify_types"
    assert marker["error"] == "type failure"
    assert surface["derived_state"] == "failed"
