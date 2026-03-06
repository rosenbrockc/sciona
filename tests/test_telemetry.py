"""Tests for the telemetry module (Issue 10)."""

import json
import tempfile
from pathlib import Path

from ageom.telemetry import (
    EventLog,
    PipelineEvent,
    configure_dashboard_output,
    increment_run_metadata_counter,
    finish_prompt_dispatch,
    finish_run,
    get_event_log,
    get_runtime_run,
    load_persisted_runs,
    log_event,
    merge_run_metadata,
    reset_telemetry_runtime,
    start_prompt_dispatch,
    start_run,
    telemetry_scope,
    update_stage,
)


def test_pipeline_event_creation():
    ev = PipelineEvent(
        timestamp=1000.0,
        round="architect",
        phase="decompose",
        event_type="DECOMPOSITION_START",
        node_id="root",
        payload={"goal": "test"},
    )
    assert ev.round == "architect"
    assert ev.event_type == "DECOMPOSITION_START"
    assert ev.payload == {"goal": "test"}


def test_event_log_append_and_len():
    log = EventLog()
    assert len(log) == 0

    ev = PipelineEvent(
        timestamp=1.0,
        round="hunter",
        phase="search",
        event_type="SEARCH_START",
    )
    log.append(ev)
    assert len(log) == 1
    assert log.events[0].event_type == "SEARCH_START"


def test_event_log_to_jsonl():
    log = EventLog()
    log.append(
        PipelineEvent(
            timestamp=1.0,
            round="a",
            phase="b",
            event_type="E1",
        )
    )
    log.append(
        PipelineEvent(
            timestamp=2.0,
            round="c",
            phase="d",
            event_type="E2",
            payload={"key": "value"},
        )
    )
    jsonl = log.to_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event_type"] == "E1"
    second = json.loads(lines[1])
    assert second["payload"] == {"key": "value"}


def test_event_log_save():
    log = EventLog()
    log.append(
        PipelineEvent(
            timestamp=1.0,
            round="synth",
            phase="compile",
            event_type="COMPILE_ATTEMPT",
            duration_ms=42.5,
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace.jsonl"
        log.save(path)
        assert path.exists()
        content = path.read_text()
        data = json.loads(content.strip())
        assert data["duration_ms"] == 42.5


def test_event_log_clear():
    log = EventLog()
    log.append(
        PipelineEvent(
            timestamp=1.0,
            round="a",
            phase="b",
            event_type="X",
        )
    )
    assert len(log) == 1
    log.clear()
    assert len(log) == 0


def test_log_event_convenience():
    # Reset the global log
    global_log = get_event_log()
    global_log.clear()

    ev = log_event(
        "hunter",
        "verify",
        "VERIFICATION_ATTEMPT",
        node_id="n1",
        payload={"candidate": "foo"},
        duration_ms=100.0,
    )

    assert ev.round == "hunter"
    assert ev.phase == "verify"
    assert ev.node_id == "n1"
    assert ev.duration_ms == 100.0
    assert len(global_log) >= 1

    # Clean up
    global_log.clear()


def test_get_event_log_singleton():
    a = get_event_log()
    b = get_event_log()
    assert a is b


class _DummyClient:
    _model = "dummy-model"


def test_runtime_run_stage_and_prompt_tracking():
    reset_telemetry_runtime()
    get_event_log().clear()
    run_id = start_run("algorithm_creation", run_id="run-test")
    assert run_id == "run-test"

    with telemetry_scope(run_id=run_id, stage="architect_decompose"):
        update_stage(stage="architect_decompose", status="running", total=1, completed=0)
        dispatch_id = start_prompt_dispatch("architect_decompose", client=_DummyClient())
        finish_prompt_dispatch(dispatch_id, ok=True)
        update_stage(stage="architect_decompose", status="completed", total=1, completed=1)

    finish_run(run_id, status="completed")
    snapshot = get_runtime_run(run_id)
    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert snapshot["prompt_dispatches"] == 1
    assert snapshot["prompt_successes"] == 1
    assert snapshot["prompt_inflight"] == 0
    stages = snapshot["stages"]
    assert "architect_decompose" in stages
    assert stages["architect_decompose"]["status"] == "completed"

    done_events = [
        ev for ev in get_event_log().events if ev.event_type == "PROMPT_DISPATCH_DONE"
    ]
    assert len(done_events) == 1
    done = done_events[0]
    assert done.prompt_key == "architect_decompose"
    assert done.stage == "architect_decompose"
    assert done.model == "dummy-model"
    assert done.duration_ms is not None
    assert done.duration_ms >= 0.0


def test_persisted_run_snapshot(tmp_path: Path):
    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    run_id = start_run("pipeline_x", run_id="persist-1")
    finish_run(run_id, status="completed")

    rows = load_persisted_runs(tmp_path, limit=10)
    assert rows
    assert rows[0]["run_id"] == "persist-1"


def test_run_metadata_merge_and_increment():
    reset_telemetry_runtime()
    run_id = start_run("algorithm_creation", run_id="meta-run")

    with telemetry_scope(run_id=run_id):
        merge_run_metadata(
            {
                "architect_metrics": {
                    "node_status_counts": {"pending": 2},
                    "unresolved_leaf_count": 2,
                }
            }
        )
        increment_run_metadata_counter(
            "architect_metrics",
            "critique_reject_counts_by_category",
            "type_compatibility",
        )
        increment_run_metadata_counter(
            "architect_metrics",
            "critique_reject_counts_by_category",
            "type_compatibility",
        )

    snapshot = get_runtime_run(run_id)
    assert snapshot is not None
    architect = snapshot["metadata"]["architect_metrics"]
    assert architect["node_status_counts"]["pending"] == 2
    assert architect["unresolved_leaf_count"] == 2
    assert architect["critique_reject_counts_by_category"]["type_compatibility"] == 2
