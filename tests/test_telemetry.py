"""Tests for the telemetry module (Issue 10)."""

import json
import tempfile
from pathlib import Path

from ageom.telemetry import EventLog, PipelineEvent, get_event_log, log_event


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
        timestamp=1.0, round="hunter", phase="search",
        event_type="SEARCH_START",
    )
    log.append(ev)
    assert len(log) == 1
    assert log.events[0].event_type == "SEARCH_START"


def test_event_log_to_jsonl():
    log = EventLog()
    log.append(PipelineEvent(
        timestamp=1.0, round="a", phase="b", event_type="E1",
    ))
    log.append(PipelineEvent(
        timestamp=2.0, round="c", phase="d", event_type="E2",
        payload={"key": "value"},
    ))
    jsonl = log.to_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event_type"] == "E1"
    second = json.loads(lines[1])
    assert second["payload"] == {"key": "value"}


def test_event_log_save():
    log = EventLog()
    log.append(PipelineEvent(
        timestamp=1.0, round="synth", phase="compile",
        event_type="COMPILE_ATTEMPT", duration_ms=42.5,
    ))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "trace.jsonl"
        log.save(path)
        assert path.exists()
        content = path.read_text()
        data = json.loads(content.strip())
        assert data["duration_ms"] == 42.5


def test_event_log_clear():
    log = EventLog()
    log.append(PipelineEvent(
        timestamp=1.0, round="a", phase="b", event_type="X",
    ))
    assert len(log) == 1
    log.clear()
    assert len(log) == 0


def test_log_event_convenience():
    # Reset the global log
    global_log = get_event_log()
    global_log.clear()

    ev = log_event("hunter", "verify", "VERIFICATION_ATTEMPT",
                   node_id="n1", payload={"candidate": "foo"}, duration_ms=100.0)

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
