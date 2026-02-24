"""CLI tests for ingest status monitoring command."""

from __future__ import annotations

import argparse
import json
import time
from unittest.mock import patch

import pytest

from ageom.cli import _cmd_ingest_status, main
from ageom.ingester.monitor import COMPLETED_FILE, FAILED_FILE, STATUS_FILE


def _args(tmp_path, *, json_mode: bool = False, stale_seconds: int = 120):
    return argparse.Namespace(
        output=str(tmp_path),
        json=json_mode,
        stale_seconds=stale_seconds,
    )


def test_parser_accepts_ingest_status_command():
    with patch("sys.argv", ["ageom", "ingest-status", "outdir"]):
        with patch("ageom.cli._cmd_ingest_status") as mock_cmd:
            main()
            mock_cmd.assert_called_once()
            parsed = mock_cmd.call_args[0][0]
            assert parsed.command == "ingest-status"
            assert parsed.output == "outdir"


def test_ingest_status_completed_marker_overrides_running(tmp_path, capsys):
    (tmp_path / STATUS_FILE).write_text(
        json.dumps(
            {
                "state": "running",
                "phase": "phase2_chunk",
                "current_step": "propose_macro_atoms",
                "last_heartbeat_at": time.time(),
            }
        )
    )
    (tmp_path / COMPLETED_FILE).write_text(json.dumps({"ok": True}))

    _cmd_ingest_status(_args(tmp_path))
    out = capsys.readouterr().out
    assert "state=completed" in out


def test_ingest_status_failed_marker_returns_exit_two(tmp_path):
    (tmp_path / FAILED_FILE).write_text(json.dumps({"error": "bad"}))

    with pytest.raises(SystemExit) as exc_info:
        _cmd_ingest_status(_args(tmp_path))
    assert exc_info.value.code == 2


def test_ingest_status_stalled_returns_exit_two(tmp_path):
    (tmp_path / STATUS_FILE).write_text(
        json.dumps(
            {
                "state": "running",
                "phase": "phase2_chunk",
                "current_step": "critic_validate",
                "last_heartbeat_at": time.time() - 1000.0,
            }
        )
    )

    with pytest.raises(SystemExit) as exc_info:
        _cmd_ingest_status(_args(tmp_path, stale_seconds=30))
    assert exc_info.value.code == 2


def test_ingest_status_missing_returns_exit_one(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        _cmd_ingest_status(_args(tmp_path))
    assert exc_info.value.code == 1


def test_ingest_status_json_output(tmp_path, capsys):
    (tmp_path / STATUS_FILE).write_text(
        json.dumps(
            {
                "state": "running",
                "phase": "phase3_emit",
                "current_step": "emit",
                "last_heartbeat_at": time.time(),
            }
        )
    )
    _cmd_ingest_status(_args(tmp_path, json_mode=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["derived_state"] == "running"
    assert payload["status"]["phase"] == "phase3_emit"
