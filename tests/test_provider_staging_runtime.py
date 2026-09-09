"""Staging subprocess orchestration checks using synthetic runtime reports."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_provider_staging_e2e.py"
SPEC = importlib.util.spec_from_file_location("provider_staging_runtime", SCRIPT)
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


@pytest.mark.parametrize("tabular_output", [
    '{"status": "passed", "refinement_improved": false}',
    '{"status": "failed"}',
    '',
])
def test_staging_runs_and_gates_tabular_evaluation(tmp_path, monkeypatch, tabular_output):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output = '{}' if command[1] == '-c' else tabular_output
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(RUNTIME, "_run", fake_run)
    kwargs = dict(
        python=tmp_path / "python", api_url="http://localhost:1234",
        cert_path=tmp_path / "cert.pem", work_dir=tmp_path,
        matcher_root=SCRIPT.parents[1], tabular_cache_dir=tmp_path / "cache",
    )
    if tabular_output and json.loads(tabular_output)["status"] == "passed":
        result = RUNTIME._exercise_cold_runtime(**kwargs)
        assert result["tabular"]["refinement_improved"] is False
    else:
        with pytest.raises(RuntimeError, match="Tabular runtime"):
            RUNTIME._exercise_cold_runtime(**kwargs)
    assert len(calls) == 2
    command, options = calls[-1]
    assert Path(command[1]).name == "tabular_ml_e2e_runtime.py"
    assert command[2:] == ["--api-url", "http://localhost:1234", "--cache-dir", str(tmp_path / "cache")]
    assert options["env"]["SSL_CERT_FILE"] == str(tmp_path / "cert.pem")


def test_staging_propagates_tabular_process_failure(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        if command[1] == "-c":
            return subprocess.CompletedProcess(command, 0, stdout="{}")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(RUNTIME, "_run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        RUNTIME._exercise_cold_runtime(
            python=tmp_path / "python", api_url="http://localhost:1234",
            cert_path=tmp_path / "cert.pem", work_dir=tmp_path,
            matcher_root=SCRIPT.parents[1],
        )
