from __future__ import annotations

import argparse
import json

import pytest


@pytest.mark.asyncio
async def test_benchmark_validate_writes_telemetry_metadata(monkeypatch, tmp_path):
    from ageom.cli import _cmd_benchmark_validate
    from ageom.telemetry import configure_dashboard_output, get_persisted_run, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

    async def _fake_run_benchmark_validation(output_dir):
        return {
            "summary_report": "build/benchmark_validation/summary.json",
            "prompt_report": "build/benchmark_validation/prompt_benchmark.json",
            "flow_report": "build/benchmark_validation/flow_benchmark.json",
            "prompt_cases": 12,
            "prompt_results": 24,
            "prompt_summary": "prompt summary",
            "prompt_stability_summary": "fixture_good/tuned 12/12",
            "flow_cases": 4,
            "flow_results": 16,
            "flow_summary": "flow summary",
            "flow_stability_summary": "rapid 4/4, verified 4/4",
            "flow_avg_prompt_calls": {"rapid": 6.0, "verified": 7.0},
            "prompt_tuned_failures": 0,
            "prompt_tuned_unstable_groups": 0,
            "flow_mode_failures": 0,
            "flow_mode_unstable_groups": 0,
        }

    monkeypatch.setattr(
        "ageom.benchmark_validation.run_benchmark_validation",
        _fake_run_benchmark_validation,
    )

    await _cmd_benchmark_validate(argparse.Namespace(output="build/benchmark_validation"))

    persisted = sorted(tmp_path.glob("run_*.json"))
    assert persisted
    payload = json.loads(persisted[-1].read_text(encoding="utf-8"))
    assert payload["pipeline"] == "benchmark_validation"
    assert payload["status"] == "completed"
    bench = payload["metadata"]["benchmark_validation"]
    assert bench["prompt_results"] == 24
    assert bench["flow_results"] == 16
    assert bench["flow_avg_prompt_calls"]["rapid"] == 6.0
    assert bench["flow_mode_failures"] == 0


@pytest.mark.asyncio
async def test_release_validate_writes_telemetry_metadata(monkeypatch, tmp_path):
    from ageom.cli import _cmd_release_validate
    from ageom.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

    async def _fake_run_release_validation(output_dir):
        return {
            "manifest": "build/release_validation/release_validation.json",
            "benchmarks_dir": "build/release_validation/benchmarks",
            "benchmark_summary": {
                "summary_report": "build/release_validation/benchmarks/summary.json",
                "prompt_report": "build/release_validation/benchmarks/prompt_benchmark.json",
                "flow_report": "build/release_validation/benchmarks/flow_benchmark.json",
                "prompt_cases": 12,
                "prompt_results": 24,
                "prompt_summary": "prompt summary",
                "prompt_stability_summary": "fixture_good/tuned 12/12",
                "flow_cases": 4,
                "flow_results": 16,
                "flow_summary": "flow summary",
                "flow_stability_summary": "rapid 4/4, verified 4/4",
                "flow_avg_prompt_calls": {"rapid": 6.0, "verified": 7.0},
                "prompt_tuned_failures": 0,
                "prompt_tuned_unstable_groups": 0,
                "flow_mode_failures": 0,
                "flow_mode_unstable_groups": 0,
            },
        }

    monkeypatch.setattr(
        "ageom.release_validation.run_release_validation",
        _fake_run_release_validation,
    )

    await _cmd_release_validate(argparse.Namespace(output="build/release_validation"))

    persisted = sorted(tmp_path.glob("run_*.json"))
    assert persisted
    payload = json.loads(persisted[-1].read_text(encoding="utf-8"))
    assert payload["pipeline"] == "release_validation"
    assert payload["status"] == "completed"
    assert payload["metadata"]["release_validation"]["status"] == "passed"
    assert payload["metadata"]["benchmark_validation"]["flow_results"] == 16
    assert payload["metadata"]["benchmark_validation"]["flow_avg_prompt_calls"]["verified"] == 7.0
