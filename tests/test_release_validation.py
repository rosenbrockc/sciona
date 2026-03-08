from __future__ import annotations

import json
from pathlib import Path

import pytest

from ageom.release_validation import run_release_validation


@pytest.mark.asyncio
async def test_run_release_validation_writes_manifest_and_benchmark_bundle(tmp_path):
    summary = await run_release_validation(tmp_path)

    manifest_path = tmp_path / "release_validation.json"
    assert summary["manifest"] == str(manifest_path)
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    bench = manifest["checks"]["benchmark_validation"]
    runtime = manifest["checks"]["runtime_complexity"]
    assert bench["prompt_results"] > 0
    assert bench["flow_results"] > 0
    assert "required[structured,verified]" in bench["flow_gate_summary"]
    assert "rapid=rapid_direct" in bench["flow_execution_path_summary"]
    assert bench["flow_required_variants"] == ["structured", "verified"]
    assert set(bench["flow_comparison_variants"]) == {"direct_baseline", "rapid"}
    assert bench["flow_execution_paths"]["observed"]["rapid"] == ["rapid_direct"]
    assert bench["prompt_tuned_failures"] == 0
    assert bench["flow_mode_failures"] == 0
    assert runtime["provider_count"] > 0
    assert runtime["legacy_provider_count"] == 0
    assert runtime["violations"] == []
    assert (tmp_path / "benchmarks" / "summary.json").exists()


@pytest.mark.asyncio
async def test_run_release_validation_fails_when_nonbaseline_regressions_exist(
    monkeypatch, tmp_path
):
    async def _fake_run_benchmark_validation(output_dir):
        return {
            "status": "failed",
            "summary_report": str(tmp_path / "benchmarks" / "summary.json"),
            "prompt_report": str(tmp_path / "benchmarks" / "prompt_benchmark.json"),
            "flow_report": str(tmp_path / "benchmarks" / "flow_benchmark.json"),
            "prompt_cases": 12,
            "prompt_results": 24,
            "prompt_summary": "prompt summary",
            "prompt_stability_summary": "fixture_good/tuned 10/12",
            "flow_cases": 4,
            "flow_results": 16,
            "flow_summary": "flow summary",
            "flow_stability_summary": "rapid 3/4, verified 4/4",
            "flow_gate_summary": "required[structured,verified] 0/1; comparison[direct_baseline,rapid] 2/0",
            "flow_execution_path_summary": "rapid=rapid_direct",
            "flow_required_variants": ["structured", "verified"],
            "flow_comparison_variants": ["direct_baseline", "rapid"],
            "flow_execution_paths": {
                "expected": {"rapid": "rapid_direct"},
                "observed": {"rapid": ["rapid_direct"]},
                "violations": [],
            },
            "flow_avg_prompt_calls": {"rapid": 6.0, "verified": 7.0},
            "runtime_complexity": {"violations": []},
            "prompt_tuned_failures": 1,
            "prompt_tuned_unstable_groups": 2,
            "flow_mode_failures": 0,
            "flow_mode_unstable_groups": 1,
            "flow_comparison_failures": 2,
            "flow_comparison_unstable_groups": 0,
        }

    monkeypatch.setattr(
        "ageom.release_validation.run_benchmark_validation",
        _fake_run_benchmark_validation,
    )

    summary = await run_release_validation(tmp_path)

    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    bench = manifest["checks"]["benchmark_validation"]
    assert bench["prompt_tuned_failures"] == 1
    assert bench["prompt_tuned_unstable_groups"] == 2
    assert bench["flow_mode_unstable_groups"] == 1


@pytest.mark.asyncio
async def test_run_release_validation_fails_when_runtime_complexity_budget_exceeded(
    monkeypatch, tmp_path
):
    async def _fake_run_benchmark_validation(output_dir):
        return {
            "status": "failed",
            "summary_report": str(tmp_path / "benchmarks" / "summary.json"),
            "prompt_report": str(tmp_path / "benchmarks" / "prompt_benchmark.json"),
            "flow_report": str(tmp_path / "benchmarks" / "flow_benchmark.json"),
            "prompt_cases": 12,
            "prompt_results": 24,
            "prompt_summary": "prompt summary",
            "prompt_stability_summary": "fixture_good/tuned 12/12",
            "flow_cases": 4,
            "flow_results": 16,
            "flow_summary": "flow summary",
            "flow_stability_summary": "rapid 4/4, verified 4/4",
            "flow_gate_summary": "required[structured,verified] 0/0; comparison[direct_baseline,rapid] 2/0",
            "flow_execution_path_summary": "rapid=rapid_direct",
            "flow_required_variants": ["structured", "verified"],
            "flow_comparison_variants": ["direct_baseline", "rapid"],
            "flow_execution_paths": {
                "expected": {"rapid": "rapid_direct"},
                "observed": {"rapid": ["rapid_direct"]},
                "violations": [],
            },
            "flow_avg_prompt_calls": {"rapid": 6.0, "verified": 7.0},
            "runtime_complexity": {
                "provider_count": 6,
                "provider_model_count": 7,
                "transport_count": 4,
                "legacy_provider_count": 1,
                "legacy_providers": ["codex_cli"],
                "budget": {
                    "max_provider_count": 4,
                    "max_provider_model_count": 5,
                    "max_transport_count": 3,
                    "allow_legacy_providers": False,
                },
                "violations": [
                    "provider_count=6 exceeds budget 4",
                    "legacy_providers_present=codex_cli",
                ],
            },
            "prompt_tuned_failures": 0,
            "prompt_tuned_unstable_groups": 0,
            "flow_mode_failures": 0,
            "flow_mode_unstable_groups": 0,
            "flow_comparison_failures": 2,
            "flow_comparison_unstable_groups": 0,
        }

    monkeypatch.setattr(
        "ageom.release_validation.run_benchmark_validation",
        _fake_run_benchmark_validation,
    )

    summary = await run_release_validation(tmp_path)

    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    runtime = manifest["checks"]["runtime_complexity"]
    assert runtime["legacy_provider_count"] == 1
    assert any("provider_count=6 exceeds budget 4" == item for item in runtime["violations"])
    assert any("legacy_providers_present=codex_cli" == item for item in runtime["violations"])
