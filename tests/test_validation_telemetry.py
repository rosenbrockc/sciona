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
            "status": "passed",
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
            "flow_gate_summary": "required[structured,verified] 0/0; comparison[direct_baseline,rapid] 2/0",
            "flow_execution_path_summary": "rapid=rapid_direct",
            "runtime_override_policy_summary": "verified=5/0/0",
            "flow_required_variants": ["structured", "verified"],
            "flow_comparison_variants": ["direct_baseline", "rapid"],
            "flow_execution_paths": {
                "expected": {"rapid": "rapid_direct"},
                "observed": {"rapid": ["rapid_direct"]},
                "violations": [],
            },
            "flow_prompt_volume": {
                "averages": {"rapid": 6.0, "structured": 7.0, "verified": 8.0},
                "violations": [],
            },
            "flow_prompt_volume_summary": "rapid=6.0, structured=7.0, verified=8.0",
            "flow_avg_prompt_calls": {"rapid": 6.0, "verified": 7.0},
            "runtime_complexity": {"violations": [], "provider_count": 3},
            "prompt_tuned_failures": 0,
            "prompt_tuned_unstable_groups": 0,
            "flow_mode_failures": 0,
            "flow_mode_unstable_groups": 0,
            "flow_comparison_failures": 2,
            "flow_comparison_unstable_groups": 0,
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
    assert payload["status"] == "completed"
    assert bench["status"] == "passed"
    assert bench["prompt_results"] == 24
    assert bench["flow_results"] == 16
    assert bench["flow_avg_prompt_calls"]["rapid"] == 6.0
    assert bench["flow_mode_failures"] == 0
    assert "required[structured,verified]" in bench["flow_gate_summary"]
    assert "rapid=rapid_direct" in bench["flow_execution_path_summary"]
    assert "verified=8.0" in bench["flow_prompt_volume_summary"]
    assert "verified=5/0/0" in bench["runtime_override_policy_summary"]
    assert bench["flow_required_variants"] == ["structured", "verified"]
    assert set(bench["flow_comparison_variants"]) == {"direct_baseline", "rapid"}
    assert bench["flow_execution_paths"]["observed"]["rapid"] == ["rapid_direct"]
    assert bench["runtime_complexity"]["provider_count"] == 3


@pytest.mark.asyncio
async def test_benchmark_validate_fails_telemetry_when_runtime_budget_fails(monkeypatch, tmp_path):
    from ageom.cli import _cmd_benchmark_validate
    from ageom.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

    async def _fake_run_benchmark_validation(output_dir):
        return {
            "status": "failed",
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
            "flow_gate_summary": "required[structured,verified] 0/0; comparison[direct_baseline,rapid] 2/0",
            "flow_execution_path_summary": "rapid=rapid_direct",
            "runtime_override_policy_summary": "verified=5/1/1",
            "flow_required_variants": ["structured", "verified"],
            "flow_comparison_variants": ["direct_baseline", "rapid"],
            "flow_execution_paths": {
                "expected": {"rapid": "rapid_direct"},
                "observed": {"rapid": ["rapid_direct"]},
                "violations": [],
            },
            "flow_prompt_volume": {
                "averages": {"rapid": 8.0, "structured": 7.0, "verified": 9.0},
                "violations": ["rapid_prompt_calls=8.0 exceeds structured=7.0"],
            },
            "flow_prompt_volume_summary": "rapid=8.0, structured=7.0, verified=9.0 violations=1",
            "flow_avg_prompt_calls": {"rapid": 6.0, "verified": 7.0},
            "runtime_complexity": {
                "provider_count": 5,
                "legacy_provider_count": 1,
                "violations": ["legacy_providers_present=codex_cli"],
            },
            "prompt_tuned_failures": 0,
            "prompt_tuned_unstable_groups": 0,
            "flow_mode_failures": 0,
            "flow_mode_unstable_groups": 0,
            "flow_comparison_failures": 2,
            "flow_comparison_unstable_groups": 0,
        }

    monkeypatch.setattr(
        "ageom.benchmark_validation.run_benchmark_validation",
        _fake_run_benchmark_validation,
    )

    with pytest.raises(RuntimeError, match="benchmark validation failed"):
        await _cmd_benchmark_validate(argparse.Namespace(output="build/benchmark_validation"))

    persisted = sorted(tmp_path.glob("run_*.json"))
    assert persisted
    payload = json.loads(persisted[-1].read_text(encoding="utf-8"))
    assert payload["pipeline"] == "benchmark_validation"
    assert payload["status"] == "failed"
    assert payload["metadata"]["benchmark_validation"]["status"] == "failed"
    assert payload["metadata"]["benchmark_validation"]["runtime_complexity"]["legacy_provider_count"] == 1


@pytest.mark.asyncio
async def test_release_validate_writes_telemetry_metadata(monkeypatch, tmp_path):
    from ageom.cli import _cmd_release_validate
    from ageom.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

    async def _fake_run_release_validation(output_dir):
        return {
            "status": "failed",
            "manifest": "build/release_validation/release_validation.json",
            "benchmarks_dir": "build/release_validation/benchmarks",
            "warning_summary": "runtime=1 top=legacy_providers_present=codex_cli catalog=0",
            "runtime_warning_count": 1,
            "catalog_warning_count": 0,
            "top_runtime_warning": "legacy_providers_present=codex_cli",
            "top_catalog_warning": "",
            "failure_summary": "check=runtime_complexity benchmark_check=runtime_budget benchmark=legacy_providers_present=codex_cli runtime=legacy_providers_present=codex_cli catalog=missing_source:hpy-atoms",
            "top_failed_check": "runtime_complexity",
            "top_benchmark_subcheck": "runtime_budget",
            "top_benchmark_failure": "legacy_providers_present=codex_cli",
            "top_runtime_failure": "legacy_providers_present=codex_cli",
            "top_catalog_failure": "missing_source:hpy-atoms",
            "catalog_validation": {
                "status": "failed",
                "report": "build/release_validation/catalog/catalog_validation.json",
                "configured_sources": 2,
                "resolved_sources": 1,
                "source_candidates": 3,
                "source_added": 3,
                "coverage_summary": "resolved=1/2 added=3/3 missing=1 zero=1",
                "alignment_summary": "severity=critical matched=3 registry_only=0 ast_only=1 drift=1",
                "warning_summary": "warnings=0 high=0 medium=0",
                "high_severity_sources": [],
                "medium_severity_sources": [],
                "warnings": [],
                "missing_sources": ["hpy-atoms"],
                "zero_candidate_sources": ["hpy-atoms"],
                "violations": ["missing_source:hpy-atoms"],
            },
            "runtime_complexity": {
                "provider_count": 5,
                "provider_model_count": 6,
                "transport_count": 4,
                "legacy_provider_count": 1,
                "legacy_providers": ["codex_cli"],
                "violations": ["legacy_providers_present=codex_cli"],
            },
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
                "flow_gate_summary": "required[structured,verified] 0/0; comparison[direct_baseline,rapid] 2/0",
                "flow_execution_path_summary": "rapid=rapid_direct",
                "runtime_override_policy_summary": "verified=5/1/1",
                "flow_required_variants": ["structured", "verified"],
                "flow_comparison_variants": ["direct_baseline", "rapid"],
                "flow_execution_paths": {
                    "expected": {"rapid": "rapid_direct"},
                    "observed": {"rapid": ["rapid_direct"]},
                    "violations": [],
                },
                "flow_prompt_volume": {
                    "averages": {"rapid": 8.0, "structured": 7.0, "verified": 9.0},
                    "violations": ["rapid_prompt_calls=8.0 exceeds structured=7.0"],
                },
                "flow_prompt_volume_summary": "rapid=8.0, structured=7.0, verified=9.0 violations=1",
                "flow_avg_prompt_calls": {"rapid": 6.0, "verified": 7.0},
                "status": "failed",
                "runtime_complexity": {"provider_count": 5, "violations": ["legacy_providers_present=codex_cli"]},
                "prompt_tuned_failures": 0,
                "prompt_tuned_unstable_groups": 0,
                "flow_mode_failures": 0,
                "flow_mode_unstable_groups": 0,
                "flow_comparison_failures": 2,
                "flow_comparison_unstable_groups": 0,
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
    assert payload["metadata"]["release_validation"]["status"] == "failed"
    assert payload["metadata"]["release_validation"]["warning_summary"] == "runtime=1 top=legacy_providers_present=codex_cli catalog=0"
    assert payload["metadata"]["release_validation"]["top_runtime_warning"] == "legacy_providers_present=codex_cli"
    assert payload["metadata"]["release_validation"]["top_catalog_warning"] == ""
    assert payload["metadata"]["release_validation"]["failure_summary"].startswith(
        "check=runtime_complexity benchmark_check=runtime_budget "
    )
    assert payload["metadata"]["release_validation"]["top_failed_check"] == "runtime_complexity"
    assert payload["metadata"]["release_validation"]["top_benchmark_subcheck"] == "runtime_budget"
    assert payload["metadata"]["release_validation"]["top_benchmark_failure"] == "legacy_providers_present=codex_cli"
    assert payload["metadata"]["release_validation"]["top_runtime_failure"] == "legacy_providers_present=codex_cli"
    assert payload["metadata"]["release_validation"]["top_catalog_failure"] == "missing_source:hpy-atoms"
    assert payload["metadata"]["release_validation"]["catalog_validation"]["status"] == "failed"
    assert payload["metadata"]["release_validation"]["runtime_complexity"]["legacy_provider_count"] == 1
    assert payload["metadata"]["benchmark_validation"]["flow_results"] == 16
    assert payload["metadata"]["benchmark_validation"]["flow_avg_prompt_calls"]["verified"] == 7.0
    assert payload["metadata"]["benchmark_validation"]["status"] == "failed"
