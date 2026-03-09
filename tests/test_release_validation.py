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
    assert manifest["warnings"]["warning_summary"].startswith("runtime=")
    assert manifest["warnings"]["top_runtime_warning"] == ""
    assert isinstance(manifest["warnings"]["top_catalog_warning"], str)
    assert manifest["failures"]["failure_summary"] == "none"
    assert manifest["failures"]["top_failed_check"] == "none"
    assert manifest["failures"]["top_benchmark_subcheck"] == ""
    assert manifest["failures"]["top_benchmark_failure"] == ""
    assert manifest["failures"]["top_runtime_failure"] == ""
    assert manifest["failures"]["top_catalog_failure"] == ""
    bench = manifest["checks"]["benchmark_validation"]
    runtime = manifest["checks"]["runtime_complexity"]
    catalog = manifest["checks"]["catalog_validation"]
    assert bench["prompt_results"] > 0
    assert bench["flow_results"] > 0
    assert "required[structured,verified]" in bench["flow_gate_summary"]
    assert "rapid=rapid_direct" in bench["flow_execution_path_summary"]
    assert "rapid=" in bench["flow_prompt_volume_summary"]
    assert "verified=5/0/0" in bench["runtime_override_policy_summary"]
    assert bench["flow_required_variants"] == ["structured", "verified"]
    assert set(bench["flow_comparison_variants"]) == {"direct_baseline", "rapid"}
    assert bench["flow_execution_paths"]["observed"]["rapid"] == ["rapid_direct"]
    assert bench["prompt_tuned_failures"] == 0
    assert bench["flow_mode_failures"] == 0
    assert runtime["provider_count"] > 0
    assert runtime["legacy_provider_count"] == 0
    assert runtime["violations"] == []
    assert catalog["status"] == "passed"
    assert catalog["source_candidates"] > 0
    assert "resolved=" in catalog["coverage_summary"]
    assert "matched=" in catalog["alignment_summary"]
    assert "warnings=" in catalog["warning_summary"]
    assert summary["warning_summary"] == manifest["warnings"]["warning_summary"]
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
            "runtime_override_policy_summary": "verified=5/1/1",
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
    async def _fake_run_catalog_validation(output_dir):
        return {
            "status": "passed",
            "report": str(tmp_path / "catalog" / "catalog_validation.json"),
            "configured_sources": 2,
            "resolved_sources": 2,
            "source_candidates": 10,
            "source_added": 8,
            "coverage_summary": "resolved=2/2 added=8/10 missing=0 zero=0",
            "alignment_summary": "severity=healthy matched=8 registry_only=0 ast_only=0 drift=0",
            "warning_summary": "warnings=0 high=0 medium=0",
            "missing_sources": [],
            "zero_candidate_sources": [],
            "high_severity_sources": [],
            "medium_severity_sources": [],
            "warnings": [],
            "violations": [],
        }

    monkeypatch.setattr(
        "ageom.release_validation.run_catalog_validation",
        _fake_run_catalog_validation,
    )

    summary = await run_release_validation(tmp_path)

    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["warnings"]["warning_summary"] == "runtime=0 catalog=0"
    assert manifest["warnings"]["top_runtime_warning"] == ""
    assert manifest["warnings"]["top_catalog_warning"] == ""
    assert manifest["failures"]["top_failed_check"] == "benchmark_validation"
    assert manifest["failures"]["top_benchmark_subcheck"] == "prompt_tuning"
    assert manifest["failures"]["failure_summary"].startswith(
        "check=benchmark_validation benchmark_check=prompt_tuning "
    )
    assert manifest["failures"]["top_benchmark_failure"] == "prompt_tuned_failures=1"
    assert manifest["failures"]["top_runtime_failure"] == ""
    assert manifest["failures"]["top_catalog_failure"] == ""
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
    async def _fake_run_catalog_validation(output_dir):
        return {
            "status": "passed",
            "report": str(tmp_path / "catalog" / "catalog_validation.json"),
            "configured_sources": 2,
            "resolved_sources": 2,
            "source_candidates": 10,
            "source_added": 8,
            "coverage_summary": "resolved=2/2 added=8/10 missing=0 zero=0",
            "alignment_summary": "severity=healthy matched=8 registry_only=0 ast_only=0 drift=0",
            "warning_summary": "warnings=0 high=0 medium=0",
            "missing_sources": [],
            "zero_candidate_sources": [],
            "high_severity_sources": [],
            "medium_severity_sources": [],
            "warnings": [],
            "violations": [],
        }

    monkeypatch.setattr(
        "ageom.release_validation.run_catalog_validation",
        _fake_run_catalog_validation,
    )

    summary = await run_release_validation(tmp_path)

    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["warnings"]["warning_summary"] == "runtime=2 top=provider_count=6 exceeds budget 4 catalog=0"
    assert manifest["warnings"]["top_runtime_warning"] == "provider_count=6 exceeds budget 4"
    assert manifest["warnings"]["top_catalog_warning"] == ""
    assert manifest["failures"]["top_failed_check"] == "runtime_complexity"
    assert manifest["failures"]["top_benchmark_subcheck"] == "runtime_budget"
    assert manifest["failures"]["failure_summary"].startswith(
        "check=runtime_complexity benchmark_check=runtime_budget "
    )
    assert manifest["failures"]["top_benchmark_failure"] == "provider_count=6 exceeds budget 4"
    assert manifest["failures"]["top_runtime_failure"] == "provider_count=6 exceeds budget 4"
    assert manifest["failures"]["top_catalog_failure"] == ""
    runtime = manifest["checks"]["runtime_complexity"]
    assert runtime["legacy_provider_count"] == 1
    assert any("provider_count=6 exceeds budget 4" == item for item in runtime["violations"])
    assert any("legacy_providers_present=codex_cli" == item for item in runtime["violations"])


@pytest.mark.asyncio
async def test_run_release_validation_surfaces_benchmark_execution_path_subcheck(
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
            "flow_execution_path_summary": "rapid=rapid_direct violations=1",
            "runtime_override_policy_summary": "verified=5/0/0",
            "flow_required_variants": ["structured", "verified"],
            "flow_comparison_variants": ["direct_baseline", "rapid"],
            "flow_execution_paths": {
                "expected": {"rapid": "rapid_direct"},
                "observed": {"rapid": ["verified_orchestration"]},
                "violations": ["rapid expected rapid_direct got verified_orchestration"],
            },
            "flow_prompt_volume": {
                "averages": {"rapid": 6.0, "structured": 7.0, "verified": 8.0},
                "violations": [],
            },
            "flow_prompt_volume_summary": "rapid=6.0, structured=7.0, verified=8.0",
            "flow_avg_prompt_calls": {"rapid": 6.0, "verified": 7.0},
            "runtime_complexity": {"violations": []},
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

    async def _fake_run_catalog_validation(output_dir):
        return {
            "status": "passed",
            "report": str(tmp_path / "catalog" / "catalog_validation.json"),
            "configured_sources": 2,
            "resolved_sources": 2,
            "source_candidates": 10,
            "source_added": 8,
            "coverage_summary": "resolved=2/2 added=8/10 missing=0 zero=0",
            "alignment_summary": "severity=healthy matched=8 registry_only=0 ast_only=0 drift=0",
            "warning_summary": "warnings=0 high=0 medium=0",
            "missing_sources": [],
            "zero_candidate_sources": [],
            "high_severity_sources": [],
            "medium_severity_sources": [],
            "warnings": [],
            "violations": [],
        }

    monkeypatch.setattr(
        "ageom.release_validation.run_catalog_validation",
        _fake_run_catalog_validation,
    )

    summary = await run_release_validation(tmp_path)

    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failures"]["top_failed_check"] == "benchmark_validation"
    assert manifest["failures"]["top_benchmark_subcheck"] == "execution_path"
    assert manifest["failures"]["top_benchmark_failure"] == (
        "rapid expected rapid_direct got verified_orchestration"
    )
    assert manifest["failures"]["failure_summary"].startswith(
        "check=benchmark_validation benchmark_check=execution_path "
    )


@pytest.mark.asyncio
async def test_run_release_validation_fails_when_catalog_validation_fails(
    monkeypatch, tmp_path
):
    async def _fake_run_benchmark_validation(output_dir):
        return {
            "status": "passed",
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
            "runtime_complexity": {"violations": []},
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
    async def _fake_run_catalog_validation(output_dir):
        return {
            "status": "failed",
            "report": str(tmp_path / "catalog" / "catalog_validation.json"),
            "configured_sources": 2,
            "resolved_sources": 1,
            "source_candidates": 3,
            "source_added": 3,
            "coverage_summary": "resolved=1/2 added=3/3 missing=1 zero=1",
            "alignment_summary": "severity=critical matched=3 registry_only=0 ast_only=1 drift=1",
            "warning_summary": "warnings=0 high=0 medium=0",
            "missing_sources": ["hpy-atoms"],
            "zero_candidate_sources": ["hpy-atoms"],
            "high_severity_sources": [],
            "medium_severity_sources": [],
            "warnings": [],
            "violations": ["missing_source:hpy-atoms", "source_no_candidates:hpy-atoms"],
        }

    monkeypatch.setattr(
        "ageom.release_validation.run_catalog_validation",
        _fake_run_catalog_validation,
    )

    summary = await run_release_validation(tmp_path)

    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["warnings"]["warning_summary"] == "runtime=0 catalog=0"
    assert manifest["warnings"]["top_runtime_warning"] == ""
    assert manifest["warnings"]["top_catalog_warning"] == ""
    assert manifest["failures"]["top_failed_check"] == "catalog_validation"
    assert manifest["failures"]["top_benchmark_subcheck"] == ""
    assert manifest["failures"]["failure_summary"].startswith("check=catalog_validation ")
    assert manifest["failures"]["top_benchmark_failure"] == ""
    assert manifest["failures"]["top_runtime_failure"] == ""
    assert manifest["failures"]["top_catalog_failure"] == "missing_source:hpy-atoms"
    assert manifest["checks"]["catalog_validation"]["status"] == "failed"
    assert "missing_source:hpy-atoms" in manifest["checks"]["catalog_validation"]["violations"]


@pytest.mark.asyncio
async def test_run_release_validation_fails_when_catalog_alignment_is_critical(
    monkeypatch, tmp_path
):
    async def _fake_run_benchmark_validation(output_dir):
        return {
            "status": "passed",
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
            "runtime_complexity": {"violations": []},
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

    async def _fake_run_catalog_validation(output_dir):
        return {
            "status": "failed",
            "report": str(tmp_path / "catalog" / "catalog_validation.json"),
            "configured_sources": 1,
            "resolved_sources": 1,
            "source_candidates": 5,
            "source_added": 5,
            "coverage_summary": "resolved=1/1 added=5/5 missing=0 zero=0",
            "alignment_summary": "severity=critical matched=4 registry_only=0 ast_only=1 drift=1",
            "warning_summary": "warnings=0 high=0 medium=0",
            "missing_sources": [],
            "zero_candidate_sources": [],
            "high_severity_sources": [],
            "medium_severity_sources": [],
            "warnings": [],
            "violations": ["critical_alignment_drift"],
        }

    monkeypatch.setattr(
        "ageom.release_validation.run_catalog_validation",
        _fake_run_catalog_validation,
    )

    summary = await run_release_validation(tmp_path)

    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["warnings"]["warning_summary"] == "runtime=0 catalog=0"
    assert manifest["warnings"]["top_runtime_warning"] == ""
    assert manifest["warnings"]["top_catalog_warning"] == ""
    assert manifest["failures"]["top_failed_check"] == "catalog_validation"
    assert manifest["failures"]["top_benchmark_subcheck"] == ""
    assert manifest["failures"]["failure_summary"].startswith("check=catalog_validation ")
    assert manifest["failures"]["top_benchmark_failure"] == ""
    assert manifest["failures"]["top_runtime_failure"] == ""
    assert manifest["failures"]["top_catalog_failure"] == "critical_alignment_drift"
    assert manifest["checks"]["catalog_validation"]["violations"] == [
        "critical_alignment_drift"
    ]
