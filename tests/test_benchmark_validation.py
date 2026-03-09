from __future__ import annotations

import json

import pytest

from ageom.benchmark_validation import (
    benchmark_failure_summary,
    format_benchmark_failure_summary,
    flow_prompt_volume_summary,
    run_benchmark_validation,
    runtime_complexity_summary,
)
from ageom.config import AgeomConfig
from ageom.flow_benchmark import (
    default_flow_benchmark_cases,
    run_flow_benchmark,
    save_flow_benchmark_report,
    summarize_flow_benchmark,
)


def test_save_flow_benchmark_report(tmp_path):
    report_path = tmp_path / "flow_benchmark.json"
    aggregates = []
    save_flow_benchmark_report(report_path, results=[], aggregates=aggregates)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["results"] == []
    assert payload["aggregates"] == []


@pytest.mark.asyncio
async def test_run_benchmark_validation_writes_bundle(tmp_path):
    summary = await run_benchmark_validation(tmp_path)

    assert summary["prompt_cases"] > 0
    assert summary["flow_cases"] == len(default_flow_benchmark_cases())
    assert (tmp_path / "prompt_benchmark.json").exists()
    assert (tmp_path / "flow_benchmark.json").exists()
    assert (tmp_path / "summary.json").exists()

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["prompt_results"] == summary["prompt_results"]
    assert payload["flow_results"] == summary["flow_results"]
    assert "prompt_stability_summary" in payload
    assert "flow_stability_summary" in payload
    assert "flow_avg_prompt_calls" in payload
    assert "prompt_avg_latency_ms" in payload
    assert "flow_avg_latency_ms" in payload
    assert "runtime_complexity" in payload
    assert "failure_summary" in payload
    assert "top_failed_subcheck" in payload
    assert "top_failure" in payload
    assert "prompt_tuned_failures" in payload
    assert "flow_mode_failures" in payload
    assert payload["flow_required_variants"] == ["structured", "verified"]
    assert set(payload["flow_comparison_variants"]) == {"direct_baseline", "rapid"}
    assert payload["flow_execution_paths"]["observed"]["rapid"] == ["rapid_direct"]
    assert payload["flow_execution_paths"]["observed"]["structured"] == ["structured_single_pass"]
    assert payload["flow_execution_paths"]["observed"]["verified"] == ["verified_orchestration"]
    assert payload["flow_execution_paths"]["violations"] == []
    assert "required[structured,verified]" in payload["flow_gate_summary"]
    assert "rapid=rapid_direct" in payload["flow_execution_path_summary"]
    assert "rapid=" in payload["flow_prompt_volume_summary"]
    assert "verified=5/0/0" in payload["runtime_override_policy_summary"]
    assert "flow_comparison_failures" in payload
    assert "flow_comparison_unstable_groups" in payload
    assert set(payload["flow_avg_prompt_calls"]) == {
        "direct_baseline",
        "rapid",
        "structured",
        "verified",
    }
    assert set(payload["flow_avg_latency_ms"]) == {
        "direct_baseline",
        "rapid",
        "structured",
        "verified",
    }
    assert set(payload["runtime_complexity"]["by_mode"]) == {
        "rapid",
        "structured",
        "verified",
    }
    assert isinstance(payload["runtime_complexity"]["monotonic_violations"], list)
    assert payload["runtime_complexity"]["by_mode"]["verified"]["override_policy"]["missing_required_overrides"] == []
    assert payload["runtime_complexity"]["by_mode"]["verified"]["override_policy"]["unexpected_active_overrides"] == []
    assert payload["failure_summary"] == "none"
    assert payload["top_failed_subcheck"] == ""
    assert payload["top_failure"] == ""


def test_runtime_complexity_summary_is_mode_aware():
    summary = runtime_complexity_summary(AgeomConfig(_env_file=None))

    assert set(summary["by_mode"]) == {"rapid", "structured", "verified"}
    assert summary["by_mode"]["rapid"]["mode"] == "rapid"
    assert summary["by_mode"]["rapid"]["provider_count"] == 1
    assert summary["by_mode"]["rapid"]["provider_model_count"] == 1
    assert summary["by_mode"]["rapid"]["transport_count"] == 1
    assert summary["by_mode"]["rapid"]["providers"] == ["anthropic"]
    assert summary["by_mode"]["rapid"]["budget"]["max_provider_count"] == 1
    assert summary["by_mode"]["structured"]["provider_count"] == 1
    assert summary["by_mode"]["structured"]["providers"] == ["anthropic"]
    assert summary["by_mode"]["structured"]["budget"]["max_provider_count"] == 1
    assert summary["by_mode"]["verified"]["mode"] == "verified"
    assert summary["by_mode"]["verified"]["override_policy"]["missing_required_overrides"] == []
    assert summary["by_mode"]["verified"]["override_policy"]["unexpected_active_overrides"] == []
    assert "monotonic_violations" in summary


def test_runtime_complexity_summary_flags_verified_override_policy_drift():
    summary = runtime_complexity_summary(
        AgeomConfig(
            _env_file=None,
            hunter_reformulate_llm_provider="anthropic",
            hunter_reformulate_llm_model="claude-sonnet-4-5-20250929",
        )
    )

    verified = summary["by_mode"]["verified"]
    assert {
        row["prompt_key"]: row["provider"]
        for row in verified["override_policy"]["missing_required_overrides"]
    } == {"hunter_reformulate": "gemini_shim"}
    assert {
        row["prompt_key"]: row["provider"]
        for row in verified["override_policy"]["unexpected_active_overrides"]
    } == {"hunter_reformulate": "anthropic"}
    assert "verified:missing_required_override:hunter_reformulate=gemini_shim" in summary["violations"]
    assert "verified:unexpected_active_override:hunter_reformulate=anthropic" in summary["violations"]


def test_flow_prompt_volume_summary_flags_non_monotonic_modes():
    class _Agg:
        def __init__(self, variant: str, avg_prompt_calls: float):
            self.variant = variant
            self.avg_prompt_calls = avg_prompt_calls

    summary = flow_prompt_volume_summary(
        [
            _Agg("rapid", 5.0),
            _Agg("structured", 4.0),
            _Agg("verified", 7.0),
        ]
    )

    assert summary["averages"]["rapid"] == 5.0
    assert "rapid_prompt_calls=5.0 exceeds structured=4.0" in summary["violations"]


def test_benchmark_failure_summary_prefers_execution_path_before_prompt_counts():
    summary = benchmark_failure_summary(
        {
            "runtime_complexity": {"violations": []},
            "flow_execution_paths": {
                "violations": ["rapid:expected rapid_direct but observed verified_orchestration"]
            },
            "flow_prompt_volume": {"violations": []},
            "prompt_tuned_failures": 2,
            "flow_mode_failures": 1,
        }
    )

    assert summary == {
        "top_failed_subcheck": "execution_path",
        "top_failure": "rapid:expected rapid_direct but observed verified_orchestration",
    }


def test_format_benchmark_failure_summary_renders_compact_line():
    rendered = format_benchmark_failure_summary(
        {
            "top_failed_subcheck": "runtime_budget",
            "top_failure": "legacy_providers_present=codex_cli",
        }
    )

    assert rendered == (
        "subcheck=runtime_budget failure=legacy_providers_present=codex_cli"
    )


@pytest.mark.asyncio
async def test_flow_benchmark_report_contains_all_variants(tmp_path):
    results = await run_flow_benchmark(cases=default_flow_benchmark_cases())
    aggregates = summarize_flow_benchmark(results)
    report_path = tmp_path / "flow_benchmark.json"
    save_flow_benchmark_report(report_path, results=results, aggregates=aggregates)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    variants = {row["variant"] for row in payload["aggregates"]}
    assert variants == {"direct_baseline", "rapid", "structured", "verified"}
