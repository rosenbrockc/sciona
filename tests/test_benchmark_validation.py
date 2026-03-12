from __future__ import annotations

import json

import pytest

from ageom.benchmark_validation import (
    _format_single_agent_comparison_summary,
    benchmark_failure_summary,
    benchmark_warning_summary,
    format_benchmark_failure_summary,
    format_benchmark_health_summary,
    format_benchmark_warning_summary,
    flow_prompt_volume_summary,
    run_benchmark_validation,
    single_agent_comparison_summary,
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
    assert "flow_avg_planner_tool_dispatches" in payload
    assert "flow_avg_planner_tool_latency_ms" in payload
    assert "flow_avg_planner_escalations" in payload
    assert "prompt_avg_latency_ms" in payload
    assert "flow_avg_latency_ms" in payload
    assert "single_agent_comparison" in payload
    assert "single_agent_comparison_summary" in payload
    assert "runtime_complexity" in payload
    assert "health_summary" in payload
    assert "warning_summary" in payload
    assert "top_warning_subcheck" in payload
    assert "top_warning" in payload
    assert "failure_summary" in payload
    assert "top_failed_subcheck" in payload
    assert "top_failure" in payload
    assert "prompt_tuned_failures" in payload
    assert "flow_mode_failures" in payload
    assert payload["flow_required_variants"] == ["structured", "verified"]
    assert set(payload["flow_comparison_variants"]) == {
        "direct_baseline",
        "rapid",
        "single_agent",
    }
    assert payload["flow_execution_paths"]["observed"]["rapid"] == ["rapid_direct"]
    assert payload["flow_execution_paths"]["observed"]["single_agent"] == [
        "single_agent_structured"
    ]
    assert payload["flow_execution_paths"]["observed"]["structured"] == ["structured_single_pass"]
    assert payload["flow_execution_paths"]["observed"]["verified"] == ["verified_orchestration"]
    assert payload["flow_execution_paths"]["violations"] == []
    assert payload["coverage_monotonic"] is True
    assert payload["coverage_by_variant"]["structured"] >= payload["coverage_by_variant"]["rapid"]
    assert payload["coverage_by_variant"]["verified"] >= payload["coverage_by_variant"]["structured"]
    assert "required[structured,verified]" in payload["flow_gate_summary"]
    assert "rapid=rapid_direct" in payload["flow_execution_path_summary"]
    assert "single_agent=single_agent_structured" in payload["flow_execution_path_summary"]
    assert "rapid=" in payload["flow_prompt_volume_summary"]
    assert "verified=0/0/0" in payload["runtime_override_policy_summary"]
    assert payload["single_agent_comparison"]["present"] is True
    assert payload["single_agent_comparison"]["comparisons"]["rapid"]["pass_rate_delta"] >= 0.0
    assert payload["single_agent_comparison"]["comparisons"]["structured"]["pass_rate_delta"] <= 0.0
    assert payload["single_agent_comparison"]["dominant_planner_termination_reason"] == (
        "structured_verified"
    )
    assert payload["single_agent_comparison"]["planner_action_summary"] == (
        "decompose:7,match_decomposed:7"
    )
    assert "vs_structured=" in payload["single_agent_comparison_summary"]
    assert "flow_comparison_failures" in payload
    assert "flow_comparison_unstable_groups" in payload
    assert set(payload["flow_avg_prompt_calls"]) == {
        "direct_baseline",
        "rapid",
        "single_agent",
        "structured",
        "verified",
    }
    assert set(payload["flow_avg_planner_tool_dispatches"]) == {
        "direct_baseline",
        "rapid",
        "single_agent",
        "structured",
        "verified",
    }
    assert payload["flow_avg_planner_tool_dispatches"]["single_agent"] > 0.0
    assert payload["flow_avg_planner_escalations"]["single_agent"] >= 0.0
    assert set(payload["flow_avg_latency_ms"]) == {
        "direct_baseline",
        "rapid",
        "single_agent",
        "structured",
        "verified",
    }
    assert set(payload["runtime_complexity"]["by_mode"]) == {
        "rapid",
        "single_agent",
        "structured",
        "verified",
    }
    assert isinstance(payload["runtime_complexity"]["monotonic_violations"], list)
    assert payload["runtime_complexity"]["by_mode"]["verified"]["override_policy"]["missing_required_overrides"] == []
    assert payload["runtime_complexity"]["by_mode"]["verified"]["override_policy"]["unexpected_active_overrides"] == []
    assert payload["warning_summary"] == (
        "subcheck=comparison_failures "
        f"warning=flow_comparison_failures={payload['flow_comparison_failures']}"
    )
    assert payload["health_summary"] == (
        "warnings="
        f"{payload['warning_summary']} "
        "failures=none"
    )
    assert payload["top_warning_subcheck"] == "comparison_failures"
    assert payload["top_warning"] == (
        f"flow_comparison_failures={payload['flow_comparison_failures']}"
    )
    assert payload["failure_summary"] == "none"
    assert payload["top_failed_subcheck"] == ""
    assert payload["top_failure"] == ""


def test_runtime_complexity_summary_is_mode_aware():
    summary = runtime_complexity_summary(AgeomConfig(_env_file=None))

    assert set(summary["by_mode"]) == {"rapid", "single_agent", "structured", "verified"}
    assert summary["by_mode"]["rapid"]["mode"] == "rapid"
    assert summary["by_mode"]["rapid"]["provider_count"] == 1
    assert summary["by_mode"]["rapid"]["provider_model_count"] == 1
    assert summary["by_mode"]["rapid"]["transport_count"] == 1
    assert summary["by_mode"]["rapid"]["providers"] == ["anthropic"]
    assert summary["by_mode"]["rapid"]["budget"]["max_provider_count"] == 1
    assert summary["by_mode"]["single_agent"]["mode"] == "single_agent"
    assert summary["by_mode"]["single_agent"]["provider_count"] == 1
    assert summary["by_mode"]["single_agent"]["providers"] == ["anthropic"]
    assert summary["by_mode"]["single_agent"]["budget"]["max_provider_count"] == 1
    assert summary["by_mode"]["single_agent"]["override_policy"]["missing_required_overrides"] == []
    assert summary["by_mode"]["structured"]["provider_count"] == 1
    assert summary["by_mode"]["structured"]["providers"] == ["anthropic"]
    assert summary["by_mode"]["structured"]["budget"]["max_provider_count"] == 1
    assert summary["by_mode"]["verified"]["mode"] == "verified"
    assert summary["by_mode"]["verified"]["override_policy"]["missing_required_overrides"] == []
    assert summary["by_mode"]["verified"]["override_policy"]["unexpected_active_overrides"] == []
    assert "monotonic_violations" in summary


def test_runtime_complexity_summary_treats_explicit_verified_prompt_override_as_nonblocking():
    summary = runtime_complexity_summary(
        AgeomConfig(
            _env_file=None,
            hunter_reformulate_llm_provider="anthropic",
            hunter_reformulate_llm_model="claude-sonnet-4-5-20250929",
        )
    )

    verified = summary["by_mode"]["verified"]
    assert verified["override_policy"]["required_active_overrides"] == []
    assert verified["override_policy"]["missing_required_overrides"] == []
    assert verified["override_policy"]["unexpected_active_overrides"] == []
    assert verified["active_override_count"] == 1
    assert verified["active_overrides"] == [
        {
            "prompt_key": "hunter_reformulate",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
        }
    ]
    assert all(
        "missing_required_override" not in violation
        and "unexpected_active_override" not in violation
        for violation in summary["violations"]
    )


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


def test_single_agent_comparison_summary_positions_mode_between_benchmarks():
    class _Agg:
        def __init__(
            self,
            variant: str,
            passed_cases: int,
            total_cases: int,
            avg_leaf_coverage: float,
            avg_prompt_calls: float,
            avg_latency_ms: float,
            avg_planner_tool_dispatches: float = 0.0,
            avg_planner_tool_latency_ms: float = 0.0,
            avg_planner_escalations: float = 0.0,
            dominant_planner_termination_reason: str = "",
            dominant_planner_action_signature: str = "",
            planner_action_counts: dict[str, int] | None = None,
        ):
            self.variant = variant
            self.passed_cases = passed_cases
            self.total_cases = total_cases
            self.avg_leaf_coverage = avg_leaf_coverage
            self.avg_prompt_calls = avg_prompt_calls
            self.avg_latency_ms = avg_latency_ms
            self.avg_planner_tool_dispatches = avg_planner_tool_dispatches
            self.avg_planner_tool_latency_ms = avg_planner_tool_latency_ms
            self.avg_planner_escalations = avg_planner_escalations
            self.dominant_planner_termination_reason = dominant_planner_termination_reason
            self.dominant_planner_action_signature = dominant_planner_action_signature
            self.planner_action_counts = planner_action_counts or {}

    summary = single_agent_comparison_summary(
        [
            _Agg("rapid", 0, 4, 0.2, 1.0, 100.0),
            _Agg(
                "single_agent",
                4,
                4,
                1.0,
                3.0,
                250.0,
                4.0,
                180.0,
                1.0,
                "structured_verified",
                "decompose>match_decomposed",
                {"decompose": 4, "match_decomposed": 4},
            ),
            _Agg("structured", 4, 4, 1.0, 4.0, 300.0),
            _Agg("verified", 4, 4, 1.0, 5.0, 350.0),
        ]
    )

    assert summary["present"] is True
    assert summary["pass_rate"] == 1.0
    assert summary["avg_planner_tool_dispatches"] == 4.0
    assert summary["avg_planner_tool_latency_ms"] == 180.0
    assert summary["avg_planner_escalations"] == 1.0
    assert summary["dominant_planner_termination_reason"] == "structured_verified"
    assert summary["dominant_planner_action_signature"] == "decompose>match_decomposed"
    assert summary["planner_action_summary"] == "decompose:4,match_decomposed:4"
    assert summary["overhead_driver"] == "escalation_heavy"
    assert summary["prune_recommendation"] == "keep_current"
    assert summary["comparisons"]["rapid"]["pass_rate_delta"] == 1.0
    assert summary["comparisons"]["structured"]["prompt_calls_delta"] == -1.0
    assert summary["comparisons"]["verified"]["latency_ms_delta"] == -100.0
    assert summary["comparisons"]["rapid"]["planner_tool_dispatches_delta"] == 4.0
    assert summary["comparisons"]["verified"]["planner_escalations_delta"] == 1.0

    rendered = _format_single_agent_comparison_summary(summary)
    assert "driver=escalation_heavy" in rendered
    assert "prune=keep_current" in rendered


def test_single_agent_comparison_summary_flags_prune_candidate_when_slower_than_structured():
    class _Agg:
        def __init__(
            self,
            variant: str,
            passed_cases: int,
            total_cases: int,
            avg_leaf_coverage: float,
            avg_prompt_calls: float,
            avg_latency_ms: float,
            avg_planner_tool_dispatches: float = 0.0,
            avg_planner_tool_latency_ms: float = 0.0,
            avg_planner_escalations: float = 0.0,
            dominant_planner_termination_reason: str = "",
            dominant_planner_action_signature: str = "",
            planner_action_counts: dict[str, int] | None = None,
        ):
            self.variant = variant
            self.passed_cases = passed_cases
            self.total_cases = total_cases
            self.avg_leaf_coverage = avg_leaf_coverage
            self.avg_prompt_calls = avg_prompt_calls
            self.avg_latency_ms = avg_latency_ms
            self.avg_planner_tool_dispatches = avg_planner_tool_dispatches
            self.avg_planner_tool_latency_ms = avg_planner_tool_latency_ms
            self.avg_planner_escalations = avg_planner_escalations
            self.dominant_planner_termination_reason = dominant_planner_termination_reason
            self.dominant_planner_action_signature = dominant_planner_action_signature
            self.planner_action_counts = planner_action_counts or {}

    summary = single_agent_comparison_summary(
        [
            _Agg("rapid", 2, 4, 0.5, 1.0, 100.0),
            _Agg(
                "single_agent",
                4,
                4,
                1.0,
                5.0,
                360.0,
                6.0,
                220.0,
                1.0,
                "escalated_after_unresolved_leaves",
                "decompose>match_decomposed>retry_retrieval>escalate_orchestration",
                {
                    "decompose": 4,
                    "match_decomposed": 4,
                    "retry_retrieval": 4,
                    "escalate_orchestration": 4,
                },
            ),
            _Agg("structured", 4, 4, 1.0, 4.0, 280.0),
            _Agg("verified", 4, 4, 1.0, 6.0, 420.0),
        ]
    )

    assert summary["overhead_driver"] == "escalation_heavy"
    assert summary["prune_recommendation"] == "review_single_agent_routing"
    assert summary["dominant_planner_termination_reason"] == "escalated_after_unresolved_leaves"


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


def test_benchmark_warning_summary_prefers_comparison_failures():
    summary = benchmark_warning_summary(
        {
            "flow_comparison_failures": 2,
            "flow_comparison_unstable_groups": 1,
            "single_agent_comparison": {"overhead_driver": "tool_chatter"},
        }
    )

    assert summary == {
        "top_warning_subcheck": "comparison_failures",
        "top_warning": "flow_comparison_failures=2",
    }


def test_benchmark_warning_summary_falls_back_to_single_agent_overhead():
    summary = benchmark_warning_summary(
        {
            "flow_comparison_failures": 0,
            "flow_comparison_unstable_groups": 0,
            "single_agent_comparison": {
                "overhead_driver": "tool_chatter",
                "prune_recommendation": "keep_current",
            },
        }
    )

    assert summary == {
        "top_warning_subcheck": "single_agent_overhead",
        "top_warning": "single_agent_overhead=tool_chatter",
    }


def test_benchmark_warning_summary_prefers_single_agent_prune_over_generic_overhead():
    summary = benchmark_warning_summary(
        {
            "flow_comparison_failures": 0,
            "flow_comparison_unstable_groups": 0,
            "single_agent_comparison": {"overhead_driver": "tool_chatter"},
        }
    )

    assert summary == {
        "top_warning_subcheck": "single_agent_overhead",
        "top_warning": "single_agent_overhead=tool_chatter",
    }

    summary = benchmark_warning_summary(
        {
            "flow_comparison_failures": 0,
            "flow_comparison_unstable_groups": 0,
            "single_agent_comparison": {
                "overhead_driver": "tool_chatter",
                "prune_recommendation": "review_single_agent_routing",
            },
        }
    )

    assert summary == {
        "top_warning_subcheck": "single_agent_prune",
        "top_warning": "single_agent_prune=review_single_agent_routing",
    }


def test_format_benchmark_warning_summary_renders_compact_line():
    rendered = format_benchmark_warning_summary(
        {
            "top_warning_subcheck": "comparison_instability",
            "top_warning": "flow_comparison_unstable_groups=1",
        }
    )

    assert rendered == (
        "subcheck=comparison_instability warning=flow_comparison_unstable_groups=1"
    )


def test_format_benchmark_health_summary_renders_compact_line():
    rendered = format_benchmark_health_summary(
        warning_summary="subcheck=comparison_failures warning=flow_comparison_failures=2",
        failure_summary="subcheck=runtime_budget failure=legacy_providers_present=codex_cli",
    )

    assert rendered == (
        "warnings=subcheck=comparison_failures warning=flow_comparison_failures=2 "
        "failures=subcheck=runtime_budget failure=legacy_providers_present=codex_cli"
    )


@pytest.mark.asyncio
async def test_flow_benchmark_report_contains_all_variants(tmp_path):
    results = await run_flow_benchmark(cases=default_flow_benchmark_cases())
    aggregates = summarize_flow_benchmark(results)
    report_path = tmp_path / "flow_benchmark.json"
    save_flow_benchmark_report(report_path, results=results, aggregates=aggregates)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    variants = {row["variant"] for row in payload["aggregates"]}
    assert variants == {
        "direct_baseline",
        "rapid",
        "single_agent",
        "structured",
        "verified",
    }
