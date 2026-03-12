"""Deterministic benchmark validation bundle for release-style checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageom.config import (
    AgeomConfig,
    effective_round_provider_model,
    should_apply_prompt_override,
)
from ageom.flow_benchmark import (
    default_flow_benchmark_cases,
    format_flow_benchmark_summary,
    run_flow_benchmark,
    save_flow_benchmark_report,
    summarize_flow_benchmark,
)
from ageom.prompt_benchmark import (
    PromptBenchmarkProvider,
    default_prompt_benchmark_cases,
    format_prompt_benchmark_summary,
    run_prompt_benchmark,
    save_prompt_benchmark_report,
    summarize_prompt_benchmark,
)

_ROUND_DEFAULTS: dict[str, tuple[str, str]] = {
    "architect": ("architect_llm_provider", "architect_llm_model"),
    "hunter": ("hunter_llm_provider", "hunter_llm_model"),
    "synthesizer": ("synthesizer_llm_provider", "synthesizer_llm_model"),
    "ingester": ("ingester_llm_provider", "ingester_llm_model"),
}

_PROMPT_TO_ROUND: dict[str, str] = {
    "architect_strategy": "architect",
    "architect_decompose": "architect",
    "architect_critique": "architect",
    "hunter_score": "hunter",
    "hunter_reformulate": "hunter",
    "hunter_analyze_failure": "hunter",
    "synthesizer_repair": "synthesizer",
    "synthesizer_tactic": "synthesizer",
    "ingester_chunk": "ingester",
    "ingester_hoist_state": "ingester",
    "ingester_abstract": "ingester",
    "ingester_fix_type": "ingester",
    "ingester_fix_ghost": "ingester",
    "ingester_opaque_witness": "ingester",
    "ingester_fix_message_cycle": "ingester",
    "ingester_decompose": "ingester",
    "orchestrator_refine": "architect",
}

_LEGACY_PROVIDER_SET = {"claude_cli", "codex_cli", "gemini_cli"}
_REQUIRED_FLOW_BENCHMARK_VARIANTS = {"structured", "verified"}
_EXPECTED_FLOW_EXECUTION_PATHS = {
    "direct_baseline": "direct_baseline",
    "rapid": "rapid_direct",
    "single_agent": "single_agent_structured",
    "structured": "structured_single_pass",
    "verified": "verified_orchestration",
    "verified_refinement": "verified_orchestration_refinement",
    "llm_from_scratch": "llm_from_scratch",
}
_MODE_RUNTIME_BUDGETS: dict[str, dict[str, int | bool]] = {
    "rapid": {
        "max_provider_count": 1,
        "max_provider_model_count": 1,
        "max_transport_count": 1,
        "max_active_override_count": 0,
        "allow_legacy_providers": False,
    },
    "single_agent": {
        "max_provider_count": 1,
        "max_provider_model_count": 1,
        "max_transport_count": 1,
        "max_active_override_count": 0,
        "allow_legacy_providers": False,
    },
    "structured": {
        "max_provider_count": 1,
        "max_provider_model_count": 1,
        "max_transport_count": 1,
        "max_active_override_count": 0,
        "allow_legacy_providers": False,
    },
    "verified": {
        "max_provider_count": 4,
        "max_provider_model_count": 5,
        "max_transport_count": 3,
        "max_active_override_count": 5,
        "allow_legacy_providers": False,
    },
}
_MODE_OVERRIDE_POLICIES: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "rapid": {
        "required_active_overrides": (),
    },
    "single_agent": {
        "required_active_overrides": (),
    },
    "structured": {
        "required_active_overrides": (),
    },
    "verified": {
        "required_active_overrides": (
            ("architect_strategy", "codex_shim"),
            ("architect_critique", "codex_shim"),
            ("hunter_score", "codex_shim"),
            ("hunter_reformulate", "gemini_shim"),
            ("hunter_analyze_failure", "gemini_shim"),
        ),
    },
}


def _benchmark_validation_config() -> AgeomConfig:
    """Use repo defaults for validation instead of operator-local dotenv overrides."""
    return AgeomConfig(_env_file=None)


def flow_execution_path_summary(flow_aggregates: list[Any]) -> dict[str, Any]:
    """Summarize whether flow benchmark variants still map to distinct execution paths."""
    observed = {
        agg.variant: sorted(getattr(agg, "execution_paths", []) or [])
        for agg in flow_aggregates
    }
    violations: list[str] = []
    for variant, expected in _EXPECTED_FLOW_EXECUTION_PATHS.items():
        paths = observed.get(variant, [])
        if not paths:
            continue  # variant not run — skip silently
        if len(paths) != 1:
            violations.append(f"{variant}:multiple_execution_paths={','.join(paths)}")
            continue
        if paths[0] != expected:
            violations.append(
                f"{variant}:expected {expected} but observed {paths[0]}"
            )
    required_modes = ("rapid", "structured", "verified")
    observed_required = {
        variant: observed.get(variant, ["--"])[0]
        for variant in required_modes
        if observed.get(variant)
    }
    if len(set(observed_required.values())) != len(required_modes):
        violations.append(
            "mode_paths_not_distinct:"
            + ",".join(f"{variant}={path}" for variant, path in sorted(observed_required.items()))
        )
    return {
        "expected": dict(_EXPECTED_FLOW_EXECUTION_PATHS),
        "observed": observed,
        "violations": violations,
    }


def _format_flow_gate_summary(
    required_variants: list[str],
    comparison_variants: list[str],
    *,
    mode_failures: int,
    mode_unstable_groups: int,
    comparison_failures: int,
    comparison_unstable_groups: int,
) -> str:
    return (
        f"required[{','.join(required_variants) or '--'}] "
        f"{mode_failures}/{mode_unstable_groups}; "
        f"comparison[{','.join(comparison_variants) or '--'}] "
        f"{comparison_failures}/{comparison_unstable_groups}"
    )


def _format_flow_execution_path_summary(summary: dict[str, Any]) -> str:
    observed = summary.get("observed", {}) if isinstance(summary.get("observed"), dict) else {}
    parts = []
    for variant in sorted(observed):
        paths = observed.get(variant, [])
        if not isinstance(paths, list):
            continue
        parts.append(f"{variant}={'|'.join(str(path) for path in paths) or '--'}")
    if not parts:
        return "--"
    suffix = ""
    violations = summary.get("violations", [])
    if isinstance(violations, list) and violations:
        suffix = f" violations={len(violations)}"
    return ", ".join(parts) + suffix


def _format_runtime_override_policy_summary(runtime_complexity: dict[str, Any]) -> str:
    """Render a compact summary of runtime override-policy health."""
    by_mode = (
        runtime_complexity.get("by_mode", {})
        if isinstance(runtime_complexity.get("by_mode"), dict)
        else {}
    )
    parts: list[str] = []
    for mode in ("rapid", "single_agent", "structured", "verified"):
        row = by_mode.get(mode, {})
        if not isinstance(row, dict):
            continue
        policy = (
            row.get("override_policy", {})
            if isinstance(row.get("override_policy"), dict)
            else {}
        )
        parts.append(
            f"{mode}="
            f"{len(policy.get('required_active_overrides', []) or [])}/"
            f"{len(policy.get('missing_required_overrides', []) or [])}/"
            f"{len(policy.get('unexpected_active_overrides', []) or [])}"
        )
    return ", ".join(parts) or "--"


def flow_prompt_volume_summary(flow_aggregates: list[Any]) -> dict[str, Any]:
    """Summarize whether prompt volume remains monotonic across execution modes."""
    averages = {
        agg.variant: round(float(agg.avg_prompt_calls), 3)
        for agg in flow_aggregates
    }
    violations: list[str] = []
    rapid = float(averages.get("rapid", 0.0) or 0.0)
    structured = float(averages.get("structured", 0.0) or 0.0)
    verified = float(averages.get("verified", 0.0) or 0.0)
    if rapid > structured:
        violations.append(
            f"rapid_prompt_calls={rapid} exceeds structured={structured}"
        )
    if structured > verified:
        violations.append(
            f"structured_prompt_calls={structured} exceeds verified={verified}"
        )
    return {"averages": averages, "violations": violations}


def _format_flow_prompt_volume_summary(summary: dict[str, Any]) -> str:
    averages = summary.get("averages", {}) if isinstance(summary.get("averages"), dict) else {}
    parts = [f"{variant}={float(value):.1f}" for variant, value in sorted(averages.items())]
    rendered = ", ".join(parts) or "--"
    violations = summary.get("violations", []) if isinstance(summary.get("violations"), list) else []
    if violations:
        rendered += f" violations={len(violations)}"
    return rendered


def single_agent_comparison_summary(flow_aggregates: list[Any]) -> dict[str, Any]:
    """Position single_agent against rapid/structured/verified benchmark modes."""
    by_variant = {agg.variant: agg for agg in flow_aggregates}
    single_agent = by_variant.get("single_agent")
    if single_agent is None:
        return {"present": False, "comparisons": {}}

    comparisons: dict[str, dict[str, float]] = {}
    for variant in ("rapid", "structured", "verified"):
        other = by_variant.get(variant)
        if other is None:
            continue
        single_pass_rate = (
            float(single_agent.passed_cases) / max(1, int(single_agent.total_cases or 0))
        )
        other_pass_rate = float(other.passed_cases) / max(1, int(other.total_cases or 0))
        comparisons[variant] = {
            "pass_rate_delta": round(single_pass_rate - other_pass_rate, 4),
            "coverage_delta": round(
                float(single_agent.avg_leaf_coverage) - float(other.avg_leaf_coverage), 4
            ),
            "prompt_calls_delta": round(
                float(single_agent.avg_prompt_calls) - float(other.avg_prompt_calls), 4
            ),
            "latency_ms_delta": round(
                float(single_agent.avg_latency_ms) - float(other.avg_latency_ms), 4
            ),
            "planner_tool_dispatches_delta": round(
                float(getattr(single_agent, "avg_planner_tool_dispatches", 0.0) or 0.0)
                - float(getattr(other, "avg_planner_tool_dispatches", 0.0) or 0.0),
                4,
            ),
            "planner_tool_latency_ms_delta": round(
                float(getattr(single_agent, "avg_planner_tool_latency_ms", 0.0) or 0.0)
                - float(getattr(other, "avg_planner_tool_latency_ms", 0.0) or 0.0),
                4,
            ),
            "planner_escalations_delta": round(
                float(getattr(single_agent, "avg_planner_escalations", 0.0) or 0.0)
                - float(getattr(other, "avg_planner_escalations", 0.0) or 0.0),
                4,
            ),
        }

    avg_latency_ms = round(float(single_agent.avg_latency_ms), 4)
    avg_planner_tool_dispatches = round(
        float(getattr(single_agent, "avg_planner_tool_dispatches", 0.0) or 0.0), 4
    )
    avg_planner_tool_latency_ms = round(
        float(getattr(single_agent, "avg_planner_tool_latency_ms", 0.0) or 0.0), 4
    )
    avg_planner_escalations = round(
        float(getattr(single_agent, "avg_planner_escalations", 0.0) or 0.0), 4
    )
    overhead_driver = _single_agent_overhead_driver(
        avg_latency_ms=avg_latency_ms,
        avg_planner_tool_dispatches=avg_planner_tool_dispatches,
        avg_planner_tool_latency_ms=avg_planner_tool_latency_ms,
        avg_planner_escalations=avg_planner_escalations,
    )
    prune_recommendation = _single_agent_prune_recommendation(
        overhead_driver=overhead_driver,
        comparisons=comparisons,
    )

    return {
        "present": True,
        "pass_rate": round(
            float(single_agent.passed_cases) / max(1, int(single_agent.total_cases or 0)),
            4,
        ),
        "avg_leaf_coverage": round(float(single_agent.avg_leaf_coverage), 4),
        "avg_prompt_calls": round(float(single_agent.avg_prompt_calls), 4),
        "avg_latency_ms": avg_latency_ms,
        "avg_planner_tool_dispatches": avg_planner_tool_dispatches,
        "avg_planner_tool_latency_ms": avg_planner_tool_latency_ms,
        "avg_planner_escalations": avg_planner_escalations,
        "overhead_driver": overhead_driver,
        "prune_recommendation": prune_recommendation,
        "comparisons": comparisons,
    }


def _single_agent_overhead_driver(
    *,
    avg_latency_ms: float,
    avg_planner_tool_dispatches: float,
    avg_planner_tool_latency_ms: float,
    avg_planner_escalations: float,
) -> str:
    if avg_planner_escalations >= 0.5:
        return "escalation_heavy"
    if avg_planner_tool_dispatches >= 4.0:
        return "tool_chatter"
    if avg_planner_tool_latency_ms >= max(50.0, avg_latency_ms * 0.25):
        return "tool_latency"
    return "light"


def _single_agent_prune_recommendation(
    *,
    overhead_driver: str,
    comparisons: dict[str, dict[str, float]],
) -> str:
    structured = comparisons.get("structured", {})
    if not isinstance(structured, dict):
        structured = {}
    if (
        overhead_driver in {"tool_chatter", "tool_latency", "escalation_heavy"}
        and float(structured.get("pass_rate_delta", 0.0)) <= 0.0
        and float(structured.get("latency_ms_delta", 0.0)) >= 0.0
    ):
        return "review_single_agent_routing"
    return "keep_current"


def _format_single_agent_comparison_summary(summary: dict[str, Any]) -> str:
    """Render a compact single_agent positioning line for dashboards and manifests."""
    if not bool(summary.get("present", False)):
        return "--"
    comparisons = summary.get("comparisons", {})
    if not isinstance(comparisons, dict):
        comparisons = {}
    parts = [
        f"pass={float(summary.get('pass_rate', 0.0)):.2f}",
        f"coverage={float(summary.get('avg_leaf_coverage', 0.0)):.2f}",
        f"prompts={float(summary.get('avg_prompt_calls', 0.0)):.1f}",
        f"latency_ms={float(summary.get('avg_latency_ms', 0.0)):.1f}",
        f"tools={float(summary.get('avg_planner_tool_dispatches', 0.0)):.1f}",
        f"tool_latency_ms={float(summary.get('avg_planner_tool_latency_ms', 0.0)):.1f}",
        f"escalations={float(summary.get('avg_planner_escalations', 0.0)):.1f}",
        f"driver={str(summary.get('overhead_driver', '') or 'light')}",
        f"prune={str(summary.get('prune_recommendation', '') or 'keep_current')}",
    ]
    for variant in ("rapid", "structured", "verified"):
        row = comparisons.get(variant)
        if not isinstance(row, dict):
            continue
        parts.append(
            f"vs_{variant}=pass:{float(row.get('pass_rate_delta', 0.0)):+.2f}/"
            f"prompts:{float(row.get('prompt_calls_delta', 0.0)):+.1f}/"
            f"latency:{float(row.get('latency_ms_delta', 0.0)):+.1f}/"
            f"tools:{float(row.get('planner_tool_dispatches_delta', 0.0)):+.1f}/"
            f"escalations:{float(row.get('planner_escalations_delta', 0.0)):+.1f}"
        )
    return ", ".join(parts)


def benchmark_failure_summary(summary: dict[str, Any]) -> dict[str, str]:
    """Return the first benchmark failure cause and its subcheck classification."""
    runtime_budget_failure = ""
    flow_execution_paths = summary.get("flow_execution_paths", {})
    if isinstance(summary.get("runtime_complexity"), dict):
        runtime_budget_failure = str(
            ((summary.get("runtime_complexity", {}) or {}).get("violations", []) or [""])[0]
            or ""
        )
    execution_path_failure = ""
    if isinstance(flow_execution_paths, dict):
        execution_path_failure = str(
            ((flow_execution_paths.get("violations", []) or [""])[0]) or ""
        )
    flow_prompt_volume = summary.get("flow_prompt_volume", {})
    prompt_volume_failure = ""
    if isinstance(flow_prompt_volume, dict):
        prompt_volume_failure = str(
            ((flow_prompt_volume.get("violations", []) or [""])[0]) or ""
        )
    prompt_tuned_failure = (
        f"prompt_tuned_failures={int(summary.get('prompt_tuned_failures', 0) or 0)}"
        if int(summary.get("prompt_tuned_failures", 0) or 0) > 0
        else ""
    )
    flow_mode_failure = (
        f"flow_mode_failures={int(summary.get('flow_mode_failures', 0) or 0)}"
        if int(summary.get("flow_mode_failures", 0) or 0) > 0
        else ""
    )
    if runtime_budget_failure:
        return {
            "top_failed_subcheck": "runtime_budget",
            "top_failure": runtime_budget_failure,
        }
    if execution_path_failure:
        return {
            "top_failed_subcheck": "execution_path",
            "top_failure": execution_path_failure,
        }
    if prompt_volume_failure:
        return {
            "top_failed_subcheck": "prompt_volume",
            "top_failure": prompt_volume_failure,
        }
    if prompt_tuned_failure:
        return {
            "top_failed_subcheck": "prompt_tuning",
            "top_failure": prompt_tuned_failure,
        }
    if flow_mode_failure:
        return {
            "top_failed_subcheck": "flow_mode",
            "top_failure": flow_mode_failure,
        }
    return {"top_failed_subcheck": "", "top_failure": ""}


def format_benchmark_failure_summary(summary: dict[str, Any]) -> str:
    """Render a compact benchmark failure summary line."""
    top_failed_subcheck = str(summary.get("top_failed_subcheck", "") or "")
    top_failure = str(summary.get("top_failure", "") or "")
    if not top_failed_subcheck and not top_failure:
        return "none"
    parts: list[str] = []
    if top_failed_subcheck:
        parts.append(f"subcheck={top_failed_subcheck}")
    if top_failure:
        parts.append(f"failure={top_failure}")
    return " ".join(parts) or "none"


def benchmark_warning_summary(summary: dict[str, Any]) -> dict[str, str]:
    """Return the first non-blocking benchmark warning and its subcheck classification."""
    comparison_failure = (
        f"flow_comparison_failures={int(summary.get('flow_comparison_failures', 0) or 0)}"
        if int(summary.get("flow_comparison_failures", 0) or 0) > 0
        else ""
    )
    comparison_instability = (
        "flow_comparison_unstable_groups="
        f"{int(summary.get('flow_comparison_unstable_groups', 0) or 0)}"
        if int(summary.get("flow_comparison_unstable_groups", 0) or 0) > 0
        else ""
    )
    single_agent_comparison = summary.get("single_agent_comparison", {})
    if not isinstance(single_agent_comparison, dict):
        single_agent_comparison = {}
    overhead_driver = str(single_agent_comparison.get("overhead_driver", "") or "")
    prune_recommendation = str(
        single_agent_comparison.get("prune_recommendation", "") or ""
    )
    single_agent_prune = (
        f"single_agent_prune={prune_recommendation}"
        if prune_recommendation == "review_single_agent_routing"
        else ""
    )
    single_agent_overhead = (
        f"single_agent_overhead={overhead_driver}"
        if overhead_driver in {"tool_chatter", "tool_latency", "escalation_heavy"}
        else ""
    )
    if comparison_failure:
        return {
            "top_warning_subcheck": "comparison_failures",
            "top_warning": comparison_failure,
        }
    if comparison_instability:
        return {
            "top_warning_subcheck": "comparison_instability",
            "top_warning": comparison_instability,
        }
    if single_agent_prune:
        return {
            "top_warning_subcheck": "single_agent_prune",
            "top_warning": single_agent_prune,
        }
    if single_agent_overhead:
        return {
            "top_warning_subcheck": "single_agent_overhead",
            "top_warning": single_agent_overhead,
        }
    return {"top_warning_subcheck": "", "top_warning": ""}


def format_benchmark_warning_summary(summary: dict[str, Any]) -> str:
    """Render a compact benchmark warning summary line."""
    top_warning_subcheck = str(summary.get("top_warning_subcheck", "") or "")
    top_warning = str(summary.get("top_warning", "") or "")
    if not top_warning_subcheck and not top_warning:
        return "none"
    parts: list[str] = []
    if top_warning_subcheck:
        parts.append(f"subcheck={top_warning_subcheck}")
    if top_warning:
        parts.append(f"warning={top_warning}")
    return " ".join(parts) or "none"


def format_benchmark_health_summary(
    *,
    warning_summary: str,
    failure_summary: str,
) -> str:
    """Render a compact operator-facing benchmark health line."""
    normalized_warning = str(warning_summary or "").strip() or "none"
    normalized_failure = str(failure_summary or "").strip() or "none"
    return f"warnings={normalized_warning} failures={normalized_failure}"


def _transport_for_provider(provider: str) -> str:
    lowered = provider.strip().lower()
    if lowered.endswith("_shim"):
        return "persistent_shim"
    if lowered.endswith("_cli"):
        return "legacy_cli"
    if lowered == "llama_cpp":
        return "local_server"
    if lowered in {"anthropic", "codex", "openai"}:
        return "api"
    if not lowered:
        return "--"
    return "other"


def _runtime_complexity_for_mode(
    config: AgeomConfig,
    *,
    execution_mode: str,
) -> dict[str, Any]:
    """Summarize routing complexity for one execution mode."""
    providers: set[str] = set()
    provider_models: set[str] = set()
    transports: set[str] = set()
    active_overrides: list[dict[str, str]] = []
    legacy_providers: set[str] = set()

    for round_name in _ROUND_DEFAULTS:
        provider, model = effective_round_provider_model(
            config,
            round_name,
            execution_mode=execution_mode,
        )
        if not provider:
            continue
        providers.add(provider)
        provider_models.add(f"{provider}:{model or '--'}")
        transports.add(_transport_for_provider(provider))
        if provider in _LEGACY_PROVIDER_SET:
            legacy_providers.add(provider)

    for prompt_key, round_name in _PROMPT_TO_ROUND.items():
        provider = str(getattr(config, f"{prompt_key}_llm_provider", "") or "").strip()
        if not provider:
            continue
        if not should_apply_prompt_override(config, prompt_key, execution_mode=execution_mode):
            continue
        model = str(
            getattr(config, f"{prompt_key}_llm_model", "")
            or getattr(config, _ROUND_DEFAULTS[round_name][1], "")
            or config.llm_model
            or ""
        ).strip()
        providers.add(provider)
        provider_models.add(f"{provider}:{model or '--'}")
        transports.add(_transport_for_provider(provider))
        if provider in _LEGACY_PROVIDER_SET:
            legacy_providers.add(provider)
        active_overrides.append(
            {"prompt_key": prompt_key, "provider": provider, "model": model}
        )

    required_pairs = set(
        _MODE_OVERRIDE_POLICIES.get(execution_mode, {}).get(
            "required_active_overrides", ()
        )
    )
    active_pairs = {
        (str(row["prompt_key"]), str(row["provider"]))
        for row in active_overrides
    }
    missing_required = sorted(required_pairs - active_pairs)
    unexpected_active = sorted(active_pairs - required_pairs) if required_pairs else []

    summary = {
        "mode": execution_mode,
        "provider_count": len(providers),
        "provider_model_count": len(provider_models),
        "transport_count": len(transports),
        "providers": sorted(providers),
        "provider_models": sorted(provider_models),
        "transports": sorted(transports),
        "active_override_count": len(active_overrides),
        "active_overrides": sorted(
            active_overrides,
            key=lambda row: (row["provider"], row["model"], row["prompt_key"]),
        ),
        "legacy_provider_count": len(legacy_providers),
        "legacy_providers": sorted(legacy_providers),
        "budget": dict(_MODE_RUNTIME_BUDGETS[execution_mode]),
        "override_policy": {
            "required_active_overrides": [
                {"prompt_key": prompt_key, "provider": provider}
                for prompt_key, provider in sorted(required_pairs)
            ],
            "missing_required_overrides": [
                {"prompt_key": prompt_key, "provider": provider}
                for prompt_key, provider in missing_required
            ],
            "unexpected_active_overrides": [
                {"prompt_key": prompt_key, "provider": provider}
                for prompt_key, provider in unexpected_active
            ],
        },
    }
    summary["violations"] = runtime_complexity_violations(summary)
    return summary


def runtime_complexity_violations(summary: dict[str, Any]) -> list[str]:
    """Compute budget violations from a runtime-complexity summary."""
    budget = summary.get("budget", {}) if isinstance(summary.get("budget"), dict) else {}
    violations: list[str] = []
    provider_count = int(summary.get("provider_count", 0) or 0)
    provider_model_count = int(summary.get("provider_model_count", 0) or 0)
    transport_count = int(summary.get("transport_count", 0) or 0)
    active_override_count = int(summary.get("active_override_count", 0) or 0)
    legacy_provider_count = int(summary.get("legacy_provider_count", 0) or 0)
    max_provider_count = int(budget.get("max_provider_count", provider_count) or provider_count)
    max_provider_model_count = int(
        budget.get("max_provider_model_count", provider_model_count) or provider_model_count
    )
    max_transport_count = int(
        budget.get("max_transport_count", transport_count) or transport_count
    )
    max_active_override_count = int(
        budget.get("max_active_override_count", active_override_count) or active_override_count
    )
    if provider_count > max_provider_count:
        violations.append(
            f"provider_count={provider_count} exceeds budget {max_provider_count}"
        )
    if provider_model_count > max_provider_model_count:
        violations.append(
            f"provider_model_count={provider_model_count} exceeds budget {max_provider_model_count}"
        )
    if transport_count > max_transport_count:
        violations.append(
            f"transport_count={transport_count} exceeds budget {max_transport_count}"
        )
    if active_override_count > max_active_override_count:
        violations.append(
            f"active_override_count={active_override_count} exceeds budget {max_active_override_count}"
        )
    if not bool(budget.get("allow_legacy_providers", False)) and legacy_provider_count > 0:
        violations.append(
            f"legacy_providers_present={','.join(summary.get('legacy_providers', []) or [])}"
        )
    override_policy = (
        summary.get("override_policy", {})
        if isinstance(summary.get("override_policy"), dict)
        else {}
    )
    missing_required = (
        override_policy.get("missing_required_overrides", [])
        if isinstance(override_policy.get("missing_required_overrides", []), list)
        else []
    )
    unexpected_active = (
        override_policy.get("unexpected_active_overrides", [])
        if isinstance(override_policy.get("unexpected_active_overrides", []), list)
        else []
    )
    if missing_required:
        violations.extend(
            f"missing_required_override:{row.get('prompt_key','')}={row.get('provider','')}"
            for row in missing_required
            if isinstance(row, dict)
        )
    if unexpected_active:
        violations.extend(
            f"unexpected_active_override:{row.get('prompt_key','')}={row.get('provider','')}"
            for row in unexpected_active
            if isinstance(row, dict)
        )
    return violations


def runtime_complexity_summary(config: AgeomConfig) -> dict[str, Any]:
    """Summarize routing complexity across execution modes."""
    by_mode = {
        mode: _runtime_complexity_for_mode(config, execution_mode=mode)
        for mode in ("rapid", "single_agent", "structured", "verified")
    }
    monotonic_violations: list[str] = []
    metric_names = (
        "provider_count",
        "provider_model_count",
        "transport_count",
        "active_override_count",
    )
    for metric in metric_names:
        rapid_value = int(by_mode["rapid"].get(metric, 0) or 0)
        single_agent_value = int(by_mode["single_agent"].get(metric, 0) or 0)
        structured_value = int(by_mode["structured"].get(metric, 0) or 0)
        verified_value = int(by_mode["verified"].get(metric, 0) or 0)
        if rapid_value > single_agent_value:
            monotonic_violations.append(
                f"{metric}: rapid={rapid_value} exceeds single_agent={single_agent_value}"
            )
        if single_agent_value > structured_value:
            monotonic_violations.append(
                f"{metric}: single_agent={single_agent_value} exceeds structured={structured_value}"
            )
        if structured_value > verified_value:
            monotonic_violations.append(
                f"{metric}: structured={structured_value} exceeds verified={verified_value}"
            )

    violations: list[str] = []
    for mode, row in by_mode.items():
        violations.extend(f"{mode}:{item}" for item in row.get("violations", []))
    violations.extend(f"monotonic:{item}" for item in monotonic_violations)

    verified = dict(by_mode["verified"])
    verified["by_mode"] = by_mode
    verified["monotonic_violations"] = monotonic_violations
    verified["violations"] = violations
    return verified


class FixturePromptBenchmarkLLM:
    """Deterministic provider for prompt benchmark release validation."""

    def __init__(self, model: str = "fixture-good") -> None:
        self._telemetry_model = model

    async def complete(self, system: str, user: str) -> str:
        lower = system.lower()
        if "json array of integer indices" in lower:
            return "[0, 1]"
        if "json array of strings" in lower:
            user_lower = user.lower()
            if "ecg" in user_lower:
                return '["ecg bandpass filter", "stable ecg filter", "bandpass cardiac signal"]'
            if "shortest path" in user_lower:
                return '["dijkstra shortest path", "weighted graph distances", "shortest path distance map"]'
            if "spd" in user_lower:
                return '["cholesky solve spd", "solve symmetric positive definite", "triangular solve cholesky"]'
            return '["longest common subsequence", "dynamic programming lcs", "string subsequence recurrence"]'
        if "return exactly three lines" in lower:
            user_lower = user.lower()
            if "filter" in user_lower or "ecg" in user_lower:
                return "CAUSE: wrong output artifact\nTARGET: filter primitive returning signal\nNEXT: search ecg signal filter"
            if "shortest path" in user_lower or "distance" in user_lower:
                return "CAUSE: ordering instead of distances\nTARGET: path routine returning distance map\nNEXT: search dijkstra shortest path"
            if "spd" in user_lower or "linear system" in user_lower:
                return "CAUSE: decomposition without solve step\nTARGET: solve routine returning vector\nNEXT: search cholesky solve"
            return "CAUSE: pattern matcher not subsequence\nTARGET: dynamic subsequence routine\nNEXT: search lcs dynamic programming"
        # Architect critique prompts (check first — most specific system prompt)
        if "algorithm critic" in lower:
            return '{"approved": true, "reason": "Decomposition is correct and complete"}'
        # Architect decompose prompts (check before strategy — "decompose the given" is unique)
        if "decompose the given algorithmic node" in lower:
            user_lower = user.lower()
            if "filter" in user_lower or "ecg" in user_lower:
                return '{"sub_nodes": [{"name": "Design Filter Coefficients", "description": "Compute stable bandpass coefficients"}, {"name": "Apply Filter", "description": "Apply filter to signal"}]}'
            if "shortest path" in user_lower or "graph" in user_lower:
                return '{"sub_nodes": [{"name": "Initialize Distances", "description": "Set source to 0, others to infinity"}, {"name": "Relax Edges", "description": "Iteratively improve distance estimates"}]}'
            if "linear" in user_lower or "spd" in user_lower:
                return '{"sub_nodes": [{"name": "Cholesky Factor", "description": "Compute lower triangular factor"}, {"name": "Triangular Solve", "description": "Solve using forward/back substitution"}]}'
            if "subsequence" in user_lower or "lcs" in user_lower:
                return '{"sub_nodes": [{"name": "Build DP Table", "description": "Fill table of prefix LCS lengths"}, {"name": "Backtrack Solution", "description": "Reconstruct LCS from table"}]}'
        # Architect strategy prompts
        if "select the best algorithmic paradigm" in lower:
            user_lower = user.lower()
            if "filter" in user_lower or "ecg" in user_lower:
                return '{"paradigm": "signal_filter", "rationale": "ECG filtering requires stable bandpass design"}'
            if "shortest path" in user_lower or "weighted graph" in user_lower:
                return '{"paradigm": "graph_optimization", "rationale": "Single-source shortest path is a graph optimization problem"}'
            if "linear system" in user_lower or "symmetric positive definite" in user_lower:
                return '{"paradigm": "algebra", "rationale": "SPD solve is a linear algebra operation"}'
            if "longest common subsequence" in user_lower:
                return '{"paradigm": "dynamic_programming", "rationale": "LCS uses optimal substructure via DP"}'
        # Generic critique/evaluate fallback
        if "critic" in lower or "evaluate" in lower or "approved" in lower:
            return '{"approved": true, "reason": "Decomposition is correct and complete"}'
        return ""

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


async def run_benchmark_validation(output_dir: str | Path) -> dict[str, Any]:
    """Run deterministic prompt/flow benchmark bundles and persist reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_cases = default_prompt_benchmark_cases()
    prompt_results = await run_prompt_benchmark(
        providers=[
            PromptBenchmarkProvider(
                name="fixture_good",
                client=FixturePromptBenchmarkLLM(),
            )
        ],
        cases=prompt_cases,
        compare_direct_baseline=True,
    )
    prompt_aggregates = summarize_prompt_benchmark(prompt_results)
    prompt_report = out_dir / "prompt_benchmark.json"
    save_prompt_benchmark_report(
        prompt_report,
        results=prompt_results,
        aggregates=prompt_aggregates,
    )

    flow_cases = default_flow_benchmark_cases()
    flow_results = await run_flow_benchmark(cases=flow_cases)
    flow_aggregates = summarize_flow_benchmark(flow_results)
    flow_report = out_dir / "flow_benchmark.json"
    save_flow_benchmark_report(
        flow_report,
        results=flow_results,
        aggregates=flow_aggregates,
    )

    prompt_tuned_failures = sum(
        agg.failed_cases for agg in prompt_aggregates if agg.variant != "direct_baseline"
    )
    prompt_tuned_unstable_groups = sum(
        max(0, agg.repeat_groups - agg.stable_groups)
        for agg in prompt_aggregates
        if agg.variant != "direct_baseline"
    )
    flow_mode_failures = sum(
        agg.failed_cases
        for agg in flow_aggregates
        if agg.variant in _REQUIRED_FLOW_BENCHMARK_VARIANTS
    )
    flow_mode_unstable_groups = sum(
        max(0, agg.repeat_groups - agg.stable_groups)
        for agg in flow_aggregates
        if agg.variant in _REQUIRED_FLOW_BENCHMARK_VARIANTS
    )
    comparison_flow_variants = {
        agg.variant
        for agg in flow_aggregates
        if agg.variant not in _REQUIRED_FLOW_BENCHMARK_VARIANTS
    }
    flow_comparison_failures = sum(
        agg.failed_cases
        for agg in flow_aggregates
        if agg.variant in comparison_flow_variants
    )
    flow_comparison_unstable_groups = sum(
        max(0, agg.repeat_groups - agg.stable_groups)
        for agg in flow_aggregates
        if agg.variant in comparison_flow_variants
    )
    flow_execution_paths = flow_execution_path_summary(flow_aggregates)
    flow_prompt_volume = flow_prompt_volume_summary(flow_aggregates)
    single_agent_comparison = single_agent_comparison_summary(flow_aggregates)
    runtime_complexity = runtime_complexity_summary(_benchmark_validation_config())
    benchmark_failures = benchmark_failure_summary(
        {
            "runtime_complexity": runtime_complexity,
            "flow_execution_paths": flow_execution_paths,
            "flow_prompt_volume": flow_prompt_volume,
            "prompt_tuned_failures": prompt_tuned_failures,
            "flow_mode_failures": flow_mode_failures,
        }
    )
    benchmark_warnings = benchmark_warning_summary(
        {
            "flow_comparison_failures": flow_comparison_failures,
            "flow_comparison_unstable_groups": flow_comparison_unstable_groups,
            "single_agent_comparison": single_agent_comparison,
        }
    )
    flow_agg_map = {agg.variant: agg for agg in flow_aggregates}
    _rapid_cov = getattr(flow_agg_map.get("rapid"), "avg_leaf_coverage", 0.0)
    _struct_cov = getattr(flow_agg_map.get("structured"), "avg_leaf_coverage", 0.0)
    _verif_cov = getattr(flow_agg_map.get("verified"), "avg_leaf_coverage", 0.0)
    coverage_monotonic = _struct_cov >= _rapid_cov and _verif_cov >= _struct_cov
    benchmark_passed = (
        prompt_tuned_failures == 0
        and prompt_tuned_unstable_groups == 0
        and flow_mode_failures == 0
        and flow_mode_unstable_groups == 0
        and len(flow_execution_paths["violations"]) == 0
        and len(flow_prompt_volume["violations"]) == 0
        and len(runtime_complexity["violations"]) == 0
        and coverage_monotonic
    )

    required_variants = sorted(_REQUIRED_FLOW_BENCHMARK_VARIANTS)
    comparison_variants = sorted(comparison_flow_variants)
    summary = {
        "status": "passed" if benchmark_passed else "failed",
        "prompt_cases": len(prompt_cases),
        "prompt_results": len(prompt_results),
        "prompt_report": str(prompt_report),
        "prompt_summary": format_prompt_benchmark_summary(prompt_aggregates),
        "prompt_stability_summary": ", ".join(
            f"{agg.provider}/{agg.variant} {agg.stable_groups}/{agg.repeat_groups}"
            for agg in prompt_aggregates
        ),
        "flow_cases": len(flow_cases),
        "flow_results": len(flow_results),
        "flow_report": str(flow_report),
        "flow_summary": format_flow_benchmark_summary(flow_aggregates),
        "flow_stability_summary": ", ".join(
            f"{agg.variant} {agg.stable_groups}/{agg.repeat_groups}"
            for agg in flow_aggregates
        ),
        "flow_required_variants": required_variants,
        "flow_comparison_variants": comparison_variants,
        "flow_execution_paths": flow_execution_paths,
        "flow_prompt_volume": flow_prompt_volume,
        "flow_gate_summary": _format_flow_gate_summary(
            required_variants,
            comparison_variants,
            mode_failures=flow_mode_failures,
            mode_unstable_groups=flow_mode_unstable_groups,
            comparison_failures=flow_comparison_failures,
            comparison_unstable_groups=flow_comparison_unstable_groups,
        ),
        "flow_execution_path_summary": _format_flow_execution_path_summary(
            flow_execution_paths
        ),
        "flow_prompt_volume_summary": _format_flow_prompt_volume_summary(
            flow_prompt_volume
        ),
        "single_agent_comparison": single_agent_comparison,
        "single_agent_comparison_summary": _format_single_agent_comparison_summary(
            single_agent_comparison
        ),
        "flow_avg_prompt_calls": {
            agg.variant: round(float(agg.avg_prompt_calls), 3) for agg in flow_aggregates
        },
        "flow_avg_planner_tool_dispatches": {
            agg.variant: round(
                float(getattr(agg, "avg_planner_tool_dispatches", 0.0) or 0.0), 3
            )
            for agg in flow_aggregates
        },
        "flow_avg_planner_tool_latency_ms": {
            agg.variant: round(
                float(getattr(agg, "avg_planner_tool_latency_ms", 0.0) or 0.0), 3
            )
            for agg in flow_aggregates
        },
        "flow_avg_planner_escalations": {
            agg.variant: round(
                float(getattr(agg, "avg_planner_escalations", 0.0) or 0.0), 3
            )
            for agg in flow_aggregates
        },
        "prompt_avg_latency_ms": {
            f"{agg.provider}:{agg.variant}": round(float(agg.avg_latency_ms), 3)
            for agg in prompt_aggregates
        },
        "flow_avg_latency_ms": {
            agg.variant: round(float(agg.avg_latency_ms), 3) for agg in flow_aggregates
        },
        "prompt_tuned_failures": prompt_tuned_failures,
        "prompt_tuned_unstable_groups": prompt_tuned_unstable_groups,
        "flow_mode_failures": flow_mode_failures,
        "flow_mode_unstable_groups": flow_mode_unstable_groups,
        "flow_comparison_failures": flow_comparison_failures,
        "flow_comparison_unstable_groups": flow_comparison_unstable_groups,
        "coverage_monotonic": coverage_monotonic,
        "coverage_by_variant": {
            "rapid": round(_rapid_cov, 4),
            "structured": round(_struct_cov, 4),
            "verified": round(_verif_cov, 4),
        },
        "runtime_complexity": runtime_complexity,
        "runtime_override_policy_summary": _format_runtime_override_policy_summary(
            runtime_complexity
        ),
        "top_failed_subcheck": benchmark_failures["top_failed_subcheck"],
        "top_failure": benchmark_failures["top_failure"],
        "failure_summary": format_benchmark_failure_summary(benchmark_failures),
        "top_warning_subcheck": benchmark_warnings["top_warning_subcheck"],
        "top_warning": benchmark_warnings["top_warning"],
        "warning_summary": format_benchmark_warning_summary(benchmark_warnings),
        "health_summary": format_benchmark_health_summary(
            warning_summary=format_benchmark_warning_summary(benchmark_warnings),
            failure_summary=format_benchmark_failure_summary(benchmark_failures),
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_report"] = str(summary_path)
    return summary
