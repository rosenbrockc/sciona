"""Deterministic benchmark validation bundle for release-style checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageom.config import AgeomConfig, should_apply_prompt_override
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
_MODE_RUNTIME_BUDGETS: dict[str, dict[str, int | bool]] = {
    "rapid": {
        "max_provider_count": 2,
        "max_provider_model_count": 2,
        "max_transport_count": 2,
        "max_active_override_count": 0,
        "allow_legacy_providers": False,
    },
    "structured": {
        "max_provider_count": 2,
        "max_provider_model_count": 2,
        "max_transport_count": 2,
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


def _benchmark_validation_config() -> AgeomConfig:
    """Use repo defaults for validation instead of operator-local dotenv overrides."""
    return AgeomConfig(_env_file=None)


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

    for round_name, (provider_attr, model_attr) in _ROUND_DEFAULTS.items():
        provider = str(getattr(config, provider_attr, "") or config.llm_provider or "").strip()
        model = str(getattr(config, model_attr, "") or config.llm_model or "").strip()
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
    return violations


def runtime_complexity_summary(config: AgeomConfig) -> dict[str, Any]:
    """Summarize routing complexity across execution modes."""
    by_mode = {
        mode: _runtime_complexity_for_mode(config, execution_mode=mode)
        for mode in ("rapid", "structured", "verified")
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
        structured_value = int(by_mode["structured"].get(metric, 0) or 0)
        verified_value = int(by_mode["verified"].get(metric, 0) or 0)
        if rapid_value > structured_value:
            monotonic_violations.append(
                f"{metric}: rapid={rapid_value} exceeds structured={structured_value}"
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
        agg.failed_cases for agg in flow_aggregates if agg.variant != "direct_baseline"
    )
    flow_mode_unstable_groups = sum(
        max(0, agg.repeat_groups - agg.stable_groups)
        for agg in flow_aggregates
        if agg.variant != "direct_baseline"
    )
    runtime_complexity = runtime_complexity_summary(_benchmark_validation_config())
    benchmark_passed = (
        prompt_tuned_failures == 0
        and prompt_tuned_unstable_groups == 0
        and flow_mode_failures == 0
        and flow_mode_unstable_groups == 0
        and len(runtime_complexity["violations"]) == 0
    )

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
        "flow_avg_prompt_calls": {
            agg.variant: round(float(agg.avg_prompt_calls), 3) for agg in flow_aggregates
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
        "runtime_complexity": runtime_complexity,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_report"] = str(summary_path)
    return summary
