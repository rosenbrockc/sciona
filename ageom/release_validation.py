"""Repo-native release validation entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageom.benchmark_validation import run_benchmark_validation
from ageom.config import AgeomConfig, should_apply_prompt_override


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


def _runtime_complexity_summary(config: AgeomConfig) -> dict[str, Any]:
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
        if not should_apply_prompt_override(config, prompt_key, execution_mode="verified"):
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

    return {
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
        "budget": {
            "max_provider_count": 4,
            "max_provider_model_count": 5,
            "max_transport_count": 3,
            "allow_legacy_providers": False,
        },
    }


def _runtime_complexity_violations(summary: dict[str, Any]) -> list[str]:
    budget = summary.get("budget", {}) if isinstance(summary.get("budget"), dict) else {}
    violations: list[str] = []
    provider_count = int(summary.get("provider_count", 0) or 0)
    provider_model_count = int(summary.get("provider_model_count", 0) or 0)
    transport_count = int(summary.get("transport_count", 0) or 0)
    legacy_provider_count = int(summary.get("legacy_provider_count", 0) or 0)
    if provider_count > int(budget.get("max_provider_count", provider_count) or provider_count):
        violations.append(
            f"provider_count={provider_count} exceeds budget {int(budget.get('max_provider_count', provider_count) or provider_count)}"
        )
    if provider_model_count > int(
        budget.get("max_provider_model_count", provider_model_count) or provider_model_count
    ):
        violations.append(
            f"provider_model_count={provider_model_count} exceeds budget {int(budget.get('max_provider_model_count', provider_model_count) or provider_model_count)}"
        )
    if transport_count > int(budget.get("max_transport_count", transport_count) or transport_count):
        violations.append(
            f"transport_count={transport_count} exceeds budget {int(budget.get('max_transport_count', transport_count) or transport_count)}"
        )
    if not bool(budget.get("allow_legacy_providers", False)) and legacy_provider_count > 0:
        violations.append(
            f"legacy_providers_present={','.join(summary.get('legacy_providers', []) or [])}"
        )
    return violations


async def run_release_validation(output_dir: str | Path) -> dict[str, Any]:
    """Run deterministic release validation and persist a manifest."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = await run_benchmark_validation(out_dir / "benchmarks")
    runtime_complexity = _runtime_complexity_summary(AgeomConfig())
    runtime_complexity["violations"] = _runtime_complexity_violations(runtime_complexity)
    release_passed = (
        int(benchmark_summary.get("prompt_tuned_failures", 0) or 0) == 0
        and int(benchmark_summary.get("prompt_tuned_unstable_groups", 0) or 0) == 0
        and int(benchmark_summary.get("flow_mode_failures", 0) or 0) == 0
        and int(benchmark_summary.get("flow_mode_unstable_groups", 0) or 0) == 0
        and len(runtime_complexity["violations"]) == 0
    )
    manifest = {
        "status": "passed" if release_passed else "failed",
        "checks": {
            "benchmark_validation": {
                "summary_report": benchmark_summary["summary_report"],
                "prompt_report": benchmark_summary["prompt_report"],
                "flow_report": benchmark_summary["flow_report"],
                "prompt_results": benchmark_summary["prompt_results"],
                "flow_results": benchmark_summary["flow_results"],
                "prompt_tuned_failures": benchmark_summary.get("prompt_tuned_failures", 0),
                "prompt_tuned_unstable_groups": benchmark_summary.get(
                    "prompt_tuned_unstable_groups", 0
                ),
                "flow_mode_failures": benchmark_summary.get("flow_mode_failures", 0),
                "flow_mode_unstable_groups": benchmark_summary.get(
                    "flow_mode_unstable_groups", 0
                ),
            },
            "runtime_complexity": runtime_complexity,
        },
    }
    manifest_path = out_dir / "release_validation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "status": manifest["status"],
        "manifest": str(manifest_path),
        "benchmarks_dir": str(out_dir / "benchmarks"),
        "benchmark_summary": benchmark_summary,
        "runtime_complexity": runtime_complexity,
    }
