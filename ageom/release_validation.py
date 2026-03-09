"""Repo-native release validation entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageom.benchmark_validation import (
    benchmark_failure_summary,
    benchmark_warning_summary,
    format_benchmark_failure_summary,
    format_benchmark_warning_summary,
    run_benchmark_validation,
)
from ageom.catalog_validation import run_catalog_validation


def _format_release_warning_summary(
    *,
    runtime_complexity: dict[str, Any],
    catalog_summary: dict[str, Any],
) -> dict[str, Any]:
    runtime_warnings = list(runtime_complexity.get("violations", []) or [])
    catalog_warnings = list(catalog_summary.get("warnings", []) or [])
    runtime_warning_count = len(runtime_warnings)
    catalog_warning_count = len(catalog_warnings)
    top_runtime_warning = str(runtime_warnings[0] if runtime_warnings else "")
    top_catalog_warning = str(catalog_warnings[0] if catalog_warnings else "")
    return {
        "runtime_warning_count": runtime_warning_count,
        "catalog_warning_count": catalog_warning_count,
        "top_runtime_warning": top_runtime_warning,
        "top_catalog_warning": top_catalog_warning,
        "warning_summary": (
            f"runtime={runtime_warning_count}"
            f"{f' top={top_runtime_warning}' if top_runtime_warning else ''} "
            f"catalog={catalog_warning_count}"
            f"{f' top={top_catalog_warning}' if top_catalog_warning else ''}"
        ),
    }


def _format_release_failure_summary(
    *,
    benchmark_summary: dict[str, Any],
    runtime_complexity: dict[str, Any],
    catalog_summary: dict[str, Any],
) -> dict[str, Any]:
    def _first_text(values: Any) -> str:
        if not isinstance(values, list) or not values:
            return ""
        return str(values[0] or "")

    benchmark_failure = ""
    benchmark_subcheck = ""
    if str(benchmark_summary.get("status", "failed")) != "passed":
        benchmark_subcheck = str(benchmark_summary.get("top_failed_subcheck", "") or "")
        benchmark_failure = str(benchmark_summary.get("top_failure", "") or "")
        if not benchmark_subcheck or not benchmark_failure:
            runtime_budget_failure = _first_text(
                (benchmark_summary.get("runtime_complexity", {}) or {}).get("violations", [])
            )
            execution_path_failure = _first_text(
                (benchmark_summary.get("flow_execution_paths", {}) or {}).get("violations", [])
            )
            prompt_volume_failure = _first_text(
                (benchmark_summary.get("flow_prompt_volume", {}) or {}).get("violations", [])
            )
            prompt_tuned_failure = (
                f"prompt_tuned_failures={int(benchmark_summary.get('prompt_tuned_failures', 0) or 0)}"
                if int(benchmark_summary.get("prompt_tuned_failures", 0) or 0) > 0
                else ""
            )
            flow_mode_failure = (
                f"flow_mode_failures={int(benchmark_summary.get('flow_mode_failures', 0) or 0)}"
                if int(benchmark_summary.get("flow_mode_failures", 0) or 0) > 0
                else ""
            )
            benchmark_failure = str(
                runtime_budget_failure
                or execution_path_failure
                or prompt_volume_failure
                or prompt_tuned_failure
                or flow_mode_failure
                or ""
            )
            if runtime_budget_failure:
                benchmark_subcheck = "runtime_budget"
            elif execution_path_failure:
                benchmark_subcheck = "execution_path"
            elif prompt_volume_failure:
                benchmark_subcheck = "prompt_volume"
            elif prompt_tuned_failure:
                benchmark_subcheck = "prompt_tuning"
            elif flow_mode_failure:
                benchmark_subcheck = "flow_mode"
    runtime_failure = _first_text(runtime_complexity.get("violations", []))
    catalog_failure = _first_text(catalog_summary.get("violations", []))
    top_failed_check = "none"
    if runtime_failure:
        top_failed_check = "runtime_complexity"
    elif catalog_failure:
        top_failed_check = "catalog_validation"
    elif benchmark_failure:
        top_failed_check = "benchmark_validation"
    parts: list[str] = []
    if top_failed_check != "none":
        parts.append(f"check={top_failed_check}")
    if benchmark_subcheck:
        parts.append(f"benchmark_check={benchmark_subcheck}")
    if benchmark_failure:
        parts.append(f"benchmark={benchmark_failure}")
    if runtime_failure:
        parts.append(f"runtime={runtime_failure}")
    if catalog_failure:
        parts.append(f"catalog={catalog_failure}")
    return {
        "top_failed_check": top_failed_check,
        "top_benchmark_subcheck": benchmark_subcheck,
        "top_benchmark_failure": benchmark_failure,
        "top_runtime_failure": runtime_failure,
        "top_catalog_failure": catalog_failure,
        "failure_summary": " ".join(parts) or "none",
    }


async def run_release_validation(output_dir: str | Path) -> dict[str, Any]:
    """Run deterministic release validation and persist a manifest."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = await run_benchmark_validation(out_dir / "benchmarks")
    catalog_summary = await run_catalog_validation(out_dir / "catalog")
    runtime_complexity = dict(benchmark_summary.get("runtime_complexity", {}) or {})
    benchmark_failure_details = benchmark_failure_summary(benchmark_summary)
    benchmark_failure_summary_text = str(
        benchmark_summary.get("failure_summary", "") or ""
    ) or format_benchmark_failure_summary(benchmark_failure_details)
    benchmark_top_failed_subcheck = str(
        benchmark_summary.get("top_failed_subcheck", "") or ""
    ) or benchmark_failure_details["top_failed_subcheck"]
    benchmark_top_failure = str(
        benchmark_summary.get("top_failure", "") or ""
    ) or benchmark_failure_details["top_failure"]
    benchmark_warning_details = benchmark_warning_summary(benchmark_summary)
    benchmark_warning_summary_text = str(
        benchmark_summary.get("warning_summary", "") or ""
    ) or format_benchmark_warning_summary(benchmark_warning_details)
    benchmark_top_warning_subcheck = str(
        benchmark_summary.get("top_warning_subcheck", "") or ""
    ) or benchmark_warning_details["top_warning_subcheck"]
    benchmark_top_warning = str(
        benchmark_summary.get("top_warning", "") or ""
    ) or benchmark_warning_details["top_warning"]
    warning_summary = _format_release_warning_summary(
        runtime_complexity=runtime_complexity,
        catalog_summary=catalog_summary,
    )
    failure_summary = _format_release_failure_summary(
        benchmark_summary=benchmark_summary,
        runtime_complexity=runtime_complexity,
        catalog_summary=catalog_summary,
    )
    release_passed = (
        str(benchmark_summary.get("status", "failed")) == "passed"
        and str(catalog_summary.get("status", "failed")) == "passed"
    )
    manifest = {
        "status": "passed" if release_passed else "failed",
        "warnings": warning_summary,
        "failures": failure_summary,
        "checks": {
            "benchmark_validation": {
                "summary_report": benchmark_summary["summary_report"],
                "prompt_report": benchmark_summary["prompt_report"],
                "flow_report": benchmark_summary["flow_report"],
                "prompt_results": benchmark_summary["prompt_results"],
                "flow_results": benchmark_summary["flow_results"],
                "flow_gate_summary": benchmark_summary.get("flow_gate_summary", ""),
                "flow_execution_path_summary": benchmark_summary.get(
                    "flow_execution_path_summary", ""
                ),
                "runtime_override_policy_summary": benchmark_summary.get(
                    "runtime_override_policy_summary", ""
                ),
                "warning_summary": benchmark_warning_summary_text,
                "top_warning_subcheck": benchmark_top_warning_subcheck,
                "top_warning": benchmark_top_warning,
                "failure_summary": benchmark_failure_summary_text,
                "top_failed_subcheck": benchmark_top_failed_subcheck,
                "top_failure": benchmark_top_failure,
                "flow_required_variants": list(
                    benchmark_summary.get("flow_required_variants", []) or []
                ),
                "flow_comparison_variants": list(
                    benchmark_summary.get("flow_comparison_variants", []) or []
                ),
                "flow_execution_paths": dict(
                    benchmark_summary.get("flow_execution_paths", {}) or {}
                ),
                "flow_prompt_volume": dict(
                    benchmark_summary.get("flow_prompt_volume", {}) or {}
                ),
                "flow_prompt_volume_summary": benchmark_summary.get(
                    "flow_prompt_volume_summary", ""
                ),
                "prompt_tuned_failures": benchmark_summary.get("prompt_tuned_failures", 0),
                "prompt_tuned_unstable_groups": benchmark_summary.get(
                    "prompt_tuned_unstable_groups", 0
                ),
                "flow_mode_failures": benchmark_summary.get("flow_mode_failures", 0),
                "flow_mode_unstable_groups": benchmark_summary.get(
                    "flow_mode_unstable_groups", 0
                ),
                "flow_comparison_failures": benchmark_summary.get(
                    "flow_comparison_failures", 0
                ),
                "flow_comparison_unstable_groups": benchmark_summary.get(
                    "flow_comparison_unstable_groups", 0
                ),
            },
            "catalog_validation": catalog_summary,
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
        "catalog_validation": catalog_summary,
        "runtime_complexity": runtime_complexity,
        **warning_summary,
        **failure_summary,
    }
