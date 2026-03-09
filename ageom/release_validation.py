"""Repo-native release validation entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageom.benchmark_validation import run_benchmark_validation
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


async def run_release_validation(output_dir: str | Path) -> dict[str, Any]:
    """Run deterministic release validation and persist a manifest."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = await run_benchmark_validation(out_dir / "benchmarks")
    catalog_summary = await run_catalog_validation(out_dir / "catalog")
    runtime_complexity = dict(benchmark_summary.get("runtime_complexity", {}) or {})
    warning_summary = _format_release_warning_summary(
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
    }
