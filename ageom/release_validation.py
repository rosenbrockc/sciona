"""Repo-native release validation entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageom.benchmark_validation import run_benchmark_validation


async def run_release_validation(output_dir: str | Path) -> dict[str, Any]:
    """Run deterministic release validation and persist a manifest."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = await run_benchmark_validation(out_dir / "benchmarks")
    runtime_complexity = dict(benchmark_summary.get("runtime_complexity", {}) or {})
    release_passed = str(benchmark_summary.get("status", "failed")) == "passed"
    manifest = {
        "status": "passed" if release_passed else "failed",
        "checks": {
            "benchmark_validation": {
                "summary_report": benchmark_summary["summary_report"],
                "prompt_report": benchmark_summary["prompt_report"],
                "flow_report": benchmark_summary["flow_report"],
                "prompt_results": benchmark_summary["prompt_results"],
                "flow_results": benchmark_summary["flow_results"],
                "flow_required_variants": list(
                    benchmark_summary.get("flow_required_variants", []) or []
                ),
                "flow_comparison_variants": list(
                    benchmark_summary.get("flow_comparison_variants", []) or []
                ),
                "flow_execution_paths": dict(
                    benchmark_summary.get("flow_execution_paths", {}) or {}
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
