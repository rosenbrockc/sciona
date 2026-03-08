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
    manifest = {
        "status": "passed",
        "checks": {
            "benchmark_validation": {
                "summary_report": benchmark_summary["summary_report"],
                "prompt_report": benchmark_summary["prompt_report"],
                "flow_report": benchmark_summary["flow_report"],
                "prompt_results": benchmark_summary["prompt_results"],
                "flow_results": benchmark_summary["flow_results"],
            }
        },
    }
    manifest_path = out_dir / "release_validation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "benchmarks_dir": str(out_dir / "benchmarks"),
        "benchmark_summary": benchmark_summary,
    }
