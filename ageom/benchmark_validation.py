"""Deterministic benchmark validation bundle for release-style checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

    summary = {
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
        "prompt_tuned_failures": prompt_tuned_failures,
        "prompt_tuned_unstable_groups": prompt_tuned_unstable_groups,
        "flow_mode_failures": flow_mode_failures,
        "flow_mode_unstable_groups": flow_mode_unstable_groups,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_report"] = str(summary_path)
    return summary
