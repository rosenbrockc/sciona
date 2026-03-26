"""Benchmark aggregation and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from sciona.benchmarks.core import FlowBenchmarkAggregate, FlowBenchmarkResult


def summarize_flow_benchmark(
    results: Sequence[FlowBenchmarkResult],
) -> list[FlowBenchmarkAggregate]:
    aggregates: dict[str, FlowBenchmarkAggregate] = {}
    for result in results:
        aggregate = aggregates.setdefault(
            result.variant, FlowBenchmarkAggregate(variant=result.variant)
        )
        aggregate.record(result)
    for aggregate in aggregates.values():
        aggregate.finalize()
    return sorted(
        aggregates.values(),
        key=lambda item: (
            -item.passed_cases,
            -item.stability_rate,
            item.avg_latency_ms,
            item.variant,
        ),
    )


def format_flow_benchmark_summary(
    aggregates: Sequence[FlowBenchmarkAggregate],
) -> str:
    lines = [
        "variant | paths | pass/total | stable | avg ms | avg prompts",
        "--- | --- | --- | ---: | ---: | ---:",
    ]
    for aggregate in aggregates:
        paths = ",".join(aggregate.execution_paths) or "--"
        lines.append(
            f"{aggregate.variant} | {paths} | {aggregate.passed_cases}/{aggregate.total_cases} | "
            f"{aggregate.stable_groups}/{aggregate.repeat_groups} | {aggregate.avg_latency_ms:.1f} | "
            f"{aggregate.avg_prompt_calls:.1f}"
        )
    return "\n".join(lines)


def save_flow_benchmark_report(
    path: str | Path,
    *,
    results: Sequence[FlowBenchmarkResult],
    aggregates: Sequence[FlowBenchmarkAggregate],
) -> None:
    payload = {
        "results": [result.to_dict() for result in results],
        "aggregates": [aggregate.to_dict() for aggregate in aggregates],
    }
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

