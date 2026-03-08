from __future__ import annotations

import json

import pytest

from ageom.benchmark_validation import run_benchmark_validation
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
    assert "prompt_avg_latency_ms" in payload
    assert "flow_avg_latency_ms" in payload
    assert "runtime_complexity" in payload
    assert "prompt_tuned_failures" in payload
    assert "flow_mode_failures" in payload
    assert set(payload["flow_avg_prompt_calls"]) == {
        "direct_baseline",
        "rapid",
        "structured",
        "verified",
    }
    assert set(payload["flow_avg_latency_ms"]) == {
        "direct_baseline",
        "rapid",
        "structured",
        "verified",
    }


@pytest.mark.asyncio
async def test_flow_benchmark_report_contains_all_variants(tmp_path):
    results = await run_flow_benchmark(cases=default_flow_benchmark_cases())
    aggregates = summarize_flow_benchmark(results)
    report_path = tmp_path / "flow_benchmark.json"
    save_flow_benchmark_report(report_path, results=results, aggregates=aggregates)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    variants = {row["variant"] for row in payload["aggregates"]}
    assert variants == {"direct_baseline", "rapid", "structured", "verified"}
