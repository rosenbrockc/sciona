from __future__ import annotations

import pytest

from ageom.flow_benchmark import (
    default_flow_benchmark_cases,
    format_flow_benchmark_summary,
    run_flow_benchmark,
    summarize_flow_benchmark,
)


def test_default_flow_benchmark_cases_cover_multiple_domains():
    cases = default_flow_benchmark_cases()

    assert len(cases) == 3
    assert {case.domain for case in cases} == {"sorting", "graph", "dsp"}


@pytest.mark.asyncio
async def test_flow_benchmark_summary_orders_variants_by_success():
    cases = default_flow_benchmark_cases()

    results = await run_flow_benchmark(cases=cases)
    aggregates = summarize_flow_benchmark(results)
    aggregate_map = {aggregate.variant: aggregate for aggregate in aggregates}

    assert aggregates[0].variant in {"structured", "verified"}
    assert aggregate_map["direct_baseline"].failed_cases == len(cases)
    assert aggregate_map["rapid"].failed_cases == len(cases)
    assert aggregate_map["structured"].passed_cases == len(cases)
    assert aggregate_map["verified"].passed_cases == len(cases)
    assert aggregate_map["rapid"].execution_paths == ["rapid_direct"]
    assert aggregate_map["structured"].execution_paths == ["structured_single_pass"]
    assert aggregate_map["verified"].execution_paths == ["verified_orchestration"]
    assert all(aggregate.stability_rate == pytest.approx(1.0) for aggregate in aggregates)
    assert all(aggregate.avg_prompt_calls >= 0.0 for aggregate in aggregates)
    # Coverage monotonicity: structured >= rapid, verified >= structured
    assert aggregate_map["structured"].avg_leaf_coverage == pytest.approx(1.0)
    assert aggregate_map["verified"].avg_leaf_coverage == pytest.approx(1.0)
    assert aggregate_map["structured"].avg_leaf_coverage >= aggregate_map["rapid"].avg_leaf_coverage
    assert aggregate_map["verified"].avg_leaf_coverage >= aggregate_map["structured"].avg_leaf_coverage

    summary = format_flow_benchmark_summary(aggregates)
    assert "variant | paths | pass/total | stable | avg ms | avg prompts" in summary
    assert "rapid | rapid_direct |" in summary
    assert "structured | structured_single_pass |" in summary


@pytest.mark.asyncio
async def test_flow_benchmark_repeat_stability_groups_cases():
    cases = default_flow_benchmark_cases()

    results = await run_flow_benchmark(cases=cases[:1], repeats=2)
    aggregates = summarize_flow_benchmark(results)

    assert aggregates
    assert all(aggregate.repeat_groups == 1 for aggregate in aggregates)
    assert all(aggregate.stable_groups == 1 for aggregate in aggregates)
    assert all(aggregate.total_prompt_calls >= aggregate.total_cases for aggregate in aggregates)
