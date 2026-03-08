"""End-to-end regression for the small full-flow benchmark harness."""

from __future__ import annotations

import pytest

from ageom.flow_benchmark import (
    default_flow_benchmark_cases,
    run_flow_benchmark,
    summarize_flow_benchmark,
)


@pytest.mark.asyncio
async def test_flow_benchmark_e2e_compares_direct_baseline_against_modes():
    cases = default_flow_benchmark_cases()

    results = await run_flow_benchmark(cases=cases)

    assert len(results) == len(cases) * 4

    aggregates = {aggregate.variant: aggregate for aggregate in summarize_flow_benchmark(results)}

    assert aggregates["direct_baseline"].passed_cases == 0
    assert aggregates["rapid"].passed_cases == 0
    assert aggregates["structured"].passed_cases == len(cases)
    assert aggregates["verified"].passed_cases == len(cases)
