"""End-to-end prompt benchmark regression across simple domains."""

from __future__ import annotations

import pytest

from ageom.prompt_benchmark import (
    PromptBenchmarkProvider,
    default_prompt_benchmark_cases,
    run_prompt_benchmark,
    summarize_prompt_benchmark,
)


class _DeterministicCrossDomainLLM:
    def __init__(self, provider: str, *, degrade_reformulate: bool = False) -> None:
        self._telemetry_model = f"{provider}-model"
        self._provider = provider
        self._degrade_reformulate = degrade_reformulate

    async def complete(self, system: str, user: str) -> str:
        lower = system.lower()
        user_lower = user.lower()
        if "json array of integer indices" in lower:
            return "[0, 1]"
        if "json array of strings" in lower:
            if self._degrade_reformulate:
                return '["generic query", "more search"]'
            if "ecg" in user_lower:
                return '["ecg bandpass filter", "stable ecg filter"]'
            if "shortest path" in user_lower:
                return '["dijkstra shortest path", "distance map graph"]'
            if "spd" in user_lower:
                return '["cholesky solve spd", "solve positive definite"]'
            return '["longest common subsequence", "dynamic programming lcs"]'
        if "return exactly three lines" in lower:
            if "ecg" in user_lower or "filter" in user_lower:
                return "CAUSE: wrong output\nTARGET: filter signal primitive\nNEXT: search ecg filter"
            if "shortest path" in user_lower or "distance" in user_lower:
                return "CAUSE: wrong return type\nTARGET: distance path routine\nNEXT: search dijkstra"
            if "spd" in user_lower:
                return "CAUSE: missing solve step\nTARGET: solve vector routine\nNEXT: search cholesky solve"
            return "CAUSE: wrong algorithm\nTARGET: subsequence dynamic routine\nNEXT: search lcs"
        return ""

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


@pytest.mark.asyncio
async def test_cross_domain_prompt_benchmark_e2e():
    cases = default_prompt_benchmark_cases()
    providers = [
        PromptBenchmarkProvider(
            name="provider_a",
            client=_DeterministicCrossDomainLLM("provider_a"),
        ),
        PromptBenchmarkProvider(
            name="provider_b",
            client=_DeterministicCrossDomainLLM("provider_b", degrade_reformulate=True),
        ),
    ]

    results = await run_prompt_benchmark(providers=providers, cases=cases, repeats=1)

    assert len(results) == len(cases) * len(providers)
    assert {result.domain for result in results} == {
        "dsp",
        "graph",
        "linear_algebra",
        "strings",
    }

    aggregates = summarize_prompt_benchmark(results)
    top = next(aggregate for aggregate in aggregates if aggregate.provider == "provider_a")
    weaker = next(aggregate for aggregate in aggregates if aggregate.provider == "provider_b")

    assert top.passed_cases == len(cases)
    assert weaker.failed_cases == 4
    assert weaker.by_prompt_key["hunter_reformulate"]["failed"] == 4
