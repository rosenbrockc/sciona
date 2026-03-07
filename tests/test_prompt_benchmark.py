from __future__ import annotations

import json

import pytest

from ageom.cli import _parse_prompt_benchmark_provider_specs
from ageom.prompt_benchmark import (
    PromptBenchmarkProvider,
    default_prompt_benchmark_cases,
    format_prompt_benchmark_summary,
    run_prompt_benchmark,
    save_prompt_benchmark_report,
    select_prompt_benchmark_cases,
    summarize_prompt_benchmark,
)


class _GoodBenchmarkLLM:
    def __init__(self, model: str = "fake-good") -> None:
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


class _BadBenchmarkLLM:
    def __init__(self, model: str = "fake-bad") -> None:
        self._telemetry_model = model

    async def complete(self, system: str, user: str) -> str:
        return "I think the answer is probably candidate zero."

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


def test_default_prompt_benchmark_suite_covers_domains_and_keys():
    cases = default_prompt_benchmark_cases()
    assert len(cases) == 12
    assert {case.domain for case in cases} == {"dsp", "graph", "linear_algebra", "strings"}
    assert {case.prompt_key for case in cases} == {
        "hunter_score",
        "hunter_reformulate",
        "hunter_analyze_failure",
    }


def test_select_prompt_benchmark_cases_filters_by_key():
    cases = select_prompt_benchmark_cases(prompt_keys=["hunter_score"])
    assert cases
    assert all(case.prompt_key == "hunter_score" for case in cases)


def test_parse_prompt_benchmark_provider_specs():
    assert _parse_prompt_benchmark_provider_specs(["codex_shim:gpt-5.3-codex"]) == [
        ("codex_shim", "gpt-5.3-codex")
    ]


def test_parse_prompt_benchmark_provider_specs_rejects_invalid_specs():
    with pytest.raises(ValueError, match="provider:model"):
        _parse_prompt_benchmark_provider_specs(["codex_shim"])


@pytest.mark.asyncio
async def test_run_prompt_benchmark_with_good_provider_passes_all_cases(tmp_path):
    providers = [PromptBenchmarkProvider(name="good", client=_GoodBenchmarkLLM())]
    results = await run_prompt_benchmark(
        providers=providers,
        cases=default_prompt_benchmark_cases(),
    )
    assert results
    assert all(result.ok for result in results)

    aggregates = summarize_prompt_benchmark(results)
    assert len(aggregates) == 1
    assert aggregates[0].passed_cases == len(results)

    report_path = tmp_path / "prompt_benchmark.json"
    save_prompt_benchmark_report(report_path, results=results, aggregates=aggregates)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(payload["results"]) == len(results)
    assert len(payload["aggregates"]) == 1


@pytest.mark.asyncio
async def test_run_prompt_benchmark_summarizes_provider_quality():
    providers = [
        PromptBenchmarkProvider(name="good", client=_GoodBenchmarkLLM()),
        PromptBenchmarkProvider(name="bad", client=_BadBenchmarkLLM()),
    ]
    results = await run_prompt_benchmark(
        providers=providers,
        cases=default_prompt_benchmark_cases(),
    )
    aggregates = summarize_prompt_benchmark(results)
    assert [aggregate.provider for aggregate in aggregates][:2] == ["good", "bad"]
    assert aggregates[0].passed_cases > aggregates[1].passed_cases
    summary = format_prompt_benchmark_summary(aggregates)
    assert "provider | model | pass/total" in summary
