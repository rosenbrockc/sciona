from __future__ import annotations

import json

import pytest

from sciona.cli import _parse_prompt_benchmark_provider_specs
from sciona.prompt_benchmark import (
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
        if "algorithm critic" in lower or ("evaluate" in lower and '"approved"' in lower):
            return '{"approved": true, "reason": "Decomposition is correct and complete"}'
        if "decompose" in lower and ("sub-nodes" in lower or "sub_nodes" in lower):
            user_lower = user.lower()
            if "filter" in user_lower or "ecg" in user_lower:
                return '{"sub_nodes": [{"name": "Design Coefficients", "description": "Compute bandpass coefficients"}, {"name": "Apply Filter", "description": "Apply to signal"}]}'
            if "shortest path" in user_lower or "graph" in user_lower:
                return '{"sub_nodes": [{"name": "Init Distances", "description": "Set source 0, rest inf"}, {"name": "Relax Edges", "description": "Improve estimates"}]}'
            if "linear" in user_lower or "spd" in user_lower:
                return '{"sub_nodes": [{"name": "Cholesky Factor", "description": "Lower triangular factor"}, {"name": "Triangular Solve", "description": "Forward/back substitution"}]}'
            return '{"sub_nodes": [{"name": "Build DP Table", "description": "Fill LCS lengths"}, {"name": "Backtrack", "description": "Reconstruct LCS"}]}'
        if "paradigm" in lower and ("select" in lower or "algorithmic" in lower):
            user_lower = user.lower()
            if "filter" in user_lower or "ecg" in user_lower:
                return '{"paradigm": "signal_filter", "rationale": "Bandpass filtering"}'
            if "shortest path" in user_lower or "weighted graph" in user_lower:
                return '{"paradigm": "graph_optimization", "rationale": "SSSP"}'
            if "linear system" in user_lower or "symmetric positive definite" in user_lower:
                return '{"paradigm": "algebra", "rationale": "Linear algebra"}'
            return '{"paradigm": "dynamic_programming", "rationale": "DP for LCS"}'
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
    assert len(cases) == 22
    assert {case.domain for case in cases} == {"dsp", "graph", "linear_algebra", "strings"}
    assert {case.prompt_key for case in cases} == {
        "hunter_score",
        "hunter_reformulate",
        "hunter_analyze_failure",
        "architect_strategy",
        "architect_decompose",
        "architect_critique",
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
    assert aggregates[0].stability_rate == pytest.approx(1.0)

    report_path = tmp_path / "prompt_benchmark.json"
    save_prompt_benchmark_report(report_path, results=results, aggregates=aggregates)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(payload["results"]) == len(results)
    assert len(payload["aggregates"]) == 1
    assert payload["aggregates"][0]["variant"] == "tuned"


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
    assert "provider | variant | model | pass/total | stable" in summary


@pytest.mark.asyncio
async def test_run_prompt_benchmark_compare_direct_baseline_adds_variants():
    providers = [PromptBenchmarkProvider(name="good", client=_GoodBenchmarkLLM())]
    cases = select_prompt_benchmark_cases(prompt_keys=["hunter_score"])
    results = await run_prompt_benchmark(
        providers=providers,
        cases=cases,
        compare_direct_baseline=True,
    )

    assert len(results) == len(cases) * 2
    assert {result.variant for result in results} == {"tuned", "direct_baseline"}

    aggregates = summarize_prompt_benchmark(results)
    assert {aggregate.variant for aggregate in aggregates} == {"tuned", "direct_baseline"}


@pytest.mark.asyncio
async def test_prompt_benchmark_stability_tracks_repeat_consistency():
    providers = [PromptBenchmarkProvider(name="good", client=_GoodBenchmarkLLM())]
    cases = select_prompt_benchmark_cases(prompt_keys=["hunter_score"])
    results = await run_prompt_benchmark(
        providers=providers,
        cases=cases,
        repeats=2,
    )

    aggregates = summarize_prompt_benchmark(results)
    assert len(aggregates) == 1
    assert aggregates[0].repeat_groups == len(cases)
    assert aggregates[0].stable_groups == len(cases)
    assert aggregates[0].stability_rate == pytest.approx(1.0)
