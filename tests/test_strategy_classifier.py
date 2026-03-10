from __future__ import annotations

import json

import pytest

from ageom.architect.prompts import SELECT_STRATEGY_SYSTEM, SELECT_STRATEGY_USER
from ageom.architect.strategy_classifier import StrategyClassifier


class _FallbackLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.response

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


def _strategy_prompts(goal: str, available: str) -> tuple[str, str]:
    return (
        SELECT_STRATEGY_SYSTEM.format(available_paradigms=available),
        SELECT_STRATEGY_USER.format(goal=goal),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("goal", "available", "expected"),
    [
        (
            "Design and apply a stable bandpass filter to ECG samples.",
            "sorting, searching, divide_and_conquer, greedy, dynamic_programming, graph_traversal, graph_optimization, string_matching, signal_transform, signal_filter",
            "signal_filter",
        ),
        (
            "Compute shortest path distances from a source node in a weighted graph.",
            "sorting, searching, divide_and_conquer, greedy, dynamic_programming, graph_traversal, graph_optimization, string_matching, signal_transform, signal_filter",
            "graph_optimization",
        ),
        (
            "Solve a symmetric positive definite linear system.",
            "sorting, searching, divide_and_conquer, greedy, dynamic_programming, graph_traversal, graph_optimization, algebra, analysis, signal_transform",
            "algebra",
        ),
        (
            "Compute the longest common subsequence of two strings.",
            "sorting, searching, divide_and_conquer, greedy, dynamic_programming, graph_traversal, string_matching, signal_transform",
            "dynamic_programming",
        ),
    ],
)
async def test_strategy_classifier_matches_prompt_benchmark_cases(goal: str, available: str, expected: str):
    fallback = _FallbackLLM('{"paradigm":"custom","rationale":"fallback"}')
    classifier = StrategyClassifier(fallback)
    system, user = _strategy_prompts(goal, available)

    response = await classifier.complete(system, user)

    payload = json.loads(response)
    assert payload["paradigm"] == expected
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_strategy_classifier_falls_back_for_ambiguous_goal():
    fallback = _FallbackLLM('{"paradigm":"custom","rationale":"fallback"}')
    classifier = StrategyClassifier(fallback)
    system, user = _strategy_prompts(
        "Process some input data and produce an output.",
        "sorting, searching, divide_and_conquer, greedy, dynamic_programming, graph_traversal, graph_optimization",
    )

    response = await classifier.complete(system, user)

    payload = json.loads(response)
    assert payload["paradigm"] == "custom"
    assert fallback.calls == 1
