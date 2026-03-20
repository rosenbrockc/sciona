from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sciona.architect.prompts import SELECT_STRATEGY_SYSTEM, SELECT_STRATEGY_USER
from sciona.architect.strategy_classifier import StrategyClassifier, _load_phrase_rules


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


@pytest.mark.asyncio
async def test_loads_from_data_file():
    """Verify that the default data file loads and produces identical classifications."""
    # Load rules explicitly from the default data path.
    from sciona.architect.strategy_classifier import _DEFAULT_DATA_PATH

    phrase_rules, conjunction_rules = _load_phrase_rules(_DEFAULT_DATA_PATH)
    assert len(phrase_rules) > 0
    assert len(conjunction_rules) > 0

    # Classifier with explicit path should match default behaviour.
    fallback = _FallbackLLM('{"paradigm":"custom","rationale":"fallback"}')
    classifier_default = StrategyClassifier(fallback)
    classifier_from_file = StrategyClassifier(fallback, rules_path=_DEFAULT_DATA_PATH)

    cases = [
        "Compute the longest common subsequence of two strings.",
        "Design and apply a stable bandpass filter to ECG samples.",
        "Compute shortest path distances from a source node in a weighted graph.",
    ]
    for goal in cases:
        result_default = classifier_default.classify(goal)
        result_file = classifier_from_file.classify(goal)
        assert result_default is not None
        assert result_file is not None
        assert result_default[0] == result_file[0], f"concept mismatch for: {goal}"
        assert result_default[3] == result_file[3], f"variant mismatch for: {goal}"


@pytest.mark.asyncio
async def test_custom_conjunction_rule():
    """A custom conjunction rule in a temp JSON should fire correctly."""
    custom_rules = {
        "phrase_rules": [
            {"phrase": "fizzbuzz", "concept": "arithmetic", "weight": 4.0},
        ],
        "conjunction_rules": [
            {
                "name": "custom_arith_combo",
                "sets": {
                    "compute": ["compute", "calculate"],
                    "modular": ["modular", "modulo", "mod"],
                    "batch": ["batch", "bulk", "array"],
                },
                "result_concept": "number_theory",
                "result_confidence": 0.95,
                "result_variant": "modular_batch",
            }
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(custom_rules, tmp)
        tmp_path = tmp.name

    try:
        fallback = _FallbackLLM('{"paradigm":"custom","rationale":"fallback"}')
        classifier = StrategyClassifier(fallback, rules_path=tmp_path)

        # Conjunction rule should fire when all three sets are matched.
        result = classifier.classify(
            "Compute modular residues for a batch of inputs.",
            allowed=None,
        )
        assert result is not None
        concept, confidence, rationale, variant = result
        assert concept.value == "number_theory"
        assert variant == "modular_batch"
        assert confidence == 0.95
    finally:
        Path(tmp_path).unlink()
