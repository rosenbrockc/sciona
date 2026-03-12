from __future__ import annotations

import pytest

from ageom.hunter.failure_analyzer import DeterministicFailureAnalyzer
from ageom.hunter.prompts import ANALYZE_FAILURE_SYSTEM, ANALYZE_FAILURE_USER


class _FallbackLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.response

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


def _analyze_prompt(
    *,
    statement: str,
    candidate_name: str,
    candidate_type: str,
    compiler_output: str,
) -> tuple[str, str]:
    return (
        ANALYZE_FAILURE_SYSTEM,
        ANALYZE_FAILURE_USER.format(
            statement=statement,
            candidate_name=candidate_name,
            candidate_type=candidate_type,
            compiler_output=compiler_output,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statement", "candidate_name", "candidate_type", "compiler_output", "required_terms"),
    [
        (
            "Apply a stable ECG filter",
            "compute_frequency_response",
            "coeffs -> response",
            "Type mismatch: expected filtered_signal but got frequency_response",
            ["CAUSE: Type mismatch", "TARGET:", "filtered_signal", "NEXT:"],
        ),
        (
            "Find shortest path distances",
            "topological_sort",
            "graph -> ordering",
            "unknown identifier 'Graph.Path.distanceMap'",
            ["CAUSE: Unknown identifier", "distanceMap", "namespace"],
        ),
        (
            "Solve SPD linear system",
            "qr_decomposition",
            "matrix -> q_r",
            'Argument 1 to "solve" has incompatible type "Matrix"; expected "Vector"',
            ["CAUSE: Argument type mismatch", "Matrix", "Vector"],
        ),
    ],
)
async def test_failure_analyzer_matches_benchmark_style_cases(
    statement: str,
    candidate_name: str,
    candidate_type: str,
    compiler_output: str,
    required_terms: list[str],
):
    fallback = _FallbackLLM("CAUSE: fallback\nTARGET: fallback\nNEXT: fallback")
    analyzer = DeterministicFailureAnalyzer(fallback)
    system, user = _analyze_prompt(
        statement=statement,
        candidate_name=candidate_name,
        candidate_type=candidate_type,
        compiler_output=compiler_output,
    )

    response = await analyzer.complete(system, user)

    for term in required_terms:
        assert term in response
    assert analyzer.get_last_completion_metadata()["analysis_source"] == "deterministic"
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_failure_analyzer_falls_back_for_ambiguous_compiler_output():
    fallback = _FallbackLLM("CAUSE: fallback\nTARGET: helper\nNEXT: search helper")
    analyzer = DeterministicFailureAnalyzer(fallback)
    system, user = _analyze_prompt(
        statement="Process data",
        candidate_name="helper",
        candidate_type="data -> data",
        compiler_output="candidate does not seem suitable",
    )

    response = await analyzer.complete(system, user)

    assert response == "CAUSE: fallback\nTARGET: helper\nNEXT: search helper"
    assert analyzer.get_last_completion_metadata()["analysis_source"] == "fallback"
    assert fallback.calls == 1
