from __future__ import annotations

import json

import pytest

from ageom.hunter.prompts import REFORMULATE_QUERY_SYSTEM, REFORMULATE_QUERY_USER
from ageom.hunter.query_reformulator import (
    HeuristicQueryReformulator,
    derive_catalog_hints,
)
from ageom.types import Declaration, Prover


class _FallbackLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.response

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


def _reformulate_prompt(
    *,
    predicate_id: str,
    statement: str,
    informal_desc: str,
    prover: str,
    queries_tried: list[str],
    compiler_errors: str,
    extra: str = "",
) -> tuple[str, str]:
    user = REFORMULATE_QUERY_USER.format(
        predicate_id=predicate_id,
        statement=statement,
        informal_desc=informal_desc,
        prover=prover,
        queries_tried="\n".join(f"- {query}" for query in queries_tried),
        compiler_errors=compiler_errors,
    )
    if extra:
        user += extra
    return REFORMULATE_QUERY_SYSTEM, user


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statement", "informal_desc", "compiler_errors", "required_terms"),
    [
        (
            "Bandpass raw ECG into cardiac frequency region",
            "stable digital filter design and application",
            "Expected filtered_signal but got response tuple from compute_frequency_response",
            ["filter", "ecg", "bandpass"],
        ),
        (
            "Compute shortest path distances from source",
            "weighted directed graph traversal",
            "Candidate topological_sort returns ordering, not distances",
            ["shortest", "distance", "dijkstra"],
        ),
        (
            "Solve SPD linear system",
            "matrix factorization with triangular solves",
            "qr_decomposition does not return a solved vector",
            ["cholesky", "solve", "spd"],
        ),
        (
            "Find longest common subsequence",
            "dynamic programming recurrence over strings",
            "kmp_search finds pattern matches, not longest subsequence",
            ["longest common subsequence", "dynamic programming", "lcs"],
        ),
    ],
)
async def test_query_reformulator_matches_benchmark_style_cases(
    statement: str,
    informal_desc: str,
    compiler_errors: str,
    required_terms: list[str],
):
    fallback = _FallbackLLM('["fallback"]')
    reformulator = HeuristicQueryReformulator(fallback)
    system, user = _reformulate_prompt(
        predicate_id="p_case",
        statement=statement,
        informal_desc=informal_desc,
        prover="python",
        queries_tried=["generic query"],
        compiler_errors=compiler_errors,
    )

    response = await reformulator.complete(system, user)

    queries = [str(item).lower() for item in json.loads(response)]
    assert any(any(term in query for term in required_terms) for query in queries)
    assert len(queries) >= 3
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_query_reformulator_respects_speculative_local_count():
    fallback = _FallbackLLM('["fallback"]')
    reformulator = HeuristicQueryReformulator(fallback)
    system, user = _reformulate_prompt(
        predicate_id="p_graph",
        statement="Compute shortest path distances from source",
        informal_desc="weighted directed graph traversal",
        prover="python",
        queries_tried=["graph shortest path"],
        compiler_errors="Candidate topological_sort returns ordering, not distances",
        extra=(
            "\n\nGenerate exactly 4 highly diverse queries that maximize synonym "
            "and namespace coverage."
        ),
    )

    response = await reformulator.complete_with_grammar(system, user, "ignored")

    queries = json.loads(response)
    assert len(queries) == 4
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_query_reformulator_falls_back_for_ambiguous_prompt():
    fallback = _FallbackLLM('["fallback query"]')
    reformulator = HeuristicQueryReformulator(fallback)
    system, user = _reformulate_prompt(
        predicate_id="p_generic",
        statement="Process data",
        informal_desc="choose a helper",
        prover="python",
        queries_tried=["process data"],
        compiler_errors="type mismatch",
    )

    response = await reformulator.complete(system, user)

    assert json.loads(response) == ["fallback query"]
    assert fallback.calls == 1


def test_derive_catalog_hints_prefers_relevant_namespaces_and_declarations():
    class _Index:
        def __init__(self) -> None:
            self._declarations = [
                Declaration(
                    name="Data.PriorityQueue.popMin",
                    type_signature="queue[node] -> node",
                    docstring="Pop the next frontier node from the priority queue.",
                    prover=Prover.LEAN4,
                ),
                Declaration(
                    name="Data.PriorityQueue.insert",
                    type_signature="node -> queue[node] -> queue[node]",
                    docstring="Insert a node into the priority queue frontier.",
                    prover=Prover.LEAN4,
                ),
                Declaration(
                    name="Nat.add_comm",
                    type_signature="forall n m, n + m = m + n",
                    docstring="Commutativity of addition",
                    prover=Prover.LEAN4,
                ),
            ]

    hints = derive_catalog_hints(
        _Index(),
        statement="Extract the next frontier node",
        informal_desc="priority queue based graph search",
        compiler_errors="wrong helper returned an ordering",
        queries_tried=["priority queue graph"],
    )

    assert any("namespace:Data.PriorityQueue" == hint for hint in hints)
    assert any("declaration:Data.PriorityQueue.popMin" == hint for hint in hints)


@pytest.mark.asyncio
async def test_query_reformulator_uses_catalog_hints_for_generic_prompt():
    fallback = _FallbackLLM('["fallback query"]')
    reformulator = HeuristicQueryReformulator(fallback)
    system, user = _reformulate_prompt(
        predicate_id="p_catalog",
        statement="Extract the next frontier node",
        informal_desc="search helper",
        prover="lean4",
        queries_tried=["frontier node"],
        compiler_errors="helper returned ordering",
    )
    user += (
        "\n\n## Catalog Hints\n"
        "- namespace:Data.PriorityQueue\n"
        "- declaration:Data.PriorityQueue.popMin"
    )

    response = await reformulator.complete(system, user)

    queries = json.loads(response)
    assert any("Data.PriorityQueue" in query for query in queries)
    assert fallback.calls == 0
