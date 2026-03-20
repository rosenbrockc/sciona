from __future__ import annotations

import json

import pytest

from sciona.hunter.candidate_ranker import HeuristicCandidateRanker
from sciona.hunter.prompts import SCORE_CANDIDATES_SYSTEM, SCORE_CANDIDATES_USER


class _FallbackLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.response

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


def _score_prompt(statement: str, informal_desc: str, candidates: list[tuple[str, str]]) -> tuple[str, str]:
    candidates_list = "\n".join(
        f"[{idx}] {name} : {type_sig}"
        for idx, (name, type_sig) in enumerate(candidates)
    )
    return (
        SCORE_CANDIDATES_SYSTEM,
        SCORE_CANDIDATES_USER.format(
            statement=statement,
            informal_desc=informal_desc,
            candidates_list=candidates_list,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statement", "informal_desc", "candidates"),
    [
        (
            "Apply a stable bandpass filter to ECG samples.",
            "select the filter primitive that directly filters the signal",
            [
                ("apply_iir_filter", "signal -> coeffs -> filtered_signal"),
                ("compute_frequency_response", "coeffs -> response"),
                ("plot_spectrum", "signal -> image"),
            ],
        ),
        (
            "Find shortest path distances from a source node.",
            "single-source shortest path on weighted graph",
            [
                ("dijkstra", "graph -> source -> distances"),
                ("topological_sort", "graph -> ordering"),
                ("union_find_merge", "state -> edge -> state"),
            ],
        ),
    ],
)
async def test_candidate_ranker_matches_benchmark_style_cases(statement: str, informal_desc: str, candidates):
    fallback = _FallbackLLM("[2, 1, 0]")
    ranker = HeuristicCandidateRanker(fallback)
    system, user = _score_prompt(statement, informal_desc, candidates)

    response = await ranker.complete(system, user)

    assert json.loads(response)[0] == 0
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_candidate_ranker_falls_back_for_ambiguous_candidates():
    fallback = _FallbackLLM("[1, 0]")
    ranker = HeuristicCandidateRanker(fallback)
    system, user = _score_prompt(
        "Process data.",
        "choose a helper",
        [
            ("helper_one", "data -> data"),
            ("helper_two", "data -> data"),
        ],
    )

    response = await ranker.complete(system, user)

    assert json.loads(response)[0] == 1
    assert fallback.calls == 1
