from __future__ import annotations

import pytest

from sciona.hunter.graph import HunterAgent
from tests.helpers.match_regression import (
    DeterministicHunterLLM,
    FixtureOracle,
    StaticSemanticIndex,
    load_case_pdg_nodes,
    match_results_to_name_map,
)


@pytest.mark.match_regression_deterministic
def test_match_case_fixture_paths_exist(match_cases):
    match_cases = [case for case in match_cases if not case.historical]
    if not match_cases:
        pytest.skip("legacy regression fixtures are historical-only in this slice")

    for case in match_cases:
        assert case.cdg_path.exists(), f"Missing CDG fixture: {case.cdg_path}"
        assert (
            case.witness_test_path.exists()
        ), f"Missing witness-test reference: {case.witness_test_path}"


@pytest.mark.match_regression_deterministic
@pytest.mark.asyncio
async def test_deterministic_hunter_matches_expected(
    match_cases,
    sciona_atoms_declarations,
):
    match_cases = [case for case in match_cases if not case.historical]
    if not match_cases:
        pytest.skip("legacy regression fixtures are historical-only in this slice")

    index = StaticSemanticIndex(sciona_atoms_declarations)
    llm = DeterministicHunterLLM()

    for case in match_cases:
        pdg_nodes = load_case_pdg_nodes(case)
        oracle = FixtureOracle(case.aliases)
        agent = HunterAgent(
            index=index,
            oracle=oracle,
            llm=llm,
            max_iterations=2,
            top_k_verify=10,
            search_k=50,
            mode="standard",
            use_gbnf=False,
        )

        observed_results = {}
        for node in pdg_nodes:
            result = await agent.find_match(node)
            observed_results[node.predicate_id] = result
            assert result.success, (
                f"Case '{case.case_id}' failed for node '{node.predicate_id}': "
                f"{[vr.error_message for vr in result.all_verifications]}"
            )

        observed_map = match_results_to_name_map(observed_results)
        assert set(observed_map) == set(case.expected_matches), (
            f"Node IDs mismatch for case '{case.case_id}'. "
            f"expected={sorted(case.expected_matches)} observed={sorted(observed_map)}"
        )
        assert observed_map == case.expected_matches
