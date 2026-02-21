from __future__ import annotations

import statistics

import pytest

from ageom.hunter.graph import HunterAgent
from ageom.hunter.llm import LlamaCppLLMClient
from tests.helpers.match_regression import (
    FixtureOracle,
    StaticSemanticIndex,
    alias_hit_rate,
    load_case_pdg_nodes,
    match_results_to_name_map,
    run_stability_score,
)


@pytest.mark.match_regression_live
@pytest.mark.asyncio
async def test_live_hunter_matching_is_stable(
    match_cases,
    ageo_atoms_declarations,
    llama_server,
):
    index = StaticSemanticIndex(ageo_atoms_declarations)

    for case in match_cases:
        pdg_nodes = load_case_pdg_nodes(case)
        node_ids = [node.predicate_id for node in pdg_nodes]
        run_maps: list[dict[str, str]] = []
        success_rates: list[float] = []
        alias_rates: list[float] = []

        for _ in range(case.live_runs):
            llm = LlamaCppLLMClient(
                model=llama_server["model"],
                max_tokens=1024,
                base_url=llama_server["base_url"],
                api_key="local",
            )
            oracle = FixtureOracle(case.aliases)
            agent = HunterAgent(
                index=index,
                oracle=oracle,
                llm=llm,
                max_iterations=3,
                top_k_verify=10,
                search_k=50,
                mode="standard",
                use_gbnf=False,
            )

            observed_results = {}
            for node in pdg_nodes:
                observed_results[node.predicate_id] = await agent.find_match(node)

            name_map = match_results_to_name_map(observed_results)
            run_maps.append(name_map)

            successes = sum(1 for node_id in node_ids if name_map.get(node_id, ""))
            success_rate = successes / len(node_ids) if node_ids else 1.0
            success_rates.append(success_rate)
            alias_rates.append(alias_hit_rate(name_map, case.aliases))

        stability = run_stability_score(run_maps, node_ids)

        min_success = case.live_thresholds.get("min_success_rate", 0.6)
        min_alias = case.live_thresholds.get("min_alias_hit_rate", 0.6)
        min_stability = case.live_thresholds.get("min_run_stability", 0.6)

        assert statistics.median(success_rates) >= min_success, (
            f"Live success-rate regression for case '{case.case_id}'. "
            f"rates={success_rates}, threshold={min_success}"
        )
        assert statistics.median(alias_rates) >= min_alias, (
            f"Live alias-hit regression for case '{case.case_id}'. "
            f"rates={alias_rates}, threshold={min_alias}"
        )
        assert stability >= min_stability, (
            f"Live run-stability regression for case '{case.case_id}'. "
            f"stability={stability}, threshold={min_stability}, runs={run_maps}"
        )
