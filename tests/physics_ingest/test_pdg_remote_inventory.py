from __future__ import annotations

from scripts.physics_inventory_pdg_remote import (
    analyze_core_files,
    build_pdg_remote_inventory,
)


def test_pdg_remote_inventory_classifies_parseable_derivation_wave() -> None:
    core_texts = {
        "conversion_of_data_formats/deriv.cypher": (
            'UNWIND [{id:"000001",\n'
            '         properties:{name_latex:"Newton solve"}}] AS row\n'
            "CREATE (n:derivation{id: row.id}) SET n += row.properties;\n"
        ),
        "conversion_of_data_formats/infrules.cypher": (
            'UNWIND [{id:"111111",\n'
            '         properties:{name_latex:"solve for x"}}] AS row\n'
            "CREATE (n:inference_rule{id: row.id}) SET n += row.properties;\n"
        ),
        "conversion_of_data_formats/steps.cypher": (
            'UNWIND [{id:"222222", properties:{}}] AS row\n'
            "CREATE (n:step{id: row.id}) SET n += row.properties;\n"
            'UNWIND [{start: {id:"000001"}, end: {id:"222222"}, '
            "properties:{sequence_index:1}}] AS row\n"
            "MATCH (start:derivation{id: row.start.id})\n"
            "MATCH (end:step{id: row.end.id})\n"
            "CREATE (start)-[r:HAS_STEP]->(end) SET r += row.properties;\n"
            'UNWIND [{start: {id:"222222"}, end: {id:"111111"}, properties:{}}] AS row\n'
            "MATCH (start:step{id: row.start.id})\n"
            "MATCH (end:inference_rule{id: row.end.id})\n"
            "CREATE (start)-[r:HAS_INFERENCE_RULE]->(end) SET r += row.properties;\n"
            'UNWIND [{start: {id:"222222"}, end: {id:"expr-in"}, properties:{sequence_index:"0"}}] AS row\n'
            "MATCH (start:step{id: row.start.id})\n"
            "MATCH (end:expression{id: row.end.id})\n"
            "CREATE (start)-[r:HAS_INPUT]->(end) SET r += row.properties;\n"
            'UNWIND [{start: {id:"222222"}, end: {id:"expr-out"}, properties:{sequence_index:"0"}}] AS row\n'
            "MATCH (start:step{id: row.start.id})\n"
            "MATCH (end:expression{id: row.end.id})\n"
            "CREATE (start)-[r:HAS_OUTPUT]->(end) SET r += row.properties;\n"
        ),
    }

    analysis = analyze_core_files(core_texts)
    inventory = build_pdg_remote_inventory(
        repo="allofphysicsgraph/ui_v8_website_flask_neo4j",
        ref="gh-pages",
        repo_metadata={
            "full_name": "allofphysicsgraph/ui_v8_website_flask_neo4j",
            "default_branch": "gh-pages",
            "license": {"spdx_id": "CC-BY-4.0"},
        },
        tree_payload={
            "truncated": False,
            "tree": [
                {
                    "type": "blob",
                    "path": "conversion_of_data_formats/steps.cypher",
                    "sha": "abc",
                    "size": 123,
                },
                {
                    "type": "blob",
                    "path": "webserver_for_pdg/Dockerfile",
                    "sha": "def",
                    "size": 456,
                },
            ],
        },
        core_texts=core_texts,
        request_log=(),
    )

    assert analysis["node_counts"]["derivation"] == 1
    assert analysis["relationship_counts"]["HAS_INPUT"] == 1
    assert inventory["summary"]["derivation_count"] == 1
    assert inventory["derivations"][0]["recommended_wave"] == "wave_1_cdg_candidates"
    assert inventory["ingestion_waves"][1]["items"][0]["name_latex"] == "Newton solve"
    assert inventory["tree_inventory"]["by_role"]["core_graph_payload"] == 1
