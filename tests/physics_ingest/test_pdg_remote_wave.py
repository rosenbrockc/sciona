from __future__ import annotations

from scripts.physics_ingest_pdg_remote_wave import (
    build_selected_derivation_payloads,
    parse_remote_pdg_cypher,
)


def test_remote_wave_converts_cypher_derivation_to_pdg_payload() -> None:
    graph = parse_remote_pdg_cypher(
        {
            "conversion_of_data_formats/deriv.cypher": (
                'UNWIND [{id:"000008", properties:{name_latex:"frequency relations"}}] AS row\n'
                "CREATE (n:derivation{id: row.id}) SET n += row.properties;\n"
            ),
            "conversion_of_data_formats/infrules.cypher": (
                'UNWIND [{id:"111111", properties:{name_latex:"divide both sides"}}] AS row\n'
                "CREATE (n:inference_rule{id: row.id}) SET n += row.properties;\n"
            ),
            "conversion_of_data_formats/expr_and_feed.cypher": (
                'UNWIND [{id:"expr-a", properties:{latex_lhs:"T", latex_relation:"=", latex_rhs:"1/f"}}] AS row\n'
                "CREATE (n:expression{id: row.id}) SET n += row.properties;\n"
                'UNWIND [{id:"expr-b", properties:{latex_lhs:"f", latex_relation:"=", latex_rhs:"1/T"}}] AS row\n'
                "CREATE (n:expression{id: row.id}) SET n += row.properties;\n"
                'UNWIND [{id:"feed-a", properties:{latex:"T"}}] AS row\n'
                "CREATE (n:feed{id: row.id}) SET n += row.properties;\n"
            ),
            "conversion_of_data_formats/steps.cypher": (
                'UNWIND [{id:"step-a", properties:{}}] AS row\n'
                "CREATE (n:step{id: row.id}) SET n += row.properties;\n"
                'UNWIND [{start: {id:"000008"}, end: {id:"step-a"}, properties:{sequence_index:1}}] AS row\n'
                "MATCH (start:derivation{id: row.start.id})\n"
                "MATCH (end:step{id: row.end.id})\n"
                "CREATE (start)-[r:HAS_STEP]->(end) SET r += row.properties;\n"
                'UNWIND [{start: {id:"step-a"}, end: {id:"111111"}, properties:{}}] AS row\n'
                "MATCH (start:step{id: row.start.id})\n"
                "MATCH (end:inference_rule{id: row.end.id})\n"
                "CREATE (start)-[r:HAS_INFERENCE_RULE]->(end) SET r += row.properties;\n"
                'UNWIND [{start: {id:"step-a"}, end: {id:"expr-a"}, properties:{sequence_index:"0"}}] AS row\n'
                "MATCH (start:step{id: row.start.id})\n"
                "MATCH (end:expression{id: row.end.id})\n"
                "CREATE (start)-[r:HAS_INPUT]->(end) SET r += row.properties;\n"
                'UNWIND [{start: {id:"step-a"}, end: {id:"feed-a"}, properties:{sequence_index:"0"}}] AS row\n'
                "MATCH (start:step{id: row.start.id})\n"
                "MATCH (end:feed{id: row.end.id})\n"
                "CREATE (start)-[r:HAS_FEED]->(end) SET r += row.properties;\n"
                'UNWIND [{start: {id:"step-a"}, end: {id:"expr-b"}, properties:{sequence_index:"0"}}] AS row\n'
                "MATCH (start:step{id: row.start.id})\n"
                "MATCH (end:expression{id: row.end.id})\n"
                "CREATE (start)-[r:HAS_OUTPUT]->(end) SET r += row.properties;\n"
            ),
        }
    )

    payloads = build_selected_derivation_payloads(
        graph,
        ["000008"],
        repo="allofphysicsgraph/ui_v8_website_flask_neo4j",
        ref="gh-pages",
        source_version="test-version",
        license_expression="CC-BY-4.0",
        license_evidence_url="https://derivationmap.net/developer_documentation",
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["derivation_label"] == "frequency relations"
    assert [row["id"] for row in payload["equations"]] == ["expr-a", "expr-b"]
    assert payload["equations"][0]["latex"] == "T = 1/f"
    assert payload["inference_edges"][0]["source"] == "expr-a"
    assert payload["inference_edges"][0]["target"] == "expr-b"
    assert payload["inference_edges"][0]["bindings"]["feeds"] == [
        {"feed_id": "feed-a", "latex": "T", "sympy": ""}
    ]
