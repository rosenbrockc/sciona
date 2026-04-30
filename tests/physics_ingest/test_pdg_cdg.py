from __future__ import annotations

from sciona.physics_ingest.pdg_cdg import build_pdg_relationship_ingest
from sciona.physics_ingest.sources.pdg import parse_pdg_document


EXPR_BASE = "10000000-0000-0000-0000-000000000001"
EXPR_SOLVED = "10000000-0000-0000-0000-000000000002"
EXPR_FORCE = "10000000-0000-0000-0000-000000000003"
ARTIFACT_BASE = "20000000-0000-0000-0000-000000000001"
VERSION_BASE = "30000000-0000-0000-0000-000000000001"


def _bundle():
    return parse_pdg_document(
        {
            "equations": [
                {
                    "id": "eq:base",
                    "label": "Newton's second law",
                    "latex": "F = m a",
                },
                {
                    "id": "eq:solved",
                    "label": "Acceleration from force",
                    "latex": "a = F / m",
                },
                {
                    "id": "eq:force",
                    "label": "Constant mass force",
                    "latex": "F(t) = m d^2x/dt^2",
                },
            ],
            "inference_edges": [
                {
                    "id": "edge:solve",
                    "source": "eq:base",
                    "target": "eq:solved",
                    "rule": "solve for acceleration",
                    "confidence": 0.93,
                    "bindings": {"solve_for": "a"},
                },
                {
                    "id": "edge:substitute",
                    "source": "eq:solved",
                    "target": "eq:force",
                    "rule": "substitution",
                    "assumptions": ["mass is constant"],
                    "confidence": 0.81,
                },
            ],
        }
    )


def test_pdg_phase4_builds_validated_artifact_relationship_rows() -> None:
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": EXPR_BASE,
            "eq:solved": EXPR_SOLVED,
            "eq:force": EXPR_FORCE,
        },
    )

    rows = result.relationship_insert_rows()

    assert result.skipped_edges == ()
    assert [row["relationship_kind"] for row in rows] == [
        "algebraic_rearrangement_of",
        "derives_from",
    ]
    assert rows[0]["source_expression_id"] == EXPR_SOLVED
    assert rows[0]["target_expression_id"] == EXPR_BASE
    assert rows[0]["source_kind"] == "physics_derivation_graph"
    assert rows[0]["verified"] is False
    assert rows[1]["source_expression_id"] == EXPR_FORCE
    assert rows[1]["target_expression_id"] == EXPR_SOLVED


def test_pdg_phase4_extracts_cdg_candidate_manifest_with_expression_refs() -> None:
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": {
                "expression_id": EXPR_BASE,
                "label": "F = m a",
                "artifact_id": ARTIFACT_BASE,
                "version_id": VERSION_BASE,
            },
            "eq:solved": EXPR_SOLVED,
            "eq:force": EXPR_FORCE,
        },
    )

    manifest = result.cdg_candidate_manifests[0]

    assert manifest["manifest_kind"] == "pdg_derivation_chain_candidate"
    assert manifest["source_system"] == "physics_derivation_graph"
    assert [node["operation_kind"] for node in manifest["nodes"]] == [
        "solve",
        "substitute",
    ]
    assert manifest["nodes"][0]["input_expressions"] == [
        {
            "pdg_node_id": "eq:base",
            "expression_id": EXPR_BASE,
            "label": "F = m a",
            "artifact_id": ARTIFACT_BASE,
            "version_id": VERSION_BASE,
        }
    ]
    assert manifest["nodes"][0]["output_expression"]["expression_id"] == EXPR_SOLVED
    assert manifest["edges"] == [
        {
            "source_id": "pdg_step_1",
            "target_id": "pdg_step_2",
            "edge_kind": "symbolic_equation_flow",
            "pdg_node_id": "eq:solved",
            "expression_id": EXPR_SOLVED,
        }
    ]
    assert manifest["metadata"]["relationship_edge_ids"] == [
        "edge:solve",
        "edge:substitute",
    ]


def test_pdg_phase4_skips_edges_missing_expression_bindings() -> None:
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": EXPR_BASE,
            "eq:solved": EXPR_SOLVED,
        },
    )

    assert len(result.artifact_relationship_rows) == 1
    assert result.cdg_candidate_manifests[0]["metadata"]["relationship_edge_ids"] == [
        "edge:solve"
    ]
    assert result.skipped_edges == (
        {
            "pdg_edge_id": "edge:substitute",
            "source_node_id": "eq:solved",
            "target_node_id": "eq:force",
            "reason": "missing_expression_binding",
            "missing_node_ids": ["eq:force"],
        },
    )


def test_pdg_phase4_can_scope_to_named_chain_edges() -> None:
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": EXPR_BASE,
            "eq:solved": EXPR_SOLVED,
            "eq:force": EXPR_FORCE,
        },
        chain_edge_ids=["edge:substitute"],
    )

    rows = result.relationship_insert_rows()
    manifest = result.cdg_candidate_manifests[0]

    assert len(rows) == 1
    assert rows[0]["inference_rule_id"] == "substitution"
    assert [node["pdg_edge_id"] for node in manifest["nodes"]] == ["edge:substitute"]
    assert manifest["edges"] == []
