from __future__ import annotations

import pytest

from sciona.physics_ingest.sources.pdg import (
    PDGAdapter,
    extract_derivation_cdg_sketch,
    parse_pdg_document,
)


def _sample_pdg_payload() -> dict:
    return {
        "equations": [
            {
                "id": "eq:newton_second_law",
                "label": "Newton's second law",
                "description": "Net force equals mass times acceleration.",
                "latex": "F = m a",
                "mechanism_tags": ["dynamics", "force_balance"],
            },
            {
                "id": "eq:acceleration_solved",
                "label": "Acceleration from force",
                "latex": "a = F / m",
            },
            {
                "id": "eq:constant_mass_force",
                "label": "Force for constant mass",
                "latex": "F(t) = m \\frac{d^2 x}{dt^2}",
            },
            {
                "id": "note:not_equation",
                "type": "comment",
                "label": "Metadata note",
            },
        ],
        "inference_edges": [
            {
                "id": "edge:solve_for_a",
                "source": "eq:newton_second_law",
                "target": "eq:acceleration_solved",
                "rule": "solve for acceleration",
                "confidence": 0.93,
                "bindings": {"solve_for": "a"},
            },
            {
                "id": "edge:substitute_acceleration",
                "source": "eq:acceleration_solved",
                "target": "eq:constant_mass_force",
                "rule": "substitution",
                "assumptions": ["mass is constant"],
                "confidence": 0.8,
            },
            {
                "id": "edge:ignored_missing_endpoint",
                "source": "eq:constant_mass_force",
                "rule": "limit",
            },
        ],
    }


def test_parse_pdg_document_emits_wave0_snapshot_and_candidate_rows() -> None:
    bundle = parse_pdg_document(
        _sample_pdg_payload(),
        source_uri="file:///fixtures/pdg/newton.json",
        source_version="fixture-v1",
        retrieved_at="2026-04-30T00:00:00+00:00",
        license_expression="CC-BY-SA-4.0",
        provenance_summary="Offline fixture derived from PDG shape.",
    )

    assert bundle.snapshot_row["source_system"] == "physics_derivation_graph"
    assert bundle.snapshot_row["adapter_name"] == "sciona.physics_ingest.sources.pdg"
    assert len(bundle.snapshot_row["payload_sha256"]) == 64

    rows = bundle.candidate_rows(snapshot_id="snapshot-1")
    assert [row["source_candidate_id"] for row in rows] == [
        "eq:newton_second_law",
        "eq:acceleration_solved",
        "eq:constant_mass_force",
    ]
    assert rows[0]["snapshot_id"] == "snapshot-1"
    assert rows[0]["raw_formula_format"] == "latex"
    assert rows[0]["candidate_status"] == "raw_imported"
    assert rows[0]["mechanism_tags"] == ["dynamics", "force_balance"]


def test_pdg_edges_become_relationship_hints_with_wave0_direction() -> None:
    adapter = PDGAdapter()
    bundle = adapter.parse_document(_sample_pdg_payload())

    hints = bundle.relationship_hints
    assert len(hints) == 2

    solve_hint = hints[0]
    assert solve_hint.relationship_kind == "algebraic_rearrangement_of"
    assert solve_hint.source_node_id == "eq:acceleration_solved"
    assert solve_hint.target_node_id == "eq:newton_second_law"
    assert solve_hint.inference_rule_id == "solve_for_acceleration"
    assert solve_hint.evidence_json["pdg_edge_id"] == "edge:solve_for_a"

    rows = bundle.relationship_rows(
        expression_id_by_pdg_node_id={
            "eq:newton_second_law": "expr-newton",
            "eq:acceleration_solved": "expr-accel",
            "eq:constant_mass_force": "expr-force",
        }
    )
    assert rows[0]["source_expression_id"] == "expr-accel"
    assert rows[0]["target_expression_id"] == "expr-newton"
    assert rows[0]["source_kind"] == "physics_derivation_graph"
    assert rows[0]["verified"] is False


def test_relationship_materialization_fails_closed_until_expressions_exist() -> None:
    bundle = parse_pdg_document(_sample_pdg_payload())

    with pytest.raises(KeyError):
        bundle.relationship_rows(
            expression_id_by_pdg_node_id={
                "eq:newton_second_law": "expr-newton",
            }
        )


def test_extract_derivation_cdg_sketch_for_solve_substitute_chain() -> None:
    bundle = parse_pdg_document(_sample_pdg_payload())
    sketch = extract_derivation_cdg_sketch(
        bundle.inference_edges,
        equation_labels={node.node_id: node.label for node in bundle.equations},
    )

    data = sketch.to_dict()
    assert data["metadata"]["sketch_kind"] == "solve_substitute_limit_chain"
    assert [node["operation_kind"] for node in data["nodes"]] == [
        "solve",
        "substitute",
    ]
    assert data["nodes"][0]["output_equation_id"] == "eq:acceleration_solved"
    assert data["edges"] == [
        {
            "source_id": "pdg_derivation_step_1",
            "target_id": "pdg_derivation_step_2",
            "equation_id": "eq:acceleration_solved",
            "edge_kind": "symbolic_equation_flow",
        }
    ]
