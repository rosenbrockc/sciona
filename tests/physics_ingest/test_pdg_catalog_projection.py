from __future__ import annotations

import json

from sciona.physics_ingest.pdg_cdg import (
    PDGCDGArtifactEnvelope,
    build_pdg_cdg_catalog_projection_rows,
    build_pdg_publication_write_rows,
    build_pdg_relationship_ingest,
)
from sciona.physics_ingest.retrieval import candidates_from_rows
from sciona.physics_ingest.sources.pdg import parse_pdg_document


EXPR_BASE = "10000000-0000-0000-0000-000000000001"
EXPR_SOLVED = "10000000-0000-0000-0000-000000000002"
EXPR_FORCE = "10000000-0000-0000-0000-000000000003"


def _bundle():
    return parse_pdg_document(
        {
            "equations": [
                {"id": "eq:base", "label": "Newton's second law", "latex": "F = m a"},
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
                    "bindings": {"variables": {"a": "d2x_dt2"}},
                    "confidence": 0.81,
                },
            ],
        }
    )


def _publication_rows(*, include_envelope: bool = True):
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": {
                "expression_id": EXPR_BASE,
                "metadata": {
                    "bound_artifact_fqdn": "physics.newton.base",
                    "bound_version_content_hash": "hash-base",
                },
            },
            "eq:solved": {
                "expression_id": EXPR_SOLVED,
                "metadata": {
                    "bound_artifact_fqdn": "physics.newton.solved",
                    "bound_version_content_hash": "hash-solved",
                },
            },
            "eq:force": {
                "expression_id": EXPR_FORCE,
                "metadata": {
                    "bound_artifact_fqdn": "physics.newton.force",
                    "bound_version_content_hash": "hash-force",
                },
            },
        },
    )
    envelope = (
        PDGCDGArtifactEnvelope(
            fqdn_prefix="physics.pdg.cdg",
            semver="2026.5.3",
            namespace_root="physics",
            namespace_path="pdg/cdg",
            source_package="pdg",
            source_module_path="pdg.derivations",
            is_latest=True,
        )
        if include_envelope
        else None
    )
    return build_pdg_publication_write_rows(result, cdg_artifact_envelope=envelope)


def test_pdg_cdg_catalog_projection_emits_searchable_cdg_rows() -> None:
    projection = build_pdg_cdg_catalog_projection_rows(_publication_rows())
    rows = projection.to_projection_rows()

    assert projection.diagnostics == ()
    assert set(rows) == {
        "catalog_cdg_artifacts",
        "catalog_cdg_versions",
        "catalog_cdg_nodes",
        "catalog_cdg_relationships",
        "catalog_symbolic_artifacts",
    }
    symbolic = rows["catalog_symbolic_artifacts"][0]

    assert symbolic["artifact_kind"] == "cdg"
    assert symbolic["source_system"] == "physics_derivation_graph"
    assert symbolic["fqdn"].startswith("physics.pdg.cdg.pdg_cdg_candidate_")
    assert symbolic["artifact_id"]
    assert symbolic["version_id"]
    assert symbolic["topology_hash"]
    assert symbolic["topo_hash"] == symbolic["topology_hash"]
    assert symbolic["content_hash"]
    assert symbolic["replay_key"].startswith("pdg-cdg-catalog-projection:")
    assert symbolic["operation_kinds"] == ["solve", "substitute"]
    assert symbolic["relationship_kinds"] == [
        "algebraic_rearrangement_of",
        "cdg_data_flow",
        "derives_from",
    ]
    assert symbolic["node_count"] == 2
    assert symbolic["edge_count"] == 1
    assert symbolic["expression_node_count"] == 3
    assert symbolic["relationship_count"] == 3
    assert symbolic["provenance"]["source_system"] == "physics_derivation_graph"

    candidate = candidates_from_rows([symbolic])[0]
    assert candidate.artifact_kind == "cdg"
    assert candidate.source_system == "physics_derivation_graph"
    assert candidate.topology_hash == symbolic["topology_hash"]
    assert {relationship.relationship_kind for relationship in candidate.relationships} == {
        "algebraic_rearrangement_of",
        "cdg_data_flow",
        "derives_from",
    }

    summary = projection.summary
    assert summary["projected_row_count"] == 8
    assert summary["catalogable_cdg_count"] == 1
    assert summary["diagnostic_count"] == 0
    assert summary["source_systems"] == ["physics_derivation_graph"]
    assert summary["operation_kinds"] == ["solve", "substitute"]
    assert summary["relationship_kinds"] == [
        "algebraic_rearrangement_of",
        "cdg_data_flow",
        "derives_from",
    ]


def test_pdg_cdg_catalog_projection_reports_missing_envelope_rows() -> None:
    projection = build_pdg_cdg_catalog_projection_rows(
        _publication_rows(include_envelope=False)
    )

    assert projection.to_projection_rows() == {}
    assert [diagnostic["reason"] for diagnostic in projection.diagnostics] == [
        "missing_version_envelope",
        "orphan_cdg_row",
        "no_catalogable_cdgs",
    ]
    assert projection.summary["diagnostic_count"] == 3
    assert projection.summary["diagnostics_by_reason"] == {
        "missing_version_envelope": 1,
        "no_catalogable_cdgs": 1,
        "orphan_cdg_row": 1,
    }


def test_pdg_cdg_catalog_projection_reports_orphan_and_malformed_rows() -> None:
    projection = build_pdg_cdg_catalog_projection_rows(
        {
            "artifact_versions": [
                {"version_id": "version-orphan", "artifact_id": "artifact-missing"}
            ],
            "artifact_cdg_nodes": [
                {
                    "version_id": "version-orphan",
                    "node_id": "pdg_step_1",
                    "matched_primitive": "solve",
                }
            ],
            "artifact_cdg_edges": [object()],
        }
    )

    assert projection.to_projection_rows() == {}
    assert [diagnostic["reason"] for diagnostic in projection.diagnostics] == [
        "malformed_row",
        "missing_artifact_envelope",
        "orphan_cdg_row",
        "no_catalogable_cdgs",
    ]
    assert all(
        diagnostic["stage"] == "pdg_cdg_catalog_projection"
        for diagnostic in projection.diagnostics
    )
    json.dumps(projection.to_dict(), sort_keys=True)


def test_pdg_cdg_catalog_projection_is_deterministic_and_json_safe() -> None:
    publication_rows = _publication_rows()

    first = build_pdg_cdg_catalog_projection_rows(publication_rows)
    second = build_pdg_cdg_catalog_projection_rows(publication_rows.to_insert_rows())

    assert first.to_dict() == second.to_dict()
    assert json.loads(json.dumps(first.to_dict(), sort_keys=True)) == first.to_dict()
