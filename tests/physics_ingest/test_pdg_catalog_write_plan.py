from __future__ import annotations

import json

from sciona.physics_ingest.pdg_cdg import (
    PDGCDGArtifactEnvelope,
    build_pdg_cdg_catalog_write_plan_rows,
    build_pdg_publication_write_rows,
    build_pdg_relationship_ingest,
)
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


def test_pdg_cdg_catalog_rows_merge_into_publication_write_plan_order() -> None:
    result = build_pdg_cdg_catalog_write_plan_rows(
        _publication_rows(),
        include_write_plan=True,
    )

    assert result.write_plan is not None
    assert result.write_plan.ordered_tables() == (
        "artifacts",
        "artifact_versions",
        "artifact_relationships",
        "artifact_cdg_nodes",
        "artifact_cdg_edges",
        "artifact_cdg_bindings",
        "catalog_cdg_artifacts",
        "catalog_cdg_versions",
        "catalog_cdg_nodes",
        "catalog_cdg_relationships",
        "catalog_symbolic_artifacts",
    )
    assert result.summary["write_plan_table_order"] == list(
        result.write_plan.ordered_tables()
    )
    assert result.summary["catalog_projection_table_row_counts"] == {
        "catalog_cdg_artifacts": 1,
        "catalog_cdg_nodes": 2,
        "catalog_cdg_relationships": 3,
        "catalog_cdg_versions": 1,
        "catalog_symbolic_artifacts": 1,
    }
    assert result.summary["merged_row_count"] == 19
    assert result.summary["write_plan_total_row_count"] == 19
    json.dumps(result.to_dict(), sort_keys=True)

    rows_only = build_pdg_cdg_catalog_write_plan_rows(_publication_rows())
    assert rows_only.write_plan is None
    assert rows_only.summary["write_plan_table_order"] == list(
        result.write_plan.ordered_tables()
    )


def test_pdg_cdg_catalog_tables_have_write_plan_conflict_metadata() -> None:
    result = build_pdg_cdg_catalog_write_plan_rows(
        _publication_rows().to_insert_rows(),
        include_write_plan=True,
    )

    assert result.write_plan is not None
    batches_by_table = result.write_plan.batches_by_table()
    assert batches_by_table["catalog_cdg_artifacts"].conflict_keys == (
        "artifact_id",
        "version_id",
        "projection_kind",
    )
    assert batches_by_table["catalog_cdg_versions"].conflict_keys == (
        "version_id",
        "projection_kind",
    )
    assert batches_by_table["catalog_cdg_nodes"].conflict_keys == (
        "version_id",
        "node_id",
        "projection_kind",
    )
    assert batches_by_table["catalog_cdg_relationships"].conflict_keys == (
        "relationship_id",
        "projection_kind",
    )
    assert batches_by_table["catalog_symbolic_artifacts"].conflict_keys == (
        "artifact_id",
        "version_id",
        "projection_kind",
    )


def test_pdg_cdg_catalog_write_plan_preserves_missing_envelope_diagnostics() -> None:
    result = build_pdg_cdg_catalog_write_plan_rows(
        _publication_rows(include_envelope=False),
        include_write_plan=True,
    )

    assert result.write_plan is not None
    assert [diagnostic["reason"] for diagnostic in result.diagnostics] == [
        "missing_version_envelope",
        "orphan_cdg_row",
        "no_catalogable_cdgs",
    ]
    assert result.summary["diagnostics_by_reason"] == {
        "missing_version_envelope": 1,
        "no_catalogable_cdgs": 1,
        "orphan_cdg_row": 1,
    }
    assert "catalog_symbolic_artifacts" not in result.to_insert_rows()
    assert result.summary["write_plan_table_order"] == [
        "artifact_relationships",
        "artifact_cdg_nodes",
        "artifact_cdg_edges",
        "artifact_cdg_bindings",
    ]
