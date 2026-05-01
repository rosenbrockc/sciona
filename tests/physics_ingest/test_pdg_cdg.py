from __future__ import annotations

from sciona.physics_ingest.pdg_cdg import (
    PDGCDGArtifactEnvelope,
    build_pdg_publication_write_rows,
    build_pdg_relationship_ingest,
)
from sciona.physics_ingest.sources.pdg import parse_pdg_document
from sciona.physics_ingest.write_plan import build_publication_write_plan


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


def test_pdg_phase4_publication_rows_merge_relationships_and_cdg_tables() -> None:
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": {
                "expression_id": EXPR_BASE,
                "metadata": {
                    "bound_artifact_fqdn": "physics.newton.base",
                    "bound_version_content_hash": "hash-base",
                    "binding_confidence": 0.92,
                    "binding_source": "fixture",
                },
            },
            "eq:solved": {
                "expression_id": EXPR_SOLVED,
                "metadata": {
                    "bound_artifact_fqdn": "physics.newton.solved",
                    "bound_version_content_hash": "hash-solved",
                },
            },
            "eq:force": EXPR_FORCE,
        },
    )

    publication_rows = build_pdg_publication_write_rows(result)
    insert_rows = publication_rows.to_insert_rows()
    plan = build_publication_write_plan(insert_rows)

    assert plan.ordered_tables() == (
        "artifact_relationships",
        "artifact_cdg_nodes",
        "artifact_cdg_edges",
        "artifact_cdg_bindings",
    )
    assert len(insert_rows["artifact_relationships"]) == 2
    assert [row["node_id"] for row in insert_rows["artifact_cdg_nodes"]] == [
        "pdg_step_1",
        "pdg_step_2",
    ]
    assert insert_rows["artifact_cdg_edges"] == [
        {
            "version_id": insert_rows["artifact_cdg_nodes"][0]["version_id"],
            "source_id": "pdg_step_1",
            "target_id": "pdg_step_2",
            "output_name": EXPR_SOLVED,
            "input_name": "input",
        }
    ]
    assert insert_rows["artifact_cdg_bindings"][0] == {
        "version_id": insert_rows["artifact_cdg_nodes"][0]["version_id"],
        "node_id": "pdg_step_1",
        "bound_artifact_fqdn": "physics.newton.base",
        "bound_version_content_hash": "hash-base",
        "binding_confidence": 0.92,
        "binding_source": "fixture",
    }
    assert {
        diagnostic["reason"] for diagnostic in publication_rows.diagnostics
    } == {"missing_cdg_binding_artifact_metadata"}


def test_pdg_phase4_publication_rows_are_deterministic() -> None:
    kwargs = {
        "expression_bindings_by_pdg_node_id": {
            "eq:base": EXPR_BASE,
            "eq:solved": EXPR_SOLVED,
            "eq:force": EXPR_FORCE,
        }
    }

    first = build_pdg_publication_write_rows(
        build_pdg_relationship_ingest(_bundle(), **kwargs)
    )
    second = build_pdg_publication_write_rows(
        build_pdg_relationship_ingest(_bundle(), **kwargs)
    )

    assert first.to_insert_rows() == second.to_insert_rows()
    assert first.diagnostics == second.diagnostics


def test_pdg_phase4_publication_can_emit_cdg_artifact_envelope_rows() -> None:
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": EXPR_BASE,
            "eq:solved": EXPR_SOLVED,
            "eq:force": EXPR_FORCE,
        },
    )

    publication_rows = build_pdg_publication_write_rows(
        result,
        cdg_artifact_envelope=PDGCDGArtifactEnvelope(
            fqdn_prefix="physics.pdg.cdg",
            semver="2026.5.1",
            namespace_root="physics",
            namespace_path="pdg/cdg",
            source_repo_id="40000000-0000-0000-0000-000000000001",
            source_package="pdg",
            source_module_path="pdg.derivations",
            status="draft",
            visibility_tier="internal",
            is_latest=True,
            s3_key_prefix="physics/pdg/cdg",
        ),
    )
    insert_rows = publication_rows.to_insert_rows()
    plan = build_publication_write_plan(insert_rows)

    assert plan.ordered_tables() == (
        "artifacts",
        "artifact_versions",
        "artifact_relationships",
        "artifact_cdg_nodes",
        "artifact_cdg_edges",
    )
    artifact = insert_rows["artifacts"][0]
    version = insert_rows["artifact_versions"][0]
    cdg_node = insert_rows["artifact_cdg_nodes"][0]

    assert artifact["artifact_kind"] == "cdg"
    assert artifact["fqdn"].startswith("physics.pdg.cdg.pdg_cdg_candidate_")
    assert artifact["source_repo_id"] == "40000000-0000-0000-0000-000000000001"
    assert artifact["namespace_root"] == "physics"
    assert artifact["namespace_path"] == "pdg/cdg"
    assert artifact["source_kind"] == "generated"
    assert artifact["is_publishable"] is False
    assert version["artifact_id"] == artifact["artifact_id"]
    assert version["version_id"] == cdg_node["version_id"]
    assert version["semver"] == "2026.5.1"
    assert version["is_latest"] is True
    assert version["s3_key"] == f"physics/pdg/cdg/{artifact['fqdn']}.json"
    assert len(version["content_hash"]) == 64
    assert version["fingerprint"] == version["content_hash"]
    assert plan.batches_by_table()["artifacts"].conflict_keys == ("artifact_id",)
    assert plan.batches_by_table()["artifact_versions"].conflict_keys == ("version_id",)


def test_pdg_phase4_artifact_envelope_requires_fqdn_identity() -> None:
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": EXPR_BASE,
            "eq:solved": EXPR_SOLVED,
            "eq:force": EXPR_FORCE,
        },
    )

    try:
        build_pdg_publication_write_rows(
            result,
            cdg_artifact_envelope=PDGCDGArtifactEnvelope(),
        )
    except ValueError as exc:
        assert "fqdn_prefix" in str(exc)
    else:
        raise AssertionError("expected missing fqdn identity to fail")
