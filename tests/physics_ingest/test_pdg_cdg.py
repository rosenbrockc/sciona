from __future__ import annotations

import json
from pathlib import Path

from sciona.physics_ingest.pdg_cdg import (
    PDGCDGArtifactEnvelope,
    build_pdg_publication_write_rows,
    build_pdg_relationship_ingest,
    validate_pdg_cdg_publication_graph,
)
from sciona.physics_ingest.sources.pdg import parse_pdg_document
from sciona.physics_ingest.write_plan import build_publication_write_plan


EXPR_BASE = "10000000-0000-0000-0000-000000000001"
EXPR_SOLVED = "10000000-0000-0000-0000-000000000002"
EXPR_FORCE = "10000000-0000-0000-0000-000000000003"
EXPR_DIMENSIONAL = "10000000-0000-0000-0000-000000000004"
EXPR_NONDIMENSIONAL = "10000000-0000-0000-0000-000000000005"
EXPR_LIMIT = "10000000-0000-0000-0000-000000000006"
EXPR_POSITION = "10000000-0000-0000-0000-000000000007"
EXPR_VELOCITY = "10000000-0000-0000-0000-000000000008"
EXPR_POSITION_FROM_VELOCITY = "10000000-0000-0000-0000-000000000009"
EXPR_DIMENSIONAL_DECAY = "10000000-0000-0000-0000-000000000010"
EXPR_NONDIMENSIONAL_DECAY = "10000000-0000-0000-0000-000000000011"
EXPR_FIRST_ORDER_DECAY = "10000000-0000-0000-0000-000000000012"
EXPR_INTEGRAL_CONSERVATION = "10000000-0000-0000-0000-000000000013"
EXPR_CONTINUITY = "10000000-0000-0000-0000-000000000014"
EXPR_DIFFUSION = "10000000-0000-0000-0000-000000000015"
ARTIFACT_BASE = "20000000-0000-0000-0000-000000000001"
VERSION_BASE = "30000000-0000-0000-0000-000000000001"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pdg_payloads"


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


def _fixture_bundle(name: str):
    return parse_pdg_document(json.loads((FIXTURE_DIR / name).read_text()))


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
    assert validate_pdg_cdg_publication_graph(publication_rows) == ()


def test_pdg_phase4_extracts_limit_and_nondimensionalization_derivations() -> None:
    result = build_pdg_relationship_ingest(
        _fixture_bundle("limit_nondimensionalization_chain.pdg.json"),
        expression_bindings_by_pdg_node_id={
            "eq:damped_oscillator_dimensional": {
                "expression_id": EXPR_DIMENSIONAL,
                "label": "dimensional oscillator",
                "metadata": {
                    "bound_artifact_fqdn": "physics.oscillator.dimensional",
                    "bound_version_content_hash": "hash-dimensional",
                },
            },
            "eq:damped_oscillator_nondim": {
                "expression_id": EXPR_NONDIMENSIONAL,
                "label": "nondimensional oscillator",
                "metadata": {
                    "bound_artifact_fqdn": "physics.oscillator.nondimensional",
                    "bound_version_content_hash": "hash-nondim",
                },
            },
            "eq:undamped_limit": {
                "expression_id": EXPR_LIMIT,
                "label": "undamped limit",
                "metadata": {
                    "bound_artifact_fqdn": "physics.oscillator.undamped_limit",
                    "bound_version_content_hash": "hash-limit",
                },
            },
        },
    )

    rows = result.relationship_insert_rows()
    manifest = result.cdg_candidate_manifests[0]

    assert result.skipped_edges == ()
    assert [row["relationship_kind"] for row in rows] == [
        "derives_from",
        "limit_case_of",
    ]
    assert [node["operation_kind"] for node in manifest["nodes"]] == [
        "derive",
        "limit",
    ]
    assert [node["relationship_kind"] for node in manifest["nodes"]] == [
        "derives_from",
        "limit_case_of",
    ]
    assert manifest["edges"] == [
        {
            "source_id": "pdg_step_1",
            "target_id": "pdg_step_2",
            "edge_kind": "symbolic_equation_flow",
            "pdg_node_id": "eq:damped_oscillator_nondim",
            "expression_id": EXPR_NONDIMENSIONAL,
        }
    ]
    assert manifest["metadata"]["relationship_edge_ids"] == [
        "edge:nondimensionalize_oscillator",
        "edge:zero_damping_limit",
    ]

    publication_rows = build_pdg_publication_write_rows(result)
    assert validate_pdg_cdg_publication_graph(publication_rows) == ()


def test_pdg_phase4_extracts_differentiate_and_integrate_derivations() -> None:
    result = build_pdg_relationship_ingest(
        _fixture_bundle("differentiate_integrate_chain.pdg.json"),
        expression_bindings_by_pdg_node_id={
            "eq:position": {
                "expression_id": EXPR_POSITION,
                "label": "position",
                "metadata": {
                    "bound_artifact_fqdn": "physics.kinematics.position",
                    "bound_version_content_hash": "hash-position",
                },
            },
            "eq:velocity": {
                "expression_id": EXPR_VELOCITY,
                "label": "velocity",
                "metadata": {
                    "bound_artifact_fqdn": "physics.kinematics.velocity",
                    "bound_version_content_hash": "hash-velocity",
                },
            },
            "eq:position_from_velocity": {
                "expression_id": EXPR_POSITION_FROM_VELOCITY,
                "label": "position from velocity",
                "metadata": {
                    "bound_artifact_fqdn": "physics.kinematics.position_from_velocity",
                    "bound_version_content_hash": "hash-position-from-velocity",
                },
            },
        },
    )

    rows = result.relationship_insert_rows()
    manifest = result.cdg_candidate_manifests[0]

    assert result.skipped_edges == ()
    assert [row["relationship_kind"] for row in rows] == [
        "derives_from",
        "derives_from",
    ]
    assert [node["operation_kind"] for node in manifest["nodes"]] == [
        "differentiate",
        "integrate",
    ]
    assert [node["relationship_kind"] for node in manifest["nodes"]] == [
        "derives_from",
        "derives_from",
    ]
    assert manifest["edges"] == [
        {
            "source_id": "pdg_step_1",
            "target_id": "pdg_step_2",
            "edge_kind": "symbolic_equation_flow",
            "pdg_node_id": "eq:velocity",
            "expression_id": EXPR_VELOCITY,
        }
    ]
    assert manifest["metadata"]["relationship_edge_ids"] == [
        "edge:differentiate_position",
        "edge:integrate_velocity",
    ]

    publication_rows = build_pdg_publication_write_rows(result)
    assert validate_pdg_cdg_publication_graph(publication_rows) == ()


def test_pdg_phase4_extracts_nondimensionalize_and_approximate_derivations() -> None:
    bundle = _fixture_bundle("nondimensionalize_approximate_chain.pdg.json")
    result = build_pdg_relationship_ingest(
        bundle,
        expression_bindings_by_pdg_node_id={
            "eq:dimensional_decay": {
                "expression_id": EXPR_DIMENSIONAL_DECAY,
                "label": "dimensional decay",
                "metadata": {
                    "bound_artifact_fqdn": "physics.decay.dimensional",
                    "bound_version_content_hash": "hash-dimensional-decay",
                },
            },
            "eq:nondimensional_decay": {
                "expression_id": EXPR_NONDIMENSIONAL_DECAY,
                "label": "nondimensional decay",
                "metadata": {
                    "bound_artifact_fqdn": "physics.decay.nondimensional",
                    "bound_version_content_hash": "hash-nondimensional-decay",
                },
            },
            "eq:first_order_decay": {
                "expression_id": EXPR_FIRST_ORDER_DECAY,
                "label": "first order decay approximation",
                "metadata": {
                    "bound_artifact_fqdn": "physics.decay.first_order",
                    "bound_version_content_hash": "hash-first-order-decay",
                },
            },
        },
    )

    rows = result.relationship_insert_rows()
    manifest = result.cdg_candidate_manifests[0]

    assert [edge.operation_kind for edge in bundle.inference_edges] == [
        "nondimensionalize",
        "approximate",
    ]
    assert result.skipped_edges == ()
    assert [row["relationship_kind"] for row in rows] == [
        "derives_from",
        "approximation_of",
    ]
    assert [row["evidence_json"]["operation_kind"] for row in rows] == [
        "nondimensionalize",
        "approximate",
    ]
    assert [node["operation_kind"] for node in manifest["nodes"]] == [
        "nondimensionalize",
        "approximate",
    ]
    assert [node["relationship_kind"] for node in manifest["nodes"]] == [
        "derives_from",
        "approximation_of",
    ]
    assert manifest["edges"] == [
        {
            "source_id": "pdg_step_1",
            "target_id": "pdg_step_2",
            "edge_kind": "symbolic_equation_flow",
            "pdg_node_id": "eq:nondimensional_decay",
            "expression_id": EXPR_NONDIMENSIONAL_DECAY,
        }
    ]
    assert manifest["metadata"]["relationship_edge_ids"] == [
        "edge:nondimensionalize_decay",
        "edge:first_order_approximation_decay",
    ]

    publication_rows = build_pdg_publication_write_rows(result)
    assert validate_pdg_cdg_publication_graph(publication_rows) == ()


def test_pdg_phase4_extracts_conservation_law_to_pde_derivation_chain() -> None:
    bundle = _fixture_bundle("conservation_pde_chain.pdg.json")
    result = build_pdg_relationship_ingest(
        bundle,
        expression_bindings_by_pdg_node_id={
            "eq:integral_conservation_balance": {
                "expression_id": EXPR_INTEGRAL_CONSERVATION,
                "label": "integral conservation balance",
                "metadata": {
                    "bound_artifact_fqdn": "physics.transport.integral_conservation",
                    "bound_version_content_hash": "hash-integral-conservation",
                },
            },
            "eq:continuity_equation": {
                "expression_id": EXPR_CONTINUITY,
                "label": "continuity equation",
                "metadata": {
                    "bound_artifact_fqdn": "physics.transport.continuity",
                    "bound_version_content_hash": "hash-continuity",
                },
            },
            "eq:diffusion_equation": {
                "expression_id": EXPR_DIFFUSION,
                "label": "diffusion equation",
                "metadata": {
                    "bound_artifact_fqdn": "physics.transport.diffusion",
                    "bound_version_content_hash": "hash-diffusion",
                },
            },
        },
    )

    rows = result.relationship_insert_rows()
    manifest = result.cdg_candidate_manifests[0]

    assert [edge.operation_kind for edge in bundle.inference_edges] == [
        "derive",
        "substitute",
    ]
    assert result.skipped_edges == ()
    assert [row["relationship_kind"] for row in rows] == [
        "derives_from",
        "derives_from",
    ]
    assert [node["operation_kind"] for node in manifest["nodes"]] == [
        "derive",
        "substitute",
    ]
    assert [node["relationship_kind"] for node in manifest["nodes"]] == [
        "derives_from",
        "derives_from",
    ]
    assert manifest["edges"] == [
        {
            "source_id": "pdg_step_1",
            "target_id": "pdg_step_2",
            "edge_kind": "symbolic_equation_flow",
            "pdg_node_id": "eq:continuity_equation",
            "expression_id": EXPR_CONTINUITY,
        }
    ]
    assert manifest["metadata"]["relationship_edge_ids"] == [
        "edge:derive_continuity_from_balance",
        "edge:substitute_fick_law",
    ]

    publication_rows = build_pdg_publication_write_rows(result)
    assert validate_pdg_cdg_publication_graph(publication_rows) == ()


def test_pdg_phase4_cdg_graph_validator_accepts_valid_insert_rows() -> None:
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": EXPR_BASE,
            "eq:solved": EXPR_SOLVED,
            "eq:force": EXPR_FORCE,
        },
    )
    publication_rows = build_pdg_publication_write_rows(result)

    diagnostics = validate_pdg_cdg_publication_graph(publication_rows.to_insert_rows())

    assert diagnostics == ()


def test_pdg_phase4_cdg_graph_validator_reports_deterministic_graph_errors() -> None:
    rows = {
        "artifact_versions": [{"version_id": "ver-a"}],
        "artifact_cdg_nodes": [
            {"version_id": "ver-a", "node_id": "step-1"},
            {"version_id": "ver-a", "node_id": "step-1"},
            {"version_id": "", "node_id": "step-missing-version"},
            {"version_id": "ver-orphan", "node_id": "step-orphan"},
        ],
        "artifact_cdg_edges": [
            {
                "version_id": "ver-a",
                "source_id": "step-1",
                "target_id": "step-missing",
                "output_name": "out",
                "input_name": "in",
            },
            {
                "version_id": "ver-a",
                "source_id": "step-1",
                "target_id": "step-missing",
                "output_name": "out",
                "input_name": "in",
            },
            {"version_id": "ver-a", "source_id": "", "target_id": "step-1"},
        ],
        "artifact_cdg_bindings": [
            {
                "version_id": "ver-a",
                "node_id": "step-missing",
                "bound_artifact_fqdn": "physics.missing",
            },
            {"version_id": "", "node_id": "step-1"},
            "not-a-row",
        ],
    }

    diagnostics = validate_pdg_cdg_publication_graph(rows)

    assert [diagnostic["reason"] for diagnostic in diagnostics] == [
        "malformed_row",
        "duplicate_node_key",
        "missing_cdg_row_identity",
        "edge_endpoint_node_missing",
        "duplicate_edge_key",
        "edge_endpoint_node_missing",
        "missing_cdg_row_identity",
        "binding_node_missing",
        "missing_cdg_row_identity",
        "orphan_cdg_version",
    ]
    assert all(diagnostic["stage"] == "pdg_cdg_publication" for diagnostic in diagnostics)
    assert all(diagnostic["severity"] == "error" for diagnostic in diagnostics)
    json.dumps(diagnostics, sort_keys=True)

    duplicate_node_detail = json.loads(diagnostics[1]["detail"])
    assert duplicate_node_detail == {
        "first_row_index": 0,
        "key": ["ver-a", "step-1"],
        "row_index": 1,
    }
    missing_edge_endpoint_detail = json.loads(diagnostics[3]["detail"])
    assert missing_edge_endpoint_detail["missing_node_ids"] == ["step-missing"]


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

    assert validate_pdg_cdg_publication_graph(publication_rows) == ()
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
