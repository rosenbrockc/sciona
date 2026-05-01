from __future__ import annotations

import json

from sciona.physics_ingest.validation import (
    VALIDATION_REPORT_KIND,
    build_physics_ingestion_validation_report,
    validate_pdg_payload,
    validate_symbolic_publication_fixture,
)


def test_symbolic_publication_fixture_validator_accepts_complete_fixture(tmp_path) -> None:
    fixture_path = tmp_path / "fixture.publication_manifest.json"
    fixture_path.write_text(json.dumps(_symbolic_manifest()), encoding="utf-8")

    check = validate_symbolic_publication_fixture(fixture_path)

    assert check.ok is True
    assert check.issues == ()
    assert check.metadata == {
        "expression_count": 1,
        "variable_count": 3,
        "validity_bound_count": 1,
    }


def test_symbolic_publication_fixture_validator_reports_metadata_gaps(tmp_path) -> None:
    manifest = _symbolic_manifest()
    manifest["artifact_symbolic_expressions"][0]["mechanism_tags"] = []
    manifest["artifact_symbolic_variables"][1]["dim_signature"] = ""
    fixture_path = tmp_path / "fixture.publication_manifest.json"
    fixture_path.write_text(json.dumps(manifest), encoding="utf-8")

    check = validate_symbolic_publication_fixture(fixture_path)

    assert check.ok is False
    assert [issue.reason for issue in check.issues] == [
        "missing_mechanism_tags",
        "missing_dim_signature",
    ]


def test_pdg_payload_validator_accepts_graph_ready_derivation_fixture() -> None:
    check = validate_pdg_payload(_pdg_payload(), subject="fixture-pdg")

    assert check.ok is True
    assert check.issues == ()
    assert check.metadata == {
        "equation_count": 3,
        "inference_edge_count": 2,
        "relationship_row_count": 2,
        "cdg_node_count": 2,
        "cdg_edge_count": 1,
        "cdg_binding_count": 4,
    }


def test_validation_report_is_json_safe_and_fails_strict_without_fixtures() -> None:
    report = build_physics_ingestion_validation_report(
        fixture_paths=(),
        include_default_pdg=True,
        strict=True,
    )

    assert report["report_kind"] == VALIDATION_REPORT_KIND
    assert report["ok"] is False
    assert report["summary"] == {
        "check_count": 2,
        "failed_check_count": 1,
        "error_count": 1,
    }
    json.dumps(report, sort_keys=True)


def _symbolic_manifest() -> dict[str, object]:
    return {
        "provider": "fixture",
        "modules": ["fixture.physics"],
        "artifact_symbolic_expressions": [
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "expression_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "sympy_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "expression_text": "Eq(F, a*m)",
                "raw_formula": "Eq(F, a*m)",
                "raw_formula_format": "plain_text",
                "expression_kind": "equation",
                "expression_role": "primary",
                "canonical_expr_hash": "a" * 64,
                "topology_hash": "b" * 64,
                "dimensional_hash": "c" * 64,
                "parse_status": "normalized",
                "parse_confidence": 1.0,
                "review_status": "automated_pass",
                "validation_status": "passed",
                "mechanism_tags": ["conservation"],
                "behavioral_archetypes": ["source"],
                "variables": {"F": "output", "m": "input", "a": "input"},
                "dim_signature": {"F": "M1L1T-2", "m": "M1", "a": "L1T-2"},
                "symbolic_dim_signature": {
                    "F": "M1L1T-2",
                    "m": "M1",
                    "a": "L1T-2",
                },
                "constants": {},
                "bibliography": ["fixture"],
                "artifact_uuid": None,
            }
        ],
        "artifact_symbolic_variables": [
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "symbol": "F",
                "symbol_name": "F",
                "source_symbol": "F",
                "source_variable_id": "fixture:F",
                "role": "output",
                "variable_role": "output",
                "dim_signature": "M1L1T-2",
                "dimension_source": "source",
                "assumptions_json": {"dim_signature": "M1L1T-2"},
                "evidence_json": {"source_symbol": "F"},
                "ordinal": 0,
            },
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "symbol": "m",
                "symbol_name": "m",
                "source_symbol": "m",
                "source_variable_id": "fixture:m",
                "role": "input",
                "variable_role": "input",
                "dim_signature": "M1",
                "dimension_source": "source",
                "assumptions_json": {"dim_signature": "M1"},
                "evidence_json": {"source_symbol": "m"},
                "ordinal": 1,
            },
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "symbol": "a",
                "symbol_name": "a",
                "source_symbol": "a",
                "source_variable_id": "fixture:a",
                "role": "input",
                "variable_role": "input",
                "dim_signature": "L1T-2",
                "dimension_source": "source",
                "assumptions_json": {"dim_signature": "L1T-2"},
                "evidence_json": {"source_symbol": "a"},
                "ordinal": 2,
            },
        ],
        "artifact_validity_bounds": [
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "provider": "fixture",
                "atom_name": "force_atom",
                "atom_module": "fixture.physics",
                "registry_name": "force_atom",
                "expression_id": "10000000-0000-0000-0000-000000000001",
                "symbol": "m",
                "variable_name": "m",
                "source_symbol": "m",
                "source_bound_id": "fixture:m:bound",
                "scope": "variable",
                "bound_kind": "domain",
                "min_value": 0.0,
                "max_value": None,
                "lower_value": 0.0,
                "upper_value": None,
                "lower_inclusive": True,
                "upper_inclusive": True,
                "dim_signature": "M1",
                "validity_statement": "m >= 0.0",
                "evidence_ref_key": "fixture:m:bound",
                "confidence": "high",
                "review_status": "automated_pass",
                "metadata": {"provider": "fixture"},
                "ordinal": 0,
            }
        ],
    }


def _pdg_payload() -> dict[str, object]:
    return {
        "equations": [
            {"id": "eq:base", "label": "Newton second law", "latex": "F = m a"},
            {"id": "eq:solved", "label": "Acceleration", "latex": "a = F / m"},
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
                "confidence": 0.81,
            },
        ],
    }
