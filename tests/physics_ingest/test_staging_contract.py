from __future__ import annotations

import pytest
from pydantic import ValidationError

from sciona.physics_ingest.staging import (
    ArtifactRelationshipRow,
    SymbolicExpressionRow,
    Wave0ContractError,
    attach_snapshot_id,
    stage_source_rows,
    validate_artifact_relationship_row,
    validate_candidate_row,
    validate_snapshot_row,
)
from sciona.physics_ingest.sources.nist import (
    DLMFFunctionEntry,
    build_codata_wave0_bundle,
    build_dlmf_wave0_bundle,
    parse_codata_ascii,
)
from sciona.physics_ingest.sources.pdg import parse_pdg_document
from sciona.physics_ingest.sources.qudt import build_qudt_snapshot_manifest
from sciona.physics_ingest.sources.wikidata import (
    build_snapshot_record,
    build_wave0_candidate_records,
)


SNAPSHOT_ID = "00000000-0000-0000-0000-000000000001"
EXPRESSION_A = "10000000-0000-0000-0000-000000000001"
EXPRESSION_B = "10000000-0000-0000-0000-000000000002"
ARTIFACT_ID = "20000000-0000-0000-0000-000000000001"
VERSION_ID = "30000000-0000-0000-0000-000000000001"


def test_stage_source_rows_validates_wikidata_adapter_output() -> None:
    response = {
        "results": {
            "bindings": [
                {
                    "item": _binding("http://www.wikidata.org/entity/Q1", "uri"),
                    "itemLabel": _binding("Example law"),
                    "itemDescription": _binding("fixture physical equation"),
                    "formulaProperty": _binding(
                        "http://www.wikidata.org/prop/direct/P2534",
                        "uri",
                    ),
                    "formula": _binding("E = m c^2"),
                    "use": _binding("http://www.wikidata.org/entity/Q2", "uri"),
                    "useLabel": _binding("relativistic mechanics"),
                }
            ]
        }
    }
    snapshot = build_snapshot_record(
        query="SELECT * WHERE { ?s ?p ?o } LIMIT 1",
        response=response,
        source_version="2026-04-30",
    )
    candidates = build_wave0_candidate_records(response)

    staged_snapshot, staged_candidates = stage_source_rows(
        snapshot_row=snapshot,
        candidate_rows=candidates,
        snapshot_id=SNAPSHOT_ID,
    )

    assert staged_snapshot.source_system == "wikidata"
    assert staged_candidates[0].snapshot_id == SNAPSHOT_ID
    assert staged_candidates[0].raw_formula_format == "wikidata_math"
    assert staged_candidates[0].candidate_status == "raw_imported"
    assert staged_candidates[0].to_insert_dict()["snapshot_id"] == SNAPSHOT_ID


def test_stage_source_rows_validates_qudt_adapter_output() -> None:
    manifest = build_qudt_snapshot_manifest(
        [
            {
                "@id": "http://qudt.org/vocab/unit/N",
                "@type": ["qudt:Unit"],
                "rdfs:label": "Newton",
                "qudt:symbol": "N",
                "qudt:hasDimensionVector": "A0E0L1I0M1H0T-2D0",
            }
        ],
        source_version="qudt-fixture",
    )

    snapshot, candidates = stage_source_rows(
        snapshot_row=manifest.snapshot_row,
        candidate_rows=manifest.candidate_rows,
        snapshot_id=SNAPSHOT_ID,
    )

    assert snapshot.source_system == "qudt"
    assert candidates[0].candidate_status == "dimension_resolved"
    assert candidates[0].raw_formula == ""
    assert candidates[0].source_payload["dim_signature"] == "M1L1T-2"


def test_stage_source_rows_validates_nist_codata_and_dlmf_outputs() -> None:
    constants = parse_codata_ascii(
        "elementary charge | 1.602 176 634 e-19 | (exact) | C | symbol=e",
        source_version="CODATA 2022",
    )
    codata = build_codata_wave0_bundle(constants, source_version="CODATA 2022")
    codata_snapshot, codata_candidates = stage_source_rows(
        snapshot_row=codata.snapshot_row,
        candidate_rows=codata.candidate_rows,
        snapshot_id=SNAPSHOT_ID,
    )

    assert codata_snapshot.source_system == "nist_codata"
    assert codata_candidates[0].candidate_status == "source_verified"
    assert codata_candidates[0].parse_confidence == 1.0

    dlmf = build_dlmf_wave0_bundle(
        [
            DLMFFunctionEntry.from_mapping(
                {
                    "source_id": "DLMF:10.2.E2",
                    "label": "Bessel equation",
                    "formula": r"z^2 w'' + z w' + (z^2-\nu^2)w = 0",
                    "formula_format": "latex",
                },
                source_version="DLMF fixture",
            )
        ],
        source_version="DLMF fixture",
    )
    dlmf_snapshot, dlmf_candidates = stage_source_rows(
        snapshot_row=dlmf.snapshot_row,
        candidate_rows=dlmf.candidate_rows,
        snapshot_id="00000000-0000-0000-0000-000000000002",
    )

    assert dlmf_snapshot.source_system == "nist_dlmf"
    assert dlmf_candidates[0].raw_formula_format == "latex"
    assert dlmf_candidates[0].candidate_status == "raw_imported"


def test_pdg_relationship_rows_validate_after_expression_ids_are_available() -> None:
    bundle = parse_pdg_document(
        {
            "equations": [
                {"id": "eq:base", "latex": "F = m a"},
                {"id": "eq:solved", "latex": "a = F / m"},
            ],
            "inference_edges": [
                {
                    "source": "eq:base",
                    "target": "eq:solved",
                    "rule": "solve for acceleration",
                    "confidence": 0.75,
                }
            ],
        }
    )

    snapshot, candidates = stage_source_rows(
        snapshot_row=bundle.snapshot_row,
        candidate_rows=bundle.candidate_rows(),
        snapshot_id=SNAPSHOT_ID,
    )
    relationship = bundle.relationship_rows(
        expression_id_by_pdg_node_id={
            "eq:base": EXPRESSION_A,
            "eq:solved": EXPRESSION_B,
        }
    )[0]
    staged_relationship = validate_artifact_relationship_row(relationship)

    assert snapshot.source_system == "physics_derivation_graph"
    assert [candidate.source_candidate_id for candidate in candidates] == [
        "eq:base",
        "eq:solved",
    ]
    assert staged_relationship.source_expression_id == EXPRESSION_B
    assert staged_relationship.target_expression_id == EXPRESSION_A
    assert staged_relationship.relationship_kind == "algebraic_rearrangement_of"
    assert staged_relationship.source_kind == "physics_derivation_graph"


def test_symbolic_expression_contract_enforces_wave0_expression_enums() -> None:
    expression = SymbolicExpressionRow.model_validate(
        {
            "artifact_id": ARTIFACT_ID,
            "version_id": VERSION_ID,
            "candidate_id": SNAPSHOT_ID,
            "expression_kind": "equation",
            "sympy_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
            "canonical_expr_hash": "a" * 64,
            "topology_hash": "b" * 64,
            "raw_formula": "F = m a",
            "raw_formula_format": "plain_text",
            "parse_status": "normalized",
            "parse_confidence": 0.9,
            "review_status": "automated_pass",
            "validation_status": "passed",
        }
    )

    assert expression.expression_role == "primary"
    assert expression.to_insert_dict()["canonical_expr_hash"] == "a" * 64

    with pytest.raises(ValidationError, match="expression_kind"):
        SymbolicExpressionRow.model_validate(
            {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
                "expression_kind": "formula",
                "raw_formula": "F = m a",
            }
        )


def test_invalid_contract_values_fail_before_db_insertion() -> None:
    with pytest.raises(ValidationError, match="source_system"):
        validate_snapshot_row(
            {
                "source_system": "arxiv",
                "payload_sha256": "a" * 64,
                "payload": {},
            }
        )

    with pytest.raises(ValidationError, match="candidate_status"):
        validate_candidate_row(
            {
                "snapshot_id": SNAPSHOT_ID,
                "candidate_status": "reviewed",
            }
        )

    with pytest.raises(ValidationError, match="raw_formula_format"):
        validate_candidate_row(
            {
                "snapshot_id": SNAPSHOT_ID,
                "raw_formula_format": "tex",
            }
        )

    with pytest.raises(ValidationError, match="source artifact"):
        ArtifactRelationshipRow.model_validate(
            {
                "target_expression_id": EXPRESSION_A,
                "relationship_kind": "derives_from",
            }
        )


def test_attach_snapshot_id_returns_copies_and_protects_existing_ids() -> None:
    rows = [{"source_candidate_id": "fixture"}]

    attached = attach_snapshot_id(rows, SNAPSHOT_ID)

    assert attached == [
        {
            "source_candidate_id": "fixture",
            "snapshot_id": SNAPSHOT_ID,
        }
    ]
    assert rows == [{"source_candidate_id": "fixture"}]

    with pytest.raises(Wave0ContractError, match="different snapshot_id"):
        attach_snapshot_id(
            [
                {
                    "source_candidate_id": "fixture",
                    "snapshot_id": "00000000-0000-0000-0000-000000000002",
                }
            ],
            SNAPSHOT_ID,
        )


def _binding(value: str, kind: str = "literal") -> dict[str, str]:
    return {"type": kind, "value": value}
