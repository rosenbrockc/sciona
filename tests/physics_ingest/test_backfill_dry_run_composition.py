from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from sciona.physics_ingest.backfill import build_physics_ingest_backfill_report
from sciona.physics_ingest.ids import plan_source_bundle_ids
from sciona.physics_ingest.normalization import normalize_candidate_expression_drafts
from sciona.physics_ingest.pdg_cdg import (
    build_pdg_publication_write_rows,
    build_pdg_relationship_ingest,
)
from sciona.physics_ingest.retrieval import (
    SymbolicRetrievalQuery,
    build_symbolic_retrieval_report,
)
from sciona.physics_ingest.sources.pdg import parse_pdg_document
from sciona.physics_ingest.sources.qudt import (
    build_qudt_snapshot_manifest,
    build_qudt_symbolic_variable_dimension_updates,
)
from sciona.physics_ingest.write_plan import merge_publication_insert_rows


ARTIFACT_ID = "20000000-0000-0000-0000-000000000111"
VERSION_ID = "30000000-0000-0000-0000-000000000111"
EXPR_BASE = "10000000-0000-0000-0000-000000000111"
EXPR_FAILED = "10000000-0000-0000-0000-000000000112"
EXPR_SOLVED = "10000000-0000-0000-0000-000000000113"
VARIABLE_FORCE = "40000000-0000-0000-0000-000000000111"


def test_dry_run_backfill_composes_retrieval_normalization_qudt_and_pdg() -> None:
    retrieval_report = build_symbolic_retrieval_report(
        SymbolicRetrievalQuery(
            topology_hashes=("topo-force",),
            mechanism_tags=("newtonian_mechanics",),
            raw_trust_policy="prefer_reviewed",
        ),
        _retrieval_candidates(),
    )
    source_retrieval_plan = retrieval_report[
        "raw_candidate_external_knowledge_suggestions"
    ]
    assert [item["candidate_key"] for item in source_retrieval_plan] == [
        "expr-force-raw"
    ]

    qudt_manifest = _qudt_manifest()
    pdg_bundle = _pdg_bundle()
    _, planned_bundles = plan_source_bundle_ids(
        [
            _retrieval_source_bundle(source_retrieval_plan),
            qudt_manifest,
            {
                "bundle_key": "fixture-pdg",
                "snapshot_row": pdg_bundle.snapshot_row,
                "candidate_rows": pdg_bundle.candidate_rows(),
            },
        ]
    )

    retrieval_candidates = planned_bundles[0]["candidate_rows"]
    normalization_drafts = normalize_candidate_expression_drafts(
        retrieval_candidates,
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
        require_dimensions=False,
    )
    expression_rows = [
        {
            **draft.to_insert_dict(),
            "expression_id": expression_id,
        }
        for draft, expression_id in zip(
            normalization_drafts,
            (EXPR_BASE, EXPR_FAILED),
            strict=True,
        )
    ]
    assert [row["parse_status"] for row in expression_rows] == [
        "normalized",
        "parse_failed",
    ]

    variable_updates = build_qudt_symbolic_variable_dimension_updates(
        qudt_manifest.records,
        [
            {
                "variable_id": VARIABLE_FORCE,
                "expression_id": EXPR_BASE,
                "symbol_name": "F",
                "source_symbol": "F",
                "variable_role": "output",
                "quantity_kind_uri": "http://qudt.org/vocab/quantitykind/Force",
                "dim_signature": "",
                "dimension_source": "unknown",
                "evidence_json": {"source": "normalization_draft"},
            }
        ],
    )
    assert variable_updates[0]["dim_signature"] == "M1L1T-2"

    pdg_publication_rows = build_pdg_publication_write_rows(
        build_pdg_relationship_ingest(
            pdg_bundle,
            expression_bindings_by_pdg_node_id={
                "eq:base": EXPR_BASE,
                "eq:solved": EXPR_SOLVED,
            },
        )
    )
    pdg_rows = pdg_publication_rows.to_insert_rows()
    pdg_rows["artifact_relationships"][0]["relationship_id"] = (
        "50000000-0000-0000-0000-000000000111"
    )
    report = build_physics_ingest_backfill_report(
        source_bundles=planned_bundles,
        pdg_publication_rows=merge_publication_insert_rows(
            {
                "artifact_symbolic_expressions": expression_rows,
                "artifact_symbolic_variables": variable_updates,
            },
            pdg_rows,
        ),
        normalization_diagnostics=[
            *_normalization_diagnostics(normalization_drafts),
            {
                "stage": "qudt_dimension_resolution",
                "table": "artifact_symbolic_variables",
                "reason": "qudt_dimension_resolved",
                "severity": "info",
                "detail": "F -> M1L1T-2",
            },
        ],
        include_rows=True,
    )

    assert report["input_summary"]["source_bundle_count"] == 3
    assert report["input_summary"]["phase7_coverage_row_count"] == 7
    assert report["phase7_coverage_row_counts"] == {
        "artifact_symbolic_expressions": 2,
        "physics_equation_candidates": 5,
    }
    assert report["phase7_coverage_summary"]["report_version"] == (
        "physics-phase7-coverage-summary.v1"
    )
    assert report["phase7_coverage_summary"]["summary"] == {
        "total_rows": 7,
        "discovered": 7,
        "parsed": 2,
        "dimensioned": 2,
        "reviewed": 0,
        "published": 0,
        "blocked": 1,
        "metrics": {
            "parsed_rate": 0.285714,
            "dimensioned_rate": 0.285714,
            "reviewed_rate": 0.0,
            "published_rate": 0.0,
            "blocked_rate": 0.142857,
            "discovered_to_parsed_loss": 5,
            "parsed_to_dimensioned_loss": 0,
            "dimensioned_to_reviewed_loss": 2,
            "reviewed_to_published_loss": 0,
        },
    }
    encoded_coverage_summary = json.dumps(
        report["phase7_coverage_summary"],
        sort_keys=True,
    )
    assert json.loads(encoded_coverage_summary) == report["phase7_coverage_summary"]
    assert report["table_row_counts"]["physics_equation_candidates"] == 5
    assert report["table_row_counts"]["artifact_symbolic_expressions"] == 2
    assert report["insert_rows_by_table"]["artifact_symbolic_expressions"][1][
        "parse_status"
    ] == "parse_failed"
    assert report["external_diagnostics"]["normalization"][-1] == {
        "stage": "qudt_dimension_resolution",
        "table": "artifact_symbolic_variables",
        "reason": "qudt_dimension_resolved",
        "severity": "info",
        "artifact_key": "",
        "atom_name": "",
        "detail": "F -> M1L1T-2",
    }
    assert {
        row["reason"]
        for row in report["external_diagnostics"]["normalization"]
    } >= {"parse_roundtrip_failed", "qudt_dimension_resolved"}
    assert report["replay_keys"]["artifact_symbolic_expressions"] == [
        f"artifact_symbolic_expressions|expression_id={EXPR_BASE}",
        f"artifact_symbolic_expressions|expression_id={EXPR_FAILED}",
    ]
    assert report["replay_keys"]["artifact_symbolic_variables"] == [
        f"artifact_symbolic_variables|variable_id={VARIABLE_FORCE}"
    ]
    assert report["replay_keys"]["artifact_relationships"] == [
        "artifact_relationships|relationship_id=50000000-0000-0000-0000-000000000111"
    ]
    assert report["diagnostic_summary"]["by_reason"]["qudt_dimension_resolved"] == 1


def _retrieval_candidates() -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": ARTIFACT_ID,
            "version_id": VERSION_ID,
            "expression_id": EXPR_BASE,
            "fqdn": "fixture.force",
            "raw_formula": "F = m * a",
            "topology_hash": "topo-force",
            "dim_signatures": ["M1L1T-2"],
            "mechanism_tags": ["newtonian_mechanics"],
            "review_status": "human_reviewed",
            "validation_status": "passed",
            "candidate_status": "published",
        },
        {
            "artifact_id": ARTIFACT_ID,
            "version_id": VERSION_ID,
            "expression_id": "expr-force-raw",
            "fqdn": "fixture.force.raw_gap",
            "raw_formula": "F =",
            "candidate_status": "raw_imported",
            "mechanism_tags": ["newtonian_mechanics"],
        },
    ]


def _retrieval_source_bundle(
    source_retrieval_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "bundle_key": "fixture-retrieval-plan",
        "snapshot_row": {
            "source_system": "manual",
            "source_version": "retrieval-plan-v1",
            "adapter_name": "fixture.retrieval_plan",
            "payload_sha256": "b" * 64,
            "payload": {"source_retrieval_plan": source_retrieval_plan},
        },
        "candidate_rows": [
            {
                "source_candidate_id": "fixture:eq:force",
                "source_label": "Newton second law",
                "raw_formula": "F = m * a",
                "raw_formula_format": "plain_text",
                "candidate_status": "raw_imported",
                "parse_confidence": 0.5,
                "source_payload": {"retrieval_plan_index": 0},
            },
            {
                "source_candidate_id": "fixture:eq:force:failed",
                "source_label": "Incomplete force candidate",
                "raw_formula": "F =",
                "raw_formula_format": "plain_text",
                "candidate_status": "raw_imported",
                "parse_confidence": 0.1,
                "source_payload": {"source_retrieval_plan": source_retrieval_plan},
            },
        ],
    }


def _qudt_manifest() -> Any:
    return build_qudt_snapshot_manifest(
        [
            {
                "@id": "http://qudt.org/vocab/quantitykind/Force",
                "@type": "qudt:QuantityKind",
                "rdfs:label": "Force",
                "qudt:symbol": "F",
                "qudt:hasDimensionVector": {
                    "@id": (
                        "http://qudt.org/vocab/dimensionvector/"
                        "A0E0L1I0M1H0T-2D0"
                    )
                },
            }
        ],
        source_version="qudt-fixture",
        source_uri="https://qudt.org/fixture",
        retrieved_at=datetime(2026, 4, 30, 18, 0, tzinfo=timezone.utc),
    )


def _pdg_bundle() -> Any:
    return parse_pdg_document(
        {
            "equations": [
                {
                    "id": "eq:base",
                    "label": "Newton second law",
                    "formula": "F = m * a",
                    "formula_format": "plain_text",
                },
                {
                    "id": "eq:solved",
                    "label": "Solve for acceleration",
                    "formula": "a = F / m",
                    "formula_format": "plain_text",
                },
            ],
            "inference_edges": [
                {
                    "id": "edge:solve",
                    "source": "eq:base",
                    "target": "eq:solved",
                    "rule": "solve for acceleration",
                    "confidence": 0.91,
                }
            ],
        },
        source_uri="https://pdg.example/fixture",
        source_version="pdg-fixture",
        retrieved_at="2026-04-30T18:00:00+00:00",
    )


def _normalization_diagnostics(drafts: Any) -> list[dict[str, str]]:
    return [
        {
            "stage": "normalization",
            "table": "artifact_symbolic_expressions",
            "reason": diagnostic.code,
            "severity": diagnostic.severity,
            "detail": diagnostic.message,
        }
        for draft in drafts
        for diagnostic in draft.diagnostics
    ]
