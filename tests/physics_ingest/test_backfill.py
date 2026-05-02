from __future__ import annotations

import json
from typing import Any

from sciona.physics_ingest.backfill import (
    BACKFILL_REPORT_KIND,
    build_physics_ingest_backfill_report,
)
from sciona.physics_ingest.normalization import normalize_candidate_expression_draft
from sciona.physics_ingest.review import (
    assess_publishability,
    build_review_publication_status_rows,
)
from sciona.physics_ingest.sources import (
    build_physics_source_retrieval_run_plan,
    build_physics_source_retrieval_run_plan_dict,
)
from sciona.physics_ingest.sources.hitran import build_hitran_wave0_bundle
from sciona.physics_ingest.sources.opb import build_opb_wave0_bundle
from sciona.physics_ingest.sources.theoria import build_theoria_wave0_bundle


ARTIFACT_ID = "20000000-0000-0000-0000-000000000001"
VERSION_ID = "30000000-0000-0000-0000-000000000001"


def test_backfill_report_composes_pipeline_with_pdg_rows_and_replay_keys() -> None:
    report = build_physics_ingest_backfill_report(
        source_bundles=[_source_bundle()],
        publication_manifests=[_publication_manifest()],
        pdg_publication_rows={
            "artifact_relationships": [
                {
                    "relationship_id": "rel-1",
                    "source_expression_id": "expr-solved",
                    "target_expression_id": "expr-base",
                    "source_kind": "physics_derivation_graph",
                }
            ],
            "artifact_cdg_nodes": [
                {
                    "version_id": VERSION_ID,
                    "node_id": "pdg_step_1",
                    "operation_kind": "solve",
                    "source_system": "physics_derivation_graph",
                }
            ],
        },
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        table_modes={"artifact_symbolic_expressions": "upsert"},
    )

    assert json.loads(json.dumps(report))["report_kind"] == BACKFILL_REPORT_KIND
    assert report["ok"] is True
    assert report["input_summary"] == {
        "source_bundle_count": 1,
        "publication_manifest_count": 1,
        "data_artifact_seed_count": 0,
        "normalized_draft_count": 0,
        "normalized_draft_table_count": 0,
        "normalized_draft_row_count": 0,
        "pdg_table_count": 2,
        "pdg_row_count": 2,
        "review_publication_table_count": 0,
        "review_publication_row_count": 0,
        "phase7_coverage_row_count": 2,
        "review_diagnostic_count": 0,
        "normalization_diagnostic_count": 0,
        "source_retrieval_step_count": 0,
        "source_retrieval_diagnostic_count": 0,
    }
    assert report["source_family_counts"] == {
        "source_bundles": {"manual": 1},
        "publication_manifests": {"fixture": 1},
        "normalized_drafts": {},
        "pdg_publication_rows": {"physics_derivation_graph": 2},
        "review_publication_rows": {},
        "combined": {
            "fixture": 1,
            "manual": 1,
            "physics_derivation_graph": 2,
        },
    }
    assert report["table_row_counts"] == {
        "physics_ingest_snapshots": 1,
        "physics_equation_candidates": 1,
        "artifact_symbolic_expressions": 1,
        "artifact_symbolic_variables": 1,
        "artifact_relationships": 1,
        "artifact_cdg_nodes": 1,
    }
    assert [
        (batch["table"], batch["mode"], batch["row_count"], batch["dry_run"])
        for batch in report["dry_run_write_plan"]["batches"]
    ] == [
        ("physics_ingest_snapshots", "insert", 1, True),
        ("physics_equation_candidates", "insert", 1, True),
        ("artifact_symbolic_expressions", "upsert", 1, True),
        ("artifact_symbolic_variables", "insert", 1, True),
        ("artifact_relationships", "insert", 1, True),
        ("artifact_cdg_nodes", "insert", 1, True),
    ]
    assert report["replay_keys"]["artifact_relationships"] == [
        "artifact_relationships|relationship_id=rel-1"
    ]
    assert report["replay_keys"]["artifact_cdg_nodes"] == [
        f"artifact_cdg_nodes|version_id={VERSION_ID}|node_id=pdg_step_1"
    ]
    assert report["retry_diagnostics"] == []
    assert report["skip_diagnostics"] == []
    assert report["diagnostic_summary"]["by_severity"] == {"info": 6}


def test_backfill_report_groups_review_normalization_and_pdg_skip_diagnostics() -> None:
    report = build_physics_ingest_backfill_report(
        publication_manifests=[_publication_manifest()],
        pdg_publication_rows=FakePDGRows(),
        review_diagnostics=[
            {
                "table": "artifact_symbolic_expressions",
                "reason": "needs_human_review",
                "severity": "skipped",
                "artifact_key": "local:fixture.force",
            }
        ],
        normalization_diagnostics=[
            {
                "table": "artifact_symbolic_variables",
                "reason": "unit_alias_unresolved",
                "severity": "error",
                "detail": "kg*m/s^2",
            }
        ],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
    )

    assert report["ok"] is False
    assert [
        (row["stage"], row["table"], row["reason"])
        for row in report["skip_diagnostics"]
    ] == [
        (
            "review",
            "artifact_symbolic_expressions",
            "needs_human_review",
        ),
        (
            "pdg_cdg_publication",
            "artifact_cdg_bindings",
            "missing_cdg_binding_artifact_metadata",
        ),
    ]
    assert [
        (row["stage"], row["table"], row["reason"], row["detail"])
        for row in report["retry_diagnostics"]
    ] == [
        (
            "normalization",
            "artifact_symbolic_variables",
            "unit_alias_unresolved",
            "kg*m/s^2",
        )
    ]
    assert report["diagnostic_summary"]["by_reason"]["dry_run"] == 3


def test_backfill_report_accepts_normalized_drafts_and_review_publication_rows() -> None:
    draft = normalize_candidate_expression_draft(
        {
            "source_candidate_id": "fixture:eq:normalized",
            "raw_formula": "F = m a",
            "raw_formula_format": "plain_text",
            "variables": {
                "F": {"role": "output", "dim_signature": "M1L1T-2"},
                "m": {"role": "input", "dim_signature": "M1"},
                "a": {"role": "input", "dim_signature": "L1T-2"},
            },
        },
        artifact_id=ARTIFACT_ID,
        version_id=VERSION_ID,
    )
    review_rows = build_review_publication_status_rows(
        assess_publishability(
            candidate={
                "raw_formula": "F = m a",
                "candidate_status": "source_verified",
                "parse_confidence": 0.95,
            },
            expression={
                "raw_formula": "F = m a",
                "sympy_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "parse_status": "normalized",
                "parse_confidence": 0.95,
                "canonical_expr_hash": "a" * 64,
                "topology_hash": "b" * 64,
                "dimensional_hash": "c" * 64,
                "review_status": "automated_pass",
                "validation_status": "passed",
                "evidence_json": {
                    "parse_roundtrip": {"status": "passed"},
                    "dimensional_analysis": {"status": "passed"},
                    "numpy_runtime": {
                        "no_sympy_runtime": True,
                        "runtime_imports": ["numpy"],
                        "tests_passed": True,
                        "source": "import numpy as np\n",
                    },
                },
            },
            variables=[
                {"symbol_name": "F", "dim_signature": "M1L1T-2"},
                {"symbol_name": "m", "dim_signature": "M1"},
                {"symbol_name": "a", "dim_signature": "L1T-2"},
            ],
            validity_bounds=[],
        ),
        candidate={"candidate_id": "candidate-review-1"},
        expression={"expression_id": "expression-review-1"},
    )

    report = build_physics_ingest_backfill_report(
        normalized_drafts=[draft],
        review_publication_rows=review_rows,
        include_rows=True,
    )

    assert report["ok"] is True
    assert report["input_summary"]["normalized_draft_count"] == 1
    assert report["input_summary"]["review_publication_row_count"] == 2
    assert report["table_row_counts"] == {
        "physics_equation_candidates": 1,
        "artifact_symbolic_expressions": 2,
        "artifact_symbolic_variables": 3,
    }
    assert report["external_diagnostics"]["normalized_draft_publication"] == []
    assert report["external_diagnostics"]["review_publication"] == []
    assert [
        (batch["table"], batch["mode"], batch["row_count"])
        for batch in report["dry_run_write_plan"]["batches"]
    ] == [
        ("physics_equation_candidates", "upsert", 1),
        ("artifact_symbolic_expressions", "upsert", 2),
        ("artifact_symbolic_variables", "insert", 3),
    ]
    expression_rows = report["insert_rows_by_table"]["artifact_symbolic_expressions"]
    assert expression_rows[0]["source_expression_id"] == "fixture:eq:normalized"
    assert expression_rows[1] == {
        "expression_id": "expression-review-1",
        "review_status": "needs_human",
    }
    assert report["replay_keys"]["physics_equation_candidates"] == [
        "physics_equation_candidates|candidate_id=candidate-review-1"
    ]


def test_backfill_report_accepts_review_status_row_alias_and_mapping_rows() -> None:
    report = build_physics_ingest_backfill_report(
        review_status_rows={
            "artifact_symbolic_expressions": [
                {
                    "expression_id": "expression-review-2",
                    "review_status": "human_reviewed",
                }
            ]
        },
        include_rows=True,
    )

    assert report["input_summary"]["review_publication_row_count"] == 1
    assert report["dry_run_write_plan"]["batches"][0]["mode"] == "upsert"
    assert report["insert_rows_by_table"]["artifact_symbolic_expressions"] == [
        {
            "expression_id": "expression-review-2",
            "review_status": "human_reviewed",
        }
    ]


def test_backfill_report_groups_adapter_diagnostics() -> None:
    report = build_physics_ingest_backfill_report(
        normalized_drafts=[{"raw_formula": "x = y"}],
        review_publication_rows=FakeReviewRows(),
    )

    assert report["ok"] is False
    assert report["external_diagnostics"]["normalized_draft_publication"][0][
        "stage"
    ] == "normalized_draft_publication"
    assert report["external_diagnostics"]["review_publication"] == [
        {
            "stage": "review_publication",
            "table": "artifact_symbolic_expressions",
            "reason": "missing_expression_id",
            "severity": "skipped",
            "artifact_key": "",
            "atom_name": "",
            "detail": "review_status patch requires expression_id",
        }
    ]
    assert {
        (row["stage"], row["reason"])
        for row in report["retry_diagnostics"]
    } == {("normalized_draft_publication", "validation_error")}
    assert {
        (row["stage"], row["reason"])
        for row in report["skip_diagnostics"]
    } == {("review_publication", "missing_expression_id")}


def test_backfill_report_retains_source_data_artifact_seed_summaries() -> None:
    hitran_bundle = build_hitran_wave0_bundle(
        [
            {
                "source_id": "HITRAN:H2O:0001",
                "molecule": "H2O",
                "transition": "1-2",
                "nu": "1594.7",
            }
        ],
        source_version="fixture-hitran",
    )
    opb_bundle = build_opb_wave0_bundle(
        [
            {
                "source_id": "problem-1",
                "label": "Pendulum fixture",
                "formula": "T = 2*pi*sqrt(L/g)",
                "data": {"split": "fixture"},
            }
        ],
        source_version="fixture-opb",
        source_uri="https://example.test/opb",
    )
    theoria_bundle = build_theoria_wave0_bundle(
        [
            {
                "source_id": "theory-1",
                "label": "TheorIA fixture",
                "formula": "E = m*c**2",
                "evaluation": {"metric": "symbolic"},
            }
        ],
        source_version="fixture-theoria",
        source_uri="https://example.test/theoria",
    )

    report = build_physics_ingest_backfill_report(
        source_bundles=[hitran_bundle, opb_bundle, theoria_bundle],
    )

    assert report["ok"] is True
    assert report["input_summary"]["source_bundle_count"] == 3
    assert report["input_summary"]["data_artifact_seed_count"] == 3
    assert report["dry_run_write_plan"]["data_artifact_seed_count"] == 3
    assert report["table_row_counts"] == {
        "physics_ingest_snapshots": 3,
        "physics_equation_candidates": 3,
    }
    assert [
        (seed["source_system"], seed["fqdn"], seed["artifact_kind"])
        for seed in report["data_artifact_seeds"]
    ] == [
        ("hitran", "hitran.line.HITRAN.H2O.0001", "data_artifact"),
        ("opb", "opb.record.problem-1", "data_artifact"),
        ("theoria", "theoria.record.theory-1", "data_artifact"),
    ]


def test_backfill_report_includes_retrieval_run_plan_without_insert_rows() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan(max_jobs=2, limit=5)

    report = build_physics_ingest_backfill_report(
        source_retrieval_run_plan=retrieval_plan,
        include_rows=True,
    )

    assert report["ok"] is True
    assert report["input_summary"]["source_retrieval_step_count"] == 2
    assert report["input_summary"]["source_retrieval_diagnostic_count"] == 0
    assert report["table_row_counts"] == {}
    assert report["insert_rows_by_table"] == {}
    assert report["external_diagnostics"]["source_retrieval"] == []

    retrieval_report = report["source_retrieval_run_plan"]
    assert retrieval_report["manifest_version"] == retrieval_plan.manifest_version
    assert retrieval_report["dry_run"] is True
    assert retrieval_report["step_count"] == 2
    assert retrieval_report["diagnostic_count"] == 0
    assert retrieval_report["filters"]["limit"] == 5
    assert retrieval_report["replay_keys"] == [
        step.replay_key for step in retrieval_plan.steps
    ]
    assert [
        (step["step_index"], step["job_id"], step["replay_key"])
        for step in retrieval_report["steps"]
    ] == [
        (step.step_index, step.job_id, step.replay_key)
        for step in retrieval_plan.steps
    ]
    assert json.loads(json.dumps(report, sort_keys=True)) == report


def test_backfill_report_accepts_retrieval_run_plan_dict_and_diagnostics() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan_dict(max_jobs=1)
    retrieval_plan["diagnostics"] = [
        {
            "severity": "warning",
            "job_id": retrieval_plan["steps"][0]["job_id"],
            "endpoint_id": retrieval_plan["steps"][0]["endpoint_id"],
            "message": "fixture retrieval metadata warning",
        }
    ]

    report = build_physics_ingest_backfill_report(
        retrieval_run_plan=retrieval_plan,
    )

    assert report["input_summary"]["source_retrieval_step_count"] == 1
    assert report["input_summary"]["source_retrieval_diagnostic_count"] == 1
    assert report["source_retrieval_run_plan"]["diagnostic_count"] == 1
    assert report["external_diagnostics"]["source_retrieval"] == [
        {
            "stage": "source_retrieval",
            "table": "",
            "reason": "retrieval_run_plan_diagnostic",
            "severity": "warning",
            "artifact_key": "",
            "atom_name": "",
            "detail": "fixture retrieval metadata warning",
            "job_id": retrieval_plan["steps"][0]["job_id"],
            "endpoint_id": retrieval_plan["steps"][0]["endpoint_id"],
        }
    ]
    assert report["diagnostic_summary"]["by_severity"]["warning"] == 1


def _source_bundle() -> dict[str, Any]:
    return {
        "bundle_key": "fixture-bundle",
        "snapshot_row": {
            "source_system": "manual",
            "source_version": "fixture-v1",
            "adapter_name": "fixture.adapter",
            "payload_sha256": "a" * 64,
            "payload": {"record_count": 1},
        },
        "candidate_rows": [
            {
                "source_candidate_id": "fixture:eq:force",
                "source_label": "Newton second law",
                "raw_formula": "F = m a",
                "raw_formula_format": "plain_text",
                "candidate_status": "raw_imported",
                "parse_confidence": 0.5,
                "source_payload": {"fixture": True},
            }
        ],
    }


def _publication_manifest() -> dict[str, Any]:
    return {
        "provider": "fixture",
        "artifact_symbolic_expressions": [
            {
                "artifact_key": "local:fixture.force",
                "local_artifact_key": "local:fixture.force",
                "atom_name": "force_atom",
                "registry_name": "force_atom",
                "expression_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "expression_text": "Eq(F, a*m)",
            }
        ],
        "artifact_symbolic_variables": [
            {
                "artifact_key": "local:fixture.force",
                "atom_name": "force_atom",
                "symbol": "F",
                "role": "output",
            }
        ],
        "artifact_validity_bounds": [],
    }


class FakePDGRows:
    diagnostics = (
        {
            "table": "artifact_cdg_bindings",
            "reason": "missing_cdg_binding_artifact_metadata",
            "severity": "skipped",
        },
    )

    def to_insert_rows(self) -> dict[str, list[dict[str, str]]]:
        return {"artifact_relationships": [{"relationship_id": "rel-1"}]}


class FakeReviewRows:
    diagnostics = (
        {
            "table": "artifact_symbolic_expressions",
            "reason": "missing_expression_id",
            "severity": "skipped",
            "detail": "review_status patch requires expression_id",
        },
    )

    def to_upsert_rows(self) -> dict[str, list[dict[str, str]]]:
        return {}
