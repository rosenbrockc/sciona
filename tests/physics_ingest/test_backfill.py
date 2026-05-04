from __future__ import annotations

import hashlib
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
    audit_replay = report["audit_replay"]
    assert audit_replay["schema_version"] == (
        "physics-ingest-backfill-audit-replay.v1"
    )
    assert len(audit_replay["input_fingerprint_sha256"]) == 64
    assert "insert_rows_by_table" not in audit_replay["input_fingerprint_source"]
    cdg_node_digest = audit_replay["table_batch_digests"]["artifact_cdg_nodes"]
    assert cdg_node_digest["mode"] == "insert"
    assert cdg_node_digest["row_count"] == 1
    assert cdg_node_digest["conflict_keys"] == ["version_id", "node_id"]
    assert len(cdg_node_digest["row_hash_digest"]) == 64
    assert len(cdg_node_digest["conflict_identity_digest"]) == 64
    assert len(cdg_node_digest["batch_digest"]) == 64
    assert cdg_node_digest["missing_conflict_key_row_count"] == 0
    assert cdg_node_digest["duplicate_conflict_identity_count"] == 0
    assert audit_replay["replay_key_rollup"]["artifact_cdg_nodes"]["count"] == 1
    assert audit_replay["diagnostic_digest"]["summary"] == report[
        "diagnostic_summary"
    ]
    assert report["retry_diagnostics"] == []
    assert report["skip_diagnostics"] == []
    assert report["diagnostic_summary"]["by_severity"] == {"info": 6}


def test_backfill_report_includes_json_safe_dashboard_summary() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan_dict(max_jobs=1)
    retrieval_plan["diagnostics"] = [
        {
            "severity": "warning",
            "job_id": retrieval_plan["steps"][0]["job_id"],
            "endpoint_id": retrieval_plan["steps"][0]["endpoint_id"],
            "message": "fixture retrieval metadata warning",
        }
    ]
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

    report = build_physics_ingest_backfill_report(
        source_bundles=[hitran_bundle],
        publication_manifests=[_publication_manifest()],
        pdg_publication_rows=FakePDGRows(),
        retrieval_run_plan=retrieval_plan,
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

    summary = report["dashboard_summary"]

    assert json.loads(json.dumps(summary, sort_keys=True)) == summary
    assert "input_summary" in report
    assert "dry_run_write_plan" in report
    assert "publication_readiness_summary" in report
    assert "phase7_coverage_summary" in report
    assert summary["ok"] is False
    assert summary["dry_run"] is True
    assert summary["input_counts"] == report["input_summary"]
    assert summary["write_plan"] == {
        "batch_count": 5,
        "table_count": 5,
        "row_count": 5,
        "dry_run_batch_count": 5,
        "mode_counts": {"insert": 5},
        "table_row_counts": report["table_row_counts"],
    }
    assert summary["publication_readiness"] == {
        "row_count": 2,
        "table_row_counts": {
            "artifact_symbolic_expressions": 1,
            "physics_equation_candidates": 1,
        },
        "readiness_stage_counts": {
            "automated_pass": 1,
            "raw_or_pending": 1,
        },
        "by_candidate_status": {"raw_imported": 1},
        "by_parse_status": {"normalized": 1},
        "by_review_status": {"automated_pass": 1},
        "by_validation_status": {"passed": 1},
    }
    assert summary["phase7_coverage"] == {
        "row_count": 2,
        "table_row_counts": {
            "artifact_symbolic_expressions": 1,
            "physics_equation_candidates": 1,
        },
        "summary": report["phase7_coverage_summary"]["summary"],
    }
    assert summary["source_retrieval"] == {
        "step_count": 1,
        "diagnostic_count": 1,
    }
    assert summary["diagnostics"] == {
        "by_severity": report["diagnostic_summary"]["by_severity"],
        "by_reason": report["diagnostic_summary"]["by_reason"],
        "retry_count": 1,
        "skip_count": 1,
    }
    assert summary["source_family_counts"] == {
        "combined": {
            "fixture": 1,
            "hitran": 1,
            "physics_derivation_graph": 1,
        }
    }
    assert summary["data_artifacts"] == {"seed_count": 1}


def test_backfill_report_omits_audit_artifact_manifests_by_default() -> None:
    report = build_physics_ingest_backfill_report(
        source_bundles=[_source_bundle()],
        publication_manifests=[_publication_manifest()],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
    )

    assert "audit_artifact_manifests" not in report
    assert "audit_artifact_manifest_summary" not in report


def test_backfill_report_includes_opt_in_audit_artifact_manifests() -> None:
    report = build_physics_ingest_backfill_report(
        source_bundles=[_source_bundle()],
        publication_manifests=[_publication_manifest()],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        include_audit_artifact_manifests=True,
    )

    manifests = report["audit_artifact_manifests"]
    summary = report["audit_artifact_manifest_summary"]

    assert json.loads(json.dumps(manifests, sort_keys=True)) == manifests
    assert [row["artifact_name"] for row in manifests] == [
        "dashboard_summary",
        "audit_replay",
        "phase7_coverage_summary",
    ]
    assert all(row["content_type"] == "application/json" for row in manifests)
    assert all(
        row["source_report_kind"] == BACKFILL_REPORT_KIND
        for row in manifests
    )
    assert all(row["name"] == row["artifact_name"] for row in manifests)
    assert all("payload" not in row for row in manifests)
    assert all("payload_ref" not in row for row in manifests)
    assert all(len(row["payload_sha256"]) == 64 for row in manifests)

    dashboard_manifest = manifests[0]
    expected_dashboard_sha = _stable_json_sha256(report["dashboard_summary"])
    assert dashboard_manifest["payload_sha256"] == expected_dashboard_sha
    assert dashboard_manifest["artifact_key"] == (
        f"{BACKFILL_REPORT_KIND}/dashboard_summary/{expected_dashboard_sha}"
    )
    assert dashboard_manifest["source_replay_key"] == (
        f"{BACKFILL_REPORT_KIND}|dashboard_summary|"
        f"payload_sha256={expected_dashboard_sha}"
    )
    assert dashboard_manifest["row_count"] == report["dashboard_summary"][
        "write_plan"
    ]["row_count"]
    phase7_manifest = manifests[2]
    assert phase7_manifest["row_count"] == report["phase7_coverage_summary"][
        "summary"
    ]["total_rows"]

    assert summary["artifact_count"] == 3
    assert summary["total_payload_count"] == sum(
        row["payload_count"] for row in manifests
    )
    assert summary["content_type_counts"] == {"application/json": 3}
    assert len(summary["payload_sha256_digest"]) == 64
    assert summary["artifact_keys"] == [row["artifact_key"] for row in manifests]


def test_backfill_audit_artifact_manifest_hashes_are_deterministic() -> None:
    kwargs = {
        "source_bundles": [_source_bundle()],
        "publication_manifests": [_publication_manifest()],
        "artifact_bindings": {
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        "include_audit_artifact_manifests": True,
    }

    report = build_physics_ingest_backfill_report(**kwargs)
    repeated_report = build_physics_ingest_backfill_report(**kwargs)
    changed_report = build_physics_ingest_backfill_report(
        **{
            **kwargs,
            "review_diagnostics": [
                {
                    "reason": "fixture_warning",
                    "severity": "warning",
                }
            ],
        }
    )

    assert report["audit_artifact_manifests"] == repeated_report[
        "audit_artifact_manifests"
    ]
    assert report["audit_artifact_manifest_summary"] == repeated_report[
        "audit_artifact_manifest_summary"
    ]
    assert report["audit_artifact_manifest_summary"][
        "payload_sha256_digest"
    ] != changed_report["audit_artifact_manifest_summary"]["payload_sha256_digest"]


def test_backfill_audit_artifact_manifests_can_embed_payload_with_rows() -> None:
    report = build_physics_ingest_backfill_report(
        source_bundles=[_source_bundle()],
        include_rows=True,
        include_audit_artifact_manifests=True,
    )

    manifests = report["audit_artifact_manifests"]

    assert "insert_rows_by_table" in report
    assert all("payload" in row for row in manifests)
    assert manifests[0]["payload"] == report["dashboard_summary"]
    assert json.loads(json.dumps(report, sort_keys=True)) == report


def test_backfill_audit_artifact_manifests_follow_preflight_flags() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan(max_jobs=2, limit=5)
    review_rows = {
        "artifact_symbolic_expressions": [
            {
                "expression_id": "expression-review-2",
                "review_status": "human_reviewed",
            }
        ]
    }

    source_report = build_physics_ingest_backfill_report(
        source_retrieval_run_plan=retrieval_plan,
        review_status_rows=review_rows,
        include_source_request_envelopes=True,
        include_audit_artifact_manifests=True,
    )
    write_report = build_physics_ingest_backfill_report(
        source_retrieval_run_plan=retrieval_plan,
        review_status_rows=review_rows,
        include_publication_write_preflight=True,
        include_audit_artifact_manifests=True,
    )
    boundary_report = build_physics_ingest_backfill_report(
        source_retrieval_run_plan=retrieval_plan,
        review_status_rows=review_rows,
        include_execution_boundary_preflight=True,
        include_audit_artifact_manifests=True,
    )

    assert _manifest_names(source_report) == [
        "dashboard_summary",
        "audit_replay",
        "phase7_coverage_summary",
        "source_request_envelope_preflight",
    ]
    assert _manifest_names(write_report) == [
        "dashboard_summary",
        "audit_replay",
        "phase7_coverage_summary",
        "publication_storage_write_preflight",
    ]
    assert _manifest_names(boundary_report) == [
        "dashboard_summary",
        "audit_replay",
        "phase7_coverage_summary",
        "source_request_envelope_preflight",
        "publication_storage_write_preflight",
    ]

    source_manifest = _manifest_by_name(
        boundary_report,
        "source_request_envelope_preflight",
    )
    write_manifest = _manifest_by_name(
        boundary_report,
        "publication_storage_write_preflight",
    )
    assert source_manifest["row_count"] == boundary_report[
        "source_request_envelope_preflight"
    ]["step_count"]
    assert write_manifest["row_count"] == boundary_report[
        "publication_storage_write_preflight"
    ]["total_row_count"]


def test_backfill_dashboard_summary_omits_phase7_when_not_requested() -> None:
    report = build_physics_ingest_backfill_report(
        source_bundles=[_source_bundle()],
        include_phase7_coverage_summary=False,
    )

    assert "phase7_coverage_row_counts" not in report
    assert "phase7_coverage_summary" not in report
    assert "phase7_coverage" not in report["dashboard_summary"]
    assert report["dashboard_summary"]["input_counts"][
        "phase7_coverage_row_count"
    ] == 1

    manifest_report = build_physics_ingest_backfill_report(
        source_bundles=[_source_bundle()],
        include_phase7_coverage_summary=False,
        include_audit_artifact_manifests=True,
    )
    assert _manifest_names(manifest_report) == [
        "dashboard_summary",
        "audit_replay",
    ]


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


def test_backfill_report_includes_publication_readiness_rollups() -> None:
    report = build_physics_ingest_backfill_report(
        review_publication_rows={
            "physics_equation_candidates": [
                {
                    "candidate_id": "candidate-ready-1",
                    "candidate_status": "source_verified",
                },
                {
                    "candidate_id": "candidate-blocked-1",
                    "candidate_status": "blocked",
                },
            ],
            "artifact_symbolic_expressions": [
                {
                    "expression_id": "expression-ready-1",
                    "parse_status": "normalized",
                    "review_status": "human_reviewed",
                    "validation_status": "passed",
                },
                {
                    "expression_id": "expression-blocked-1",
                    "parse_status": "parse_failed",
                    "review_status": "blocked",
                    "validation_status": "failed",
                },
                {
                    "expression_id": "expression-needs-human-1",
                    "review_status": "needs_human",
                },
            ],
        },
    )

    readiness = report["publication_readiness_summary"]

    assert json.loads(json.dumps(readiness, sort_keys=True)) == readiness
    assert readiness["report_version"] == (
        "physics-ingest-publication-readiness-summary.v1"
    )
    assert readiness["row_count"] == 5
    assert readiness["table_row_counts"] == {
        "artifact_symbolic_expressions": 3,
        "physics_equation_candidates": 2,
    }
    assert readiness["by_candidate_status"] == {
        "blocked": 1,
        "source_verified": 1,
    }
    assert readiness["by_parse_status"] == {
        "normalized": 1,
        "parse_failed": 1,
    }
    assert readiness["by_review_status"] == {
        "blocked": 1,
        "human_reviewed": 1,
        "needs_human": 1,
    }
    assert readiness["by_validation_status"] == {
        "failed": 1,
        "passed": 1,
    }
    assert readiness["readiness_stage_counts"] == {
        "blocked": 2,
        "needs_human_review": 1,
        "parsed": 1,
        "publishable_candidate": 1,
    }
    assert readiness["by_table"]["artifact_symbolic_expressions"][
        "readiness_stage_counts"
    ] == {
        "blocked": 1,
        "needs_human_review": 1,
        "publishable_candidate": 1,
    }


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
    assert report["audit_replay"]["source_retrieval_replay"][
        "replay_key_count"
    ] == 2
    assert len(
        report["audit_replay"]["source_retrieval_replay"]["replay_key_digest"]
    ) == 64
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


def test_backfill_report_can_include_execution_boundary_preflight_sections() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan(max_jobs=3, limit=5)

    report = build_physics_ingest_backfill_report(
        source_retrieval_run_plan=retrieval_plan,
        review_status_rows={
            "artifact_symbolic_expressions": [
                {
                    "expression_id": "expression-review-2",
                    "review_status": "human_reviewed",
                }
            ]
        },
        include_execution_boundary_preflight=True,
    )
    repeated_report = build_physics_ingest_backfill_report(
        source_retrieval_run_plan=retrieval_plan,
        review_status_rows={
            "artifact_symbolic_expressions": [
                {
                    "expression_id": "expression-review-2",
                    "review_status": "human_reviewed",
                }
            ]
        },
        include_execution_boundary_preflight=True,
    )

    source_preflight = report["source_request_envelope_preflight"]
    assert json.loads(json.dumps(source_preflight, sort_keys=True)) == source_preflight
    assert source_preflight == repeated_report["source_request_envelope_preflight"]
    assert source_preflight["report_version"] == (
        "physics-ingest-source-request-envelope-preflight.v1"
    )
    assert source_preflight["step_count"] == 3
    assert source_preflight["execution_expectation_counts"] == {
        "manual": 1,
        "network": 2,
        "blocked": 0,
    }
    assert source_preflight["manual_retrieval_expected_count"] == 1
    assert source_preflight["network_retrieval_expected_count"] == 2
    assert source_preflight["blocked_retrieval_expected_count"] == 0
    assert source_preflight["readiness_summary"]["by_status"] == {
        "executable": 2,
        "manual": 1,
    }
    assert [
        row["execution_expectation"]
        for row in source_preflight["request_envelopes"]
    ] == ["manual", "network", "network"]
    assert source_preflight["request_envelopes"][0]["request_envelope"][
        "execution"
    ] == {
        "mode": "manual",
        "dry_run": True,
        "io_performed": False,
        "network_required": False,
        "network_io_allowed": False,
        "manual_source": True,
    }
    assert source_preflight["request_envelopes"][1]["request_envelope"]["paging"][
        "limit"
    ] == 5

    write_preflight = report["publication_storage_write_preflight"]
    assert json.loads(json.dumps(write_preflight, sort_keys=True)) == write_preflight
    assert write_preflight == repeated_report["publication_storage_write_preflight"]
    assert write_preflight == {
        "report_version": "physics-ingest-publication-storage-write-preflight.v1",
        "dry_run": True,
        "table_count": 1,
        "total_row_count": 1,
        "mode_counts": {"upsert": 1},
        "tables": [
            {
                "table": "artifact_symbolic_expressions",
                "mode": "upsert",
                "row_count": 1,
                "conflict_keys": ["expression_id"],
                "missing_conflict_metadata": False,
            }
        ],
        "missing_conflict_metadata_for_upserts": [],
        "adapter_capabilities": {
            "imports_supabase": False,
            "requires_injected_client": True,
            "writes_during_preflight": False,
            "supports_insert": True,
            "supports_upsert": True,
            "supports_upsert_on_conflict": True,
        },
    }
    assert json.loads(json.dumps(report, sort_keys=True)) == report


def _manifest_names(report: dict[str, Any]) -> list[str]:
    return [
        str(row["artifact_name"])
        for row in report.get("audit_artifact_manifests", ())
    ]


def _manifest_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    for row in report["audit_artifact_manifests"]:
        if row["artifact_name"] == name:
            return row
    raise AssertionError(f"missing artifact manifest {name}")


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
