from __future__ import annotations

from io import StringIO
import json
import sys

import pytest

from sciona.physics_ingest.cli import (
    COMPOSED_REPORT_KIND,
    REPORT_KIND,
    build_publication_backfill_dry_run_report_from_payload,
    build_publication_dry_run_report,
    build_publication_dry_run_report_from_payload,
    main,
)
from sciona.physics_ingest.sources import build_physics_source_retrieval_run_plan_dict


ARTIFACT_ID = "20000000-0000-0000-0000-000000000001"
VERSION_ID = "30000000-0000-0000-0000-000000000001"


def test_publication_dry_run_report_is_json_serializable_and_ordered() -> None:
    report = build_publication_dry_run_report(
        source_bundles=[_source_bundle()],
        publication_manifests=[_publication_manifest()],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        table_modes={"artifact_symbolic_expressions": "upsert"},
        include_rows=True,
    )

    assert json.loads(json.dumps(report))["report_kind"] == REPORT_KIND
    assert report["dry_run"] is True
    assert report["ok"] is True
    assert report["diagnostics"] == []
    assert report["id_planning"]["strategy"] == "deterministic"
    assert report["audit_summary"]["insert_row_counts"] == {
        "artifact_symbolic_expressions": 1,
        "artifact_symbolic_variables": 1,
        "physics_equation_candidates": 1,
        "physics_ingest_snapshots": 1,
    }
    assert [
        (batch["table"], batch["mode"], batch["row_count"], batch["dry_run"])
        for batch in report["write_plan"]["batches"]
    ] == [
        ("physics_ingest_snapshots", "insert", 1, True),
        ("physics_equation_candidates", "insert", 1, True),
        ("artifact_symbolic_expressions", "upsert", 1, True),
        ("artifact_symbolic_variables", "insert", 1, True),
    ]

    rows = report["insert_rows_by_table"]
    assert rows["physics_ingest_snapshots"][0]["snapshot_id"]
    assert rows["physics_equation_candidates"][0]["candidate_id"]
    assert rows["physics_equation_candidates"][0]["snapshot_id"] == (
        rows["physics_ingest_snapshots"][0]["snapshot_id"]
    )


def test_publication_dry_run_report_surfaces_validation_errors() -> None:
    manifest = _publication_manifest()
    manifest["artifact_symbolic_variables"] = [
        {
            "artifact_key": "local:fixture.force",
            "atom_name": "force_atom",
            "symbol": "F",
            "role": "unsupported",
        }
    ]

    report = build_publication_dry_run_report(
        source_bundles=[_source_bundle()],
        publication_manifests=[manifest],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
    )

    assert report["ok"] is False
    assert report["audit_summary"]["has_errors"] is True
    assert [
        (row["table"], row["reason"], row["severity"])
        for row in report["diagnostics"]
    ] == [("artifact_symbolic_variables", "validation_error", "error")]
    assert "variable_role" in report["diagnostics"][0]["detail"]
    assert [
        batch["table"] for batch in report["write_plan"]["batches"]
    ] == [
        "physics_ingest_snapshots",
        "physics_equation_candidates",
        "artifact_symbolic_expressions",
    ]


def test_publication_dry_run_payload_rejects_non_sequence_inputs() -> None:
    with pytest.raises(ValueError, match="source_bundles must be a sequence"):
        build_publication_dry_run_report_from_payload(
            {"source_bundles": {"bundle_key": "not-a-list"}},
        )


def test_publication_backfill_payload_composes_source_retrieval_plan() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan_dict(max_jobs=2, limit=5)
    payload = {
        "source_bundles": [_source_bundle()],
        "publication_manifests": [_publication_manifest()],
        "artifact_bindings": {
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        "source_retrieval_run_plan": retrieval_plan,
    }

    report = build_publication_backfill_dry_run_report_from_payload(payload)

    assert report["report_kind"] == COMPOSED_REPORT_KIND
    assert report["dry_run"] is True
    assert report["ok"] is True
    assert report["publication_dry_run_report"]["report_kind"] == REPORT_KIND
    assert report["publication_dry_run_report"]["source_bundle_count"] == 1
    assert report["backfill_report"]["input_summary"][
        "source_retrieval_step_count"
    ] == 2
    assert report["source_retrieval_run_plan"]["filters"]["limit"] == 5
    assert report["source_retrieval_run_plan"]["replay_keys"] == [
        step["replay_key"] for step in retrieval_plan["steps"]
    ]


def test_publication_backfill_payload_accepts_retrieval_plan_alias() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan_dict(max_jobs=1)

    report = build_publication_backfill_dry_run_report_from_payload(
        {
            "source_bundles": [_source_bundle()],
            "publication_manifests": [_publication_manifest()],
            "artifact_bindings": {
                "local:fixture.force": {
                    "artifact_id": ARTIFACT_ID,
                    "version_id": VERSION_ID,
                }
            },
            "retrieval_run_plan": retrieval_plan,
        }
    )

    assert report["source_retrieval_run_plan"]["step_count"] == 1
    assert report["backfill_report"]["source_retrieval_run_plan"] == (
        report["source_retrieval_run_plan"]
    )


def test_publication_backfill_payload_rejects_retrieval_plan_conflict() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan_dict(max_jobs=1)

    with pytest.raises(
        ValueError,
        match="pass only one of source_retrieval_run_plan or retrieval_run_plan",
    ):
        build_publication_backfill_dry_run_report_from_payload(
            {
                "source_retrieval_run_plan": retrieval_plan,
                "retrieval_run_plan": retrieval_plan,
            }
        )


def test_publication_backfill_payload_report_is_json_serializable() -> None:
    retrieval_plan = build_physics_source_retrieval_run_plan_dict(max_jobs=1)
    report = build_publication_backfill_dry_run_report_from_payload(
        {
            "source_bundles": [_source_bundle()],
            "publication_manifests": [_publication_manifest()],
            "artifact_bindings": {
                "local:fixture.force": {
                    "artifact_id": ARTIFACT_ID,
                    "version_id": VERSION_ID,
                }
            },
            "source_retrieval_run_plan": retrieval_plan,
        },
        include_rows=True,
    )

    assert json.loads(json.dumps(report, sort_keys=True)) == report
    assert report["publication_dry_run_report"]["insert_rows_by_table"]
    assert report["backfill_report"]["insert_rows_by_table"]
    assert report["phase7_coverage_row_counts"] == {
        "artifact_symbolic_expressions": 1,
        "physics_equation_candidates": 1,
    }
    assert report["phase7_coverage_summary"] == (
        report["backfill_report"]["phase7_coverage_summary"]
    )
    assert report["phase7_coverage_summary"]["summary"] == {
        "total_rows": 2,
        "discovered": 2,
        "parsed": 1,
        "dimensioned": 0,
        "reviewed": 1,
        "published": 0,
        "blocked": 0,
        "metrics": {
            "parsed_rate": 0.5,
            "dimensioned_rate": 0.0,
            "reviewed_rate": 0.5,
            "published_rate": 0.0,
            "blocked_rate": 0.0,
            "discovered_to_parsed_loss": 1,
            "parsed_to_dimensioned_loss": 1,
            "dimensioned_to_reviewed_loss": 0,
            "reviewed_to_published_loss": 1,
        },
    }


def test_publication_dry_run_main_prints_report_from_json_payload(
    tmp_path,
    monkeypatch,
) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "source_bundles": [_source_bundle()],
                "publication_manifests": [_publication_manifest()],
                "artifact_bindings": {
                    "local:fixture.force": {
                        "artifact_id": ARTIFACT_ID,
                        "version_id": VERSION_ID,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stdout = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    exit_code = main([str(payload_path)])

    assert exit_code == 0
    report = json.loads(stdout.getvalue())
    assert report["report_kind"] == REPORT_KIND
    assert report["dry_run"] is True
    assert report["ok"] is True
    assert "insert_rows_by_table" not in report


def test_publication_dry_run_main_auto_dispatches_backfill_payload(
    tmp_path,
    monkeypatch,
) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "source_bundles": [_source_bundle()],
                "publication_manifests": [_publication_manifest()],
                "artifact_bindings": {
                    "local:fixture.force": {
                        "artifact_id": ARTIFACT_ID,
                        "version_id": VERSION_ID,
                    }
                },
                "source_retrieval_run_plan": (
                    build_physics_source_retrieval_run_plan_dict(max_jobs=1)
                ),
            }
        ),
        encoding="utf-8",
    )
    stdout = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    exit_code = main([str(payload_path)])

    assert exit_code == 0
    report = json.loads(stdout.getvalue())
    assert report["report_kind"] == COMPOSED_REPORT_KIND
    assert report["publication_dry_run_report"]["report_kind"] == REPORT_KIND
    assert report["source_retrieval_run_plan"]["step_count"] == 1


def test_publication_dry_run_main_can_include_rows(tmp_path, monkeypatch) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "source_bundles": [_source_bundle()],
                "publication_manifests": [_publication_manifest()],
                "artifact_bindings": {
                    "local:fixture.force": {
                        "artifact_id": ARTIFACT_ID,
                        "version_id": VERSION_ID,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stdout = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    exit_code = main(["--include-rows", str(payload_path)])

    assert exit_code == 0
    report = json.loads(stdout.getvalue())
    assert report["insert_rows_by_table"]["physics_ingest_snapshots"]


def _source_bundle() -> dict[str, object]:
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


def _publication_manifest() -> dict[str, object]:
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
