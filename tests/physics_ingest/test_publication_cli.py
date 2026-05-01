from __future__ import annotations

import json

import pytest

from sciona.physics_ingest.cli import (
    REPORT_KIND,
    build_publication_dry_run_report,
    build_publication_dry_run_report_from_payload,
)


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
