from __future__ import annotations

import json
from typing import Any

from sciona.physics_ingest.backfill import (
    BACKFILL_REPORT_KIND,
    build_physics_ingest_backfill_report,
)


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
        "pdg_table_count": 2,
        "pdg_row_count": 2,
        "review_diagnostic_count": 0,
        "normalization_diagnostic_count": 0,
    }
    assert report["source_family_counts"] == {
        "source_bundles": {"manual": 1},
        "publication_manifests": {"fixture": 1},
        "pdg_publication_rows": {"physics_derivation_graph": 2},
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
