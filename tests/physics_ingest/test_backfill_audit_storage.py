from __future__ import annotations

import hashlib
import json
from typing import Any

from sciona.physics_ingest.audit_artifacts import (
    BACKFILL_AUDIT_ARTIFACTS_TABLE,
    build_backfill_audit_artifact_write_plan_rows,
)
from sciona.physics_ingest.backfill import build_physics_ingest_backfill_report


def test_backfill_audit_artifact_rows_and_write_plan_from_report() -> None:
    report = build_physics_ingest_backfill_report(
        include_rows=True,
        include_audit_artifact_manifests=True,
    )

    result = build_backfill_audit_artifact_write_plan_rows(
        report,
        include_write_plan=True,
        table_modes={BACKFILL_AUDIT_ARTIFACTS_TABLE: "upsert"},
    )

    assert result.diagnostics == ()
    assert result.write_plan is not None
    rows = result.to_insert_rows()[BACKFILL_AUDIT_ARTIFACTS_TABLE]
    manifests_by_key = {
        row["artifact_key"]: row for row in report["audit_artifact_manifests"]
    }

    assert len(rows) == len(report["audit_artifact_manifests"])
    for row in rows:
        manifest = manifests_by_key[row["artifact_key"]]
        assert row["name"] == manifest["name"]
        assert row["source_section"] == manifest["source_section"]
        assert row["payload_sha256"] == manifest["payload_sha256"]
        assert row["content_type"] == manifest["content_type"]
        assert row["payload"] == manifest["payload"]

    batch = result.write_plan.batches_by_table()[BACKFILL_AUDIT_ARTIFACTS_TABLE]
    assert batch.conflict_keys == ("artifact_key",)
    assert result.write_plan.mode_for(BACKFILL_AUDIT_ARTIFACTS_TABLE) == "upsert"
    assert result.summary["table"] == BACKFILL_AUDIT_ARTIFACTS_TABLE
    assert result.summary["row_count"] == len(rows)
    assert result.summary["has_payload_count"] == len(rows)
    assert result.summary["write_plan_table_order"] == [
        BACKFILL_AUDIT_ARTIFACTS_TABLE
    ]


def test_audit_artifact_storage_missing_manifests_diagnostic() -> None:
    result = build_backfill_audit_artifact_write_plan_rows(
        {"report_kind": "physics_ingest_bulk_backfill_plan"},
        include_write_plan=True,
    )

    assert result.to_insert_rows() == {}
    assert result.write_plan is not None
    assert result.write_plan.audit_summary.total_row_count == 0
    assert [row["reason"] for row in result.diagnostics] == [
        "missing_audit_artifact_manifests"
    ]
    assert result.summary["diagnostics_by_reason"] == {
        "missing_audit_artifact_manifests": 1
    }


def test_audit_artifact_storage_invalid_manifests_are_diagnostics() -> None:
    valid = _manifest_row("dashboard_summary")

    result = build_backfill_audit_artifact_write_plan_rows(
        [
            valid,
            "not-a-row",
            {"artifact_key": "missing-fields"},
            {
                "artifact_key": "invalid-sha",
                "name": "bad",
                "source_section": "audit_replay",
                "payload_sha256": "not-a-sha",
                "content_type": "application/json",
            },
        ],
        include_write_plan=True,
    )

    rows = result.to_insert_rows()[BACKFILL_AUDIT_ARTIFACTS_TABLE]
    assert rows == [valid]
    assert {
        row["reason"] for row in result.diagnostics
    } >= {
        "invalid_manifest_row_shape",
        "missing_required_manifest_field",
        "invalid_payload_sha256",
    }
    assert result.summary["row_count"] == 1
    assert result.summary["diagnostic_count"] == len(result.diagnostics)
    assert result.write_plan is not None
    assert result.write_plan.audit_summary.total_row_count == 1


def test_audit_artifact_storage_summary_is_deterministic_and_json_safe() -> None:
    dashboard = _manifest_row("dashboard_summary", payload={"b": 2, "a": 1})
    audit_replay = _manifest_row("audit_replay", payload={"rows": [1, 2]})

    result = build_backfill_audit_artifact_write_plan_rows(
        [audit_replay, dashboard],
        include_write_plan=True,
    )
    repeated = build_backfill_audit_artifact_write_plan_rows(
        [dashboard, audit_replay],
        include_write_plan=True,
    )

    assert result.to_dict() == repeated.to_dict()
    assert len(result.summary["payload_sha256_digest"]) == 64
    assert len(result.summary["row_digest_sha256"]) == 64
    assert json.loads(json.dumps(result.to_dict(), sort_keys=True)) == result.to_dict()


def _manifest_row(
    section: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(payload or {"section": section})
    payload_sha256 = _stable_json_sha256(payload)
    return {
        "artifact_key": (
            f"physics_ingest_bulk_backfill_plan/{section}/{payload_sha256}"
        ),
        "artifact_name": section,
        "name": section,
        "source_section": section,
        "payload_sha256": payload_sha256,
        "content_type": "application/json",
        "payload": payload,
    }

def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
