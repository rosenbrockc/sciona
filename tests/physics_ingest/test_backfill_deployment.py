from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from sciona.physics_ingest.audit_artifacts import BACKFILL_AUDIT_ARTIFACTS_TABLE
from sciona.physics_ingest.backfill import build_physics_ingest_backfill_report
from sciona.physics_ingest.backfill_deployment import (
    BACKFILL_DEPLOYMENT_REPORT_VERSION,
    build_physics_ingest_backfill_deployment_report,
)
from sciona.physics_ingest.sources import build_physics_source_retrieval_run_plan


def test_backfill_deployment_builds_report_and_persists_dashboard_audit_manifests() -> None:
    report = build_physics_ingest_backfill_deployment_report()
    report_dict = report.to_dict()

    assert report_dict["report_version"] == BACKFILL_DEPLOYMENT_REPORT_VERSION
    assert report_dict["input"]["backfill_report_source"] == "built_report"
    assert report_dict["summary"]["audit_artifact_manifest_count"] >= 2

    rows = report.audit_artifact_rows.to_insert_rows()[BACKFILL_AUDIT_ARTIFACTS_TABLE]
    sections = {row["source_section"] for row in rows}

    assert {"dashboard_summary", "audit_replay"} <= sections
    assert all("payload" in row for row in rows)
    assert report.storage_bundle.summary["write_plan_table_modes"][
        BACKFILL_AUDIT_ARTIFACTS_TABLE
    ] == "upsert"
    assert report.storage_bundle.summary["write_plan_conflict_keys"][
        BACKFILL_AUDIT_ARTIFACTS_TABLE
    ] == ["artifact_key"]
    assert report_dict["storage_preflight_summary"]["table_modes"][
        BACKFILL_AUDIT_ARTIFACTS_TABLE
    ] == "upsert"


def test_backfill_deployment_accepts_existing_report_without_mutation() -> None:
    existing = build_physics_ingest_backfill_report(
        include_rows=True,
        include_audit_artifact_manifests=True,
        include_publication_write_preflight=True,
    )
    original = deepcopy(existing)

    report = build_physics_ingest_backfill_deployment_report(
        backfill_report=existing,
    )

    assert existing == original
    assert report.backfill_report == original
    assert report.backfill_report is not existing

    existing["dashboard_summary"]["ok"] = False
    assert report.backfill_report["dashboard_summary"]["ok"] is True
    assert report.to_dict()["input"]["backfill_report_source"] == "existing_report"


def test_backfill_deployment_apply_dry_run_does_not_call_client() -> None:
    client = FakePostgrestClient()

    report = build_physics_ingest_backfill_deployment_report(
        client=client,
        dry_run=True,
    )

    assert client.calls == []
    assert report.storage_apply_result is not None
    assert report.storage_apply_result.write_result.dry_run is True
    assert report.storage_apply_result.write_result.affected_count == 0
    assert report.to_dict()["summary"]["storage_write_performed"] is False


def test_backfill_deployment_applies_with_injected_fake_client() -> None:
    client = FakePostgrestClient(
        responses={BACKFILL_AUDIT_ARTIFACTS_TABLE: {"count": 2}},
    )

    report = build_physics_ingest_backfill_deployment_report(
        client=client,
        dry_run=False,
    )

    assert report.storage_apply_result is not None
    assert report.storage_apply_result.write_result.upserted_count == 2
    assert report.storage_apply_result.accounting["affected_count"] == 2
    assert report.to_dict()["summary"]["storage_write_performed"] is True
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.table == BACKFILL_AUDIT_ARTIFACTS_TABLE
    assert call.mode == "upsert"
    assert call.kwargs == {"on_conflict": "artifact_key"}
    assert len(call.rows) >= 2


def test_backfill_deployment_includes_runtime_preflight_summary() -> None:
    plan = build_physics_source_retrieval_run_plan(max_jobs=2, limit=5)

    report = build_physics_ingest_backfill_deployment_report(
        source_retrieval_run_plan=plan,
    ).to_dict()

    assert report["runtime_preflight_summary"]["source_runtime_step_count"] == 2
    assert report["runtime_preflight"]["source_runtime_counts"]["total_steps"] == 2
    assert report["runtime_preflight"]["summary"]["storage_table_count"] >= 1
    assert report["summary"]["runtime_blocked"] is False
    assert report["dashboard_summary"]["runtime"]["source_runtime_step_count"] == 2
    assert report["dashboard_summary"]["runtime"]["blocked"] is False


def test_backfill_deployment_surfaces_compact_summaries_without_full_report() -> None:
    report = build_physics_ingest_backfill_deployment_report(
        include_backfill_report=False,
    ).to_dict()

    assert "backfill_report" not in report
    assert report["backfill_dashboard_summary"]["ok"] is True
    assert "publication_readiness" in report["backfill_dashboard_summary"]
    assert "phase7_coverage" in report["backfill_dashboard_summary"]
    assert {
        "by_review_status",
        "by_validation_status",
    } <= set(report["backfill_dashboard_summary"]["phase7_coverage"])
    assert report["audit_artifact_manifest_summary"]["artifact_count"] == report[
        "summary"
    ]["audit_artifact_manifest_count"]
    assert report["audit_artifact_manifest_summary"]["artifact_keys"]
    dashboard = report["dashboard_summary"]
    assert dashboard["report_version"] == (
        "physics-ingest-backfill-deployment-dashboard.v1"
    )
    assert dashboard["ok"] is True
    assert dashboard["backfill"]["phase7_row_count"] == report[
        "backfill_dashboard_summary"
    ]["phase7_coverage"]["row_count"]
    assert dashboard["audit_artifacts"]["manifest_count"] == report["summary"][
        "audit_artifact_manifest_count"
    ]
    assert dashboard["audit_artifacts"]["manifest_artifact_count"] == report[
        "audit_artifact_manifest_summary"
    ]["artifact_count"]
    assert dashboard["storage"]["total_row_count"] == report[
        "storage_preflight_summary"
    ]["total_row_count"]
    assert json.loads(json.dumps(report, sort_keys=True)) == report


def test_backfill_deployment_report_is_json_serializable() -> None:
    report = build_physics_ingest_backfill_deployment_report(
        review_diagnostics=[
            {
                "stage": "review",
                "severity": "info",
                "score": float("nan"),
                "tags": {"bulk", "backfill"},
            }
        ],
    )
    report_dict = report.to_dict()

    encoded = json.dumps(report_dict, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded == report_dict
    assert report.ok is True
    assert report.dry_run is True


@dataclass(frozen=True)
class QueryCall:
    table: str
    mode: str
    rows: tuple[dict[str, Any], ...]
    kwargs: dict[str, Any]


class FakePostgrestClient:
    def __init__(self, *, responses: dict[str, Any] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[QueryCall] = []

    def table(self, name: str) -> FakeTable:
        return FakeTable(self, name)


class FakeTable:
    def __init__(self, client: FakePostgrestClient, name: str) -> None:
        self._client = client
        self._name = name

    def insert(self, rows: tuple[dict[str, Any], ...], **kwargs: Any) -> FakeQuery:
        return FakeQuery(self._client, self._name, "insert", rows, kwargs)

    def upsert(self, rows: tuple[dict[str, Any], ...], **kwargs: Any) -> FakeQuery:
        return FakeQuery(self._client, self._name, "upsert", rows, kwargs)


class FakeQuery:
    def __init__(
        self,
        client: FakePostgrestClient,
        table: str,
        mode: str,
        rows: tuple[dict[str, Any], ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._client = client
        self._table = table
        self._mode = mode
        self._rows = rows
        self._kwargs = kwargs

    def execute(self) -> Any:
        self._client.calls.append(
            QueryCall(
                table=self._table,
                mode=self._mode,
                rows=self._rows,
                kwargs=self._kwargs,
            )
        )
        return self._client.responses.get(self._table, {"count": len(self._rows)})
