from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from sciona.physics_ingest.deployment import (
    apply_physics_ingest_deployment_storage_bundle,
    build_physics_ingest_deployment_storage_bundle,
    preflight_physics_ingest_deployment_storage_bundle,
)
from sciona.physics_ingest.supabase_adapter import apply_publication_supabase_write


def test_deployment_storage_bundle_merges_rows_in_deterministic_plan_order() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {
            "artifact_relationships": [{"relationship_id": "rel-publication"}],
            "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
            "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
        },
        pdg_catalog_rows={
            "catalog_cdg_nodes": [
                {
                    "version_id": "ver-1",
                    "node_id": "node-1",
                    "projection_kind": "pdg_cdg",
                }
            ],
            "artifact_relationships": [{"relationship_id": "rel-pdg"}],
        },
        review_queue_rows=RowsResult(
            {
                "physics_review_queue_tasks": [
                    {"task_id": "task-1", "status": "open"}
                ]
            },
            summary={"family": "review"},
            diagnostics=({"reason": "queued", "severity": "info"},),
        ),
        audit_artifact_rows={
            "physics_ingest_audit_artifacts": [
                {"artifact_key": "audit-1", "payload_sha256": "abc"}
            ]
        },
    )

    assert bundle.write_plan.ordered_tables() == (
        "physics_ingest_snapshots",
        "artifact_symbolic_expressions",
        "artifact_relationships",
        "catalog_cdg_nodes",
        "physics_review_queue_tasks",
        "physics_ingest_audit_artifacts",
    )
    assert bundle.to_insert_rows()["artifact_relationships"] == [
        {"relationship_id": "rel-publication"},
        {"relationship_id": "rel-pdg"},
    ]
    assert bundle.summary["component_order"] == [
        "publication",
        "pdg_catalog",
        "review_queue",
        "audit_artifacts",
    ]


def test_deployment_storage_bundle_carries_conflict_metadata_and_summary() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {"physics_ingest_snapshots": [{"snapshot_id": "snap-1"}]},
        review_queue_rows={
            "physics_review_queue_tasks": [{"task_id": "task-1"}],
        },
        audit_artifact_rows={
            "physics_ingest_audit_artifacts": [{"artifact_key": "audit-1"}],
        },
        table_modes={
            "physics_review_queue_tasks": "upsert",
            "physics_ingest_audit_artifacts": "upsert",
        },
    )

    assert {
        batch.table: batch.conflict_keys for batch in bundle.write_plan.batches
    } == {
        "physics_ingest_snapshots": ("snapshot_id",),
        "physics_review_queue_tasks": ("task_id",),
        "physics_ingest_audit_artifacts": ("artifact_key",),
    }
    assert bundle.summary["write_plan_conflict_keys"] == {
        "physics_ingest_snapshots": ["snapshot_id"],
        "physics_review_queue_tasks": ["task_id"],
        "physics_ingest_audit_artifacts": ["artifact_key"],
    }
    assert bundle.summary["missing_conflict_metadata_for_upserts"] == []


def test_deployment_storage_bundle_to_dict_is_json_serializable() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {
            "physics_ingest_snapshots": [
                {
                    "snapshot_id": "snap-1",
                    "tags": {"source", "physics"},
                    "score": float("nan"),
                    "metadata": ("tuple", "value"),
                }
            ]
        }
    )

    encoded = json.dumps(bundle.to_dict(), sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)

    row = decoded["insert_rows"]["physics_ingest_snapshots"][0]
    assert row["metadata"] == ["tuple", "value"]
    assert row["score"] is None
    assert row["tags"] == ["physics", "source"]


def test_preflight_and_apply_consume_composed_plan_with_injected_client() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {
            "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
            "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
        },
        review_queue_rows={
            "physics_review_queue_tasks": [{"task_id": "task-1"}],
        },
        table_modes={
            "artifact_symbolic_expressions": "upsert",
            "physics_review_queue_tasks": "upsert",
        },
    )

    preflight = preflight_physics_ingest_deployment_storage_bundle(bundle)
    client = FakePostgrestClient(
        responses={
            "physics_ingest_snapshots": {"count": 1},
            "artifact_symbolic_expressions": {"count": 1},
            "physics_review_queue_tasks": {"count": 1},
        }
    )
    result = apply_publication_supabase_write(client, bundle.write_plan)

    assert preflight.to_dict()["missing_conflict_metadata_for_upserts"] == []
    assert result.inserted_count == 1
    assert result.upserted_count == 2
    assert client.calls == [
        QueryCall(
            table="physics_ingest_snapshots",
            mode="insert",
            rows=({"snapshot_id": "snap-1"},),
            kwargs={},
        ),
        QueryCall(
            table="artifact_symbolic_expressions",
            mode="upsert",
            rows=({"expression_id": "expr-1"},),
            kwargs={"on_conflict": "expression_id"},
        ),
        QueryCall(
            table="physics_review_queue_tasks",
            mode="upsert",
            rows=({"task_id": "task-1"},),
            kwargs={"on_conflict": "task_id"},
        ),
    ]


def test_apply_deployment_storage_bundle_dry_run_does_not_call_client() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {
            "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
            "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
        },
        table_modes={"artifact_symbolic_expressions": "upsert"},
    )
    client = FakePostgrestClient()

    result = apply_physics_ingest_deployment_storage_bundle(
        bundle,
        client,
        dry_run=True,
    )

    assert client.calls == []
    assert result.preflight.total_row_count == 2
    assert result.write_result.dry_run is True
    assert result.write_result.affected_count == 0
    assert result.accounting["planned_row_count"] == 2
    assert result.summary["dry_run"] is True
    assert result.summary["wrote"] is False


def test_apply_deployment_storage_bundle_applies_with_injected_fake_client() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {
            "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
            "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
        },
        review_queue_rows={
            "physics_review_queue_tasks": [{"task_id": "task-1"}],
        },
        table_modes={
            "artifact_symbolic_expressions": "upsert",
            "physics_review_queue_tasks": "upsert",
        },
    )
    client = FakePostgrestClient(
        responses={
            "physics_ingest_snapshots": {"count": 1},
            "artifact_symbolic_expressions": {"count": 1},
            "physics_review_queue_tasks": {"count": 1},
        }
    )

    result = apply_physics_ingest_deployment_storage_bundle(
        bundle,
        client,
        dry_run=False,
    )

    assert result.write_result.inserted_count == 1
    assert result.write_result.upserted_count == 2
    assert result.accounting["affected_count"] == 3
    assert result.summary["wrote"] is True
    assert client.calls == [
        QueryCall(
            table="physics_ingest_snapshots",
            mode="insert",
            rows=({"snapshot_id": "snap-1"},),
            kwargs={},
        ),
        QueryCall(
            table="artifact_symbolic_expressions",
            mode="upsert",
            rows=({"expression_id": "expr-1"},),
            kwargs={"on_conflict": "expression_id"},
        ),
        QueryCall(
            table="physics_review_queue_tasks",
            mode="upsert",
            rows=({"task_id": "task-1"},),
            kwargs={"on_conflict": "task_id"},
        ),
    ]


def test_apply_deployment_storage_bundle_requires_client_for_non_dry_run() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {"physics_ingest_snapshots": [{"snapshot_id": "snap-1"}]},
    )

    try:
        apply_physics_ingest_deployment_storage_bundle(
            bundle,
            dry_run=False,
        )
    except ValueError as exc:
        assert str(exc) == "client is required when dry_run is False"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected missing client to raise")


def test_apply_deployment_storage_bundle_result_is_json_serializable() -> None:
    bundle = build_physics_ingest_deployment_storage_bundle(
        {
            "physics_ingest_snapshots": [
                {
                    "snapshot_id": "snap-1",
                    "score": float("inf"),
                    "tags": {"deployment", "physics"},
                }
            ]
        }
    )

    result = apply_physics_ingest_deployment_storage_bundle(bundle, dry_run=True)
    encoded = json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["preflight"]["total_row_count"] == 1
    assert decoded["write_result"]["dry_run"] is True
    assert decoded["accounting"]["planned_row_count"] == 1
    assert (
        decoded["summary"]["bundle_summary"]["summary_kind"]
        == "physics_ingest_deployment_storage_bundle.v1"
    )


def test_deployment_storage_module_does_not_import_supabase_package() -> None:
    source = Path("sciona/physics_ingest/deployment.py").read_text()

    assert "import supabase" not in source
    assert "from supabase" not in source


@dataclass(frozen=True)
class RowsResult:
    rows: dict[str, list[dict[str, Any]]]
    summary: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...] = ()

    def to_insert_rows(self) -> dict[str, list[dict[str, Any]]]:
        return {table: [dict(row) for row in rows] for table, rows in self.rows.items()}


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
