from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from sciona.physics_ingest.pdg_cdg import (
    PDGCDGArtifactEnvelope,
    build_pdg_publication_write_rows,
    build_pdg_relationship_ingest,
)
from sciona.physics_ingest.pdg_deployment import (
    build_pdg_deployment_storage_plan,
)
from sciona.physics_ingest.sources.pdg import parse_pdg_document


EXPR_BASE = "10000000-0000-0000-0000-000000000001"
EXPR_SOLVED = "10000000-0000-0000-0000-000000000002"
EXPR_FORCE = "10000000-0000-0000-0000-000000000003"
EXPECTED_CATALOG_TABLES = (
    "catalog_cdg_artifacts",
    "catalog_cdg_versions",
    "catalog_cdg_nodes",
    "catalog_cdg_relationships",
    "catalog_symbolic_artifacts",
)
EXPECTED_TABLE_ORDER = (
    "artifacts",
    "artifact_versions",
    "artifact_relationships",
    "artifact_cdg_nodes",
    "artifact_cdg_edges",
    "artifact_cdg_bindings",
    *EXPECTED_CATALOG_TABLES,
)


def test_pdg_deployment_bundle_has_deterministic_write_order_with_catalog_tables() -> None:
    plan = build_pdg_deployment_storage_plan(_publication_rows())

    assert plan.catalog_write_plan_rows.write_plan is not None
    assert plan.catalog_write_plan_rows.write_plan.ordered_tables() == EXPECTED_TABLE_ORDER
    assert plan.deployment_bundle.write_plan.ordered_tables() == EXPECTED_TABLE_ORDER
    assert plan.summary["write_plan_table_order"] == list(EXPECTED_TABLE_ORDER)
    assert plan.deployment_bundle.summary["component_order"] == [
        "publication",
        "pdg_catalog",
    ]
    assert plan.summary["catalog_projection_summary"]["table_row_counts"] == {
        "catalog_cdg_artifacts": 1,
        "catalog_cdg_nodes": 2,
        "catalog_cdg_relationships": 3,
        "catalog_cdg_versions": 1,
        "catalog_symbolic_artifacts": 1,
    }
    assert plan.summary["total_row_count"] == 19
    assert plan.dashboard_summary["report_version"] == (
        "pdg-deployment-storage-dashboard.v1"
    )
    assert plan.dashboard_summary["storage"] == {
        "table_count": len(EXPECTED_TABLE_ORDER),
        "total_row_count": 19,
        "table_row_counts": {
            "artifact_cdg_bindings": 4,
            "artifact_cdg_edges": 1,
            "artifact_cdg_nodes": 2,
            "artifact_relationships": 2,
            "artifact_versions": 1,
            "artifacts": 1,
            "catalog_cdg_artifacts": 1,
            "catalog_cdg_nodes": 2,
            "catalog_cdg_relationships": 3,
            "catalog_cdg_versions": 1,
            "catalog_symbolic_artifacts": 1,
        },
        "preflight_table_count": len(EXPECTED_TABLE_ORDER),
        "preflight_total_row_count": 19,
        "missing_conflict_metadata_count": 0,
        "missing_conflict_metadata_for_upserts": [],
    }
    assert plan.dashboard_summary["catalog"]["catalogable_cdg_count"] == 1
    assert plan.dashboard_summary["catalog"]["projected_row_count"] == 8


def test_pdg_deployment_catalog_tables_carry_conflict_metadata() -> None:
    plan = build_pdg_deployment_storage_plan(_publication_rows().to_insert_rows())
    batches_by_table = plan.deployment_bundle.write_plan.batches_by_table()

    assert batches_by_table["catalog_cdg_artifacts"].conflict_keys == (
        "artifact_id",
        "version_id",
        "projection_kind",
    )
    assert batches_by_table["catalog_cdg_versions"].conflict_keys == (
        "version_id",
        "projection_kind",
    )
    assert batches_by_table["catalog_cdg_nodes"].conflict_keys == (
        "version_id",
        "node_id",
        "projection_kind",
    )
    assert batches_by_table["catalog_cdg_relationships"].conflict_keys == (
        "relationship_id",
        "projection_kind",
    )
    assert batches_by_table["catalog_symbolic_artifacts"].conflict_keys == (
        "artifact_id",
        "version_id",
        "projection_kind",
    )
    assert plan.preflight is not None
    assert plan.preflight.to_dict()["missing_conflict_metadata_for_upserts"] == []


def test_pdg_deployment_dry_run_apply_does_not_call_client() -> None:
    client = FakePostgrestClient()

    plan = build_pdg_deployment_storage_plan(
        _publication_rows(),
        apply_storage=True,
        client=client,
    )

    assert client.calls == []
    assert plan.apply_result is not None
    assert plan.apply_result.write_result.dry_run is True
    assert plan.apply_result.accounting["planned_row_count"] == 19
    assert plan.apply_result.accounting["affected_count"] == 0
    assert plan.summary["dry_run"] is True
    assert plan.summary["wrote"] is False
    assert plan.dashboard_summary["apply"] == {
        "requested": True,
        "performed": False,
        "dry_run": True,
        "affected_count": 0,
    }


def test_pdg_deployment_non_dry_run_applies_through_injected_fake_client() -> None:
    table_modes = {table: "upsert" for table in EXPECTED_CATALOG_TABLES}
    client = FakePostgrestClient()

    plan = build_pdg_deployment_storage_plan(
        _publication_rows(),
        table_modes=table_modes,
        apply_storage=True,
        client=client,
        dry_run=False,
    )

    assert plan.apply_result is not None
    assert plan.apply_result.accounting["affected_count"] == 19
    assert plan.summary["wrote"] is True
    assert plan.dashboard_summary["apply"] == {
        "requested": True,
        "performed": True,
        "dry_run": False,
        "affected_count": 19,
    }
    assert [call.table for call in client.calls] == list(EXPECTED_TABLE_ORDER)
    assert [call.mode for call in client.calls[-5:]] == ["upsert"] * 5
    assert client.calls[-1].kwargs == {
        "on_conflict": "artifact_id,version_id,projection_kind"
    }


def test_pdg_deployment_preserves_missing_envelope_diagnostics() -> None:
    plan = build_pdg_deployment_storage_plan(
        _publication_rows(include_envelope=False),
    )

    assert [diagnostic["reason"] for diagnostic in plan.diagnostics] == [
        "missing_version_envelope",
        "orphan_cdg_row",
        "no_catalogable_cdgs",
    ]
    assert plan.summary["diagnostics_by_reason"] == {
        "missing_version_envelope": 1,
        "no_catalogable_cdgs": 1,
        "orphan_cdg_row": 1,
    }
    assert plan.dashboard_summary["diagnostics"] == {
        "diagnostic_count": 3,
        "by_reason": {
            "missing_version_envelope": 1,
            "no_catalogable_cdgs": 1,
            "orphan_cdg_row": 1,
        },
        "by_severity": {"warning": 3},
        "by_table": {
            "artifact_cdg_nodes": 1,
            "artifact_versions": 1,
            "catalog_symbolic_artifacts": 1,
        },
    }
    assert "catalog_symbolic_artifacts" not in plan.to_insert_rows()
    assert plan.summary["write_plan_table_order"] == [
        "artifact_relationships",
        "artifact_cdg_nodes",
        "artifact_cdg_edges",
        "artifact_cdg_bindings",
    ]


def test_pdg_deployment_result_is_json_serializable() -> None:
    plan = build_pdg_deployment_storage_plan(_publication_rows())

    encoded = json.dumps(plan.to_dict(), sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["summary"]["summary_kind"] == "pdg_deployment_storage_plan.v1"
    assert decoded["dashboard_summary"]["storage"]["total_row_count"] == 19
    assert decoded["preflight"]["total_row_count"] == 19
    assert decoded["deployment_bundle"]["write_plan"]["audit_summary"][
        "total_row_count"
    ] == 19


def test_pdg_deployment_module_does_not_import_supabase_package() -> None:
    source = Path("sciona/physics_ingest/pdg_deployment.py").read_text()

    assert "import supabase" not in source
    assert "from supabase" not in source


def _bundle():
    return parse_pdg_document(
        {
            "equations": [
                {"id": "eq:base", "label": "Newton's second law", "latex": "F = m a"},
                {
                    "id": "eq:solved",
                    "label": "Acceleration from force",
                    "latex": "a = F / m",
                },
                {
                    "id": "eq:force",
                    "label": "Constant mass force",
                    "latex": "F(t) = m d^2x/dt^2",
                },
            ],
            "inference_edges": [
                {
                    "id": "edge:solve",
                    "source": "eq:base",
                    "target": "eq:solved",
                    "rule": "solve for acceleration",
                    "confidence": 0.93,
                    "bindings": {"solve_for": "a"},
                },
                {
                    "id": "edge:substitute",
                    "source": "eq:solved",
                    "target": "eq:force",
                    "rule": "substitution",
                    "assumptions": ["mass is constant"],
                    "bindings": {"variables": {"a": "d2x_dt2"}},
                    "confidence": 0.81,
                },
            ],
        }
    )


def _publication_rows(*, include_envelope: bool = True):
    result = build_pdg_relationship_ingest(
        _bundle(),
        expression_bindings_by_pdg_node_id={
            "eq:base": {
                "expression_id": EXPR_BASE,
                "metadata": {
                    "bound_artifact_fqdn": "physics.newton.base",
                    "bound_version_content_hash": "hash-base",
                },
            },
            "eq:solved": {
                "expression_id": EXPR_SOLVED,
                "metadata": {
                    "bound_artifact_fqdn": "physics.newton.solved",
                    "bound_version_content_hash": "hash-solved",
                },
            },
            "eq:force": {
                "expression_id": EXPR_FORCE,
                "metadata": {
                    "bound_artifact_fqdn": "physics.newton.force",
                    "bound_version_content_hash": "hash-force",
                },
            },
        },
    )
    envelope = (
        PDGCDGArtifactEnvelope(
            fqdn_prefix="physics.pdg.cdg",
            semver="2026.5.3",
            namespace_root="physics",
            namespace_path="pdg/cdg",
            source_package="pdg",
            source_module_path="pdg.derivations",
            is_latest=True,
        )
        if include_envelope
        else None
    )
    return build_pdg_publication_write_rows(result, cdg_artifact_envelope=envelope)


@dataclass(frozen=True)
class QueryCall:
    table: str
    mode: str
    rows: tuple[dict[str, Any], ...]
    kwargs: dict[str, Any]


class FakePostgrestClient:
    def __init__(self) -> None:
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

    def execute(self) -> dict[str, int]:
        self._client.calls.append(
            QueryCall(
                table=self._table,
                mode=self._mode,
                rows=self._rows,
                kwargs=self._kwargs,
            )
        )
        return {"count": len(self._rows)}
