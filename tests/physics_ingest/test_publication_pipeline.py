from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sciona.physics_ingest.pipeline import run_physics_publication_pipeline


ARTIFACT_ID = "20000000-0000-0000-0000-000000000001"
VERSION_ID = "30000000-0000-0000-0000-000000000001"


def test_pipeline_plans_deterministic_source_ids_and_dry_run_write_plan() -> None:
    bundle = _source_bundle()

    result = run_physics_publication_pipeline(
        source_bundles=[bundle],
        publication_manifests=[_publication_manifest()],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        dry_run=True,
    )

    planned_bundle = result.planned_source_bundles[0]
    planned_snapshot = planned_bundle["snapshot_row"]
    planned_candidate = planned_bundle["candidate_rows"][0]

    assert UUID(planned_snapshot["snapshot_id"])
    assert UUID(planned_candidate["candidate_id"])
    assert result.snapshot_id_bindings == {
        "fixture-bundle": planned_snapshot["snapshot_id"],
        "fixture.adapter": planned_snapshot["snapshot_id"],
        "manual": planned_snapshot["snapshot_id"],
    }
    assert result.orchestration_result.diagnostics == ()
    assert result.write_plan.ordered_tables() == (
        "physics_ingest_snapshots",
        "physics_equation_candidates",
        "artifact_symbolic_expressions",
        "artifact_symbolic_variables",
    )
    assert result.write_result is not None
    assert result.write_result.dry_run is True
    assert result.write_result.affected_count == 0
    assert result.summary.to_dict() == {
        "dry_run": True,
        "source_bundle_count": 1,
        "publication_manifest_count": 1,
        "snapshot_binding_count": 3,
        "planned_batch_count": 4,
        "planned_row_count": 4,
        "wrote": True,
        "affected_row_count": 0,
        "orchestration_error_count": 0,
        "write_error_count": 0,
        "has_errors": False,
    }
    assert [row["stage"] for row in result.diagnostics] == ["write"] * 4
    assert bundle["snapshot_row"].get("snapshot_id") is None
    assert bundle["candidate_rows"][0].get("candidate_id") is None


def test_pipeline_executes_with_injected_client_and_table_modes() -> None:
    client = FakeTableClient(
        responses={("upsert", "artifact_symbolic_expressions"): {"count": 1}}
    )

    result = run_physics_publication_pipeline(
        source_bundles=[_source_bundle()],
        publication_manifests=[_publication_manifest()],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        client=client,
        table_modes={"artifact_symbolic_expressions": "upsert"},
        dry_run=False,
    )

    assert [(call.mode, call.table) for call in client.calls] == [
        ("insert", "physics_ingest_snapshots"),
        ("insert", "physics_equation_candidates"),
        ("upsert", "artifact_symbolic_expressions"),
        ("insert", "artifact_symbolic_variables"),
    ]
    assert result.write_result is not None
    assert result.write_result.inserted_count == 3
    assert result.write_result.upserted_count == 1
    assert result.summary.affected_row_count == 4
    assert result.summary.has_errors is False


def test_pipeline_can_stop_at_side_effect_free_plan_without_client() -> None:
    result = run_physics_publication_pipeline(
        publication_manifests=[_publication_manifest()],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        dry_run=False,
    )

    assert result.write_result is None
    assert result.summary.wrote is False
    assert result.summary.planned_row_count == 2
    assert result.diagnostics == ()


def test_pipeline_merges_additional_publication_rows_into_write_plan() -> None:
    result = run_physics_publication_pipeline(
        publication_manifests=[_publication_manifest()],
        additional_insert_rows={
            "artifact_relationships": [{"relationship_id": "rel-1"}],
            "artifact_cdg_nodes": [{"version_id": VERSION_ID, "node_id": "step-1"}],
        },
        additional_diagnostics=[
            {
                "stage": "pdg_cdg_publication",
                "table": "artifact_cdg_bindings",
                "reason": "missing_cdg_binding_artifact_metadata",
                "severity": "skipped",
                "artifact_key": "pdg-cdg",
            }
        ],
        artifact_bindings={
            "local:fixture.force": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        dry_run=False,
    )

    assert result.write_plan.ordered_tables() == (
        "artifact_symbolic_expressions",
        "artifact_symbolic_variables",
        "artifact_relationships",
        "artifact_cdg_nodes",
    )
    assert result.write_plan.audit_summary.total_row_count == 4
    assert result.diagnostics == (
        {
            "stage": "pdg_cdg_publication",
            "table": "artifact_cdg_bindings",
            "reason": "missing_cdg_binding_artifact_metadata",
            "severity": "skipped",
            "artifact_key": "pdg-cdg",
            "atom_name": "",
            "detail": "",
        },
    )


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


@dataclass(frozen=True)
class WriteCall:
    mode: str
    table: str
    rows: tuple[dict[str, Any], ...]


class FakeTableClient:
    def __init__(self, *, responses: dict[tuple[str, str], Any] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[WriteCall] = []

    def insert(self, table: str, rows: tuple[dict[str, Any], ...]) -> Any:
        return self._write("insert", table, rows)

    def upsert(self, table: str, rows: tuple[dict[str, Any], ...]) -> Any:
        return self._write("upsert", table, rows)

    def _write(self, mode: str, table: str, rows: tuple[dict[str, Any], ...]) -> Any:
        self.calls.append(WriteCall(mode=mode, table=table, rows=rows))
        return self.responses.get((mode, table), {"count": len(rows)})
