from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sciona.physics_ingest.ids import plan_source_bundle_ids, source_candidate_id
from sciona.physics_ingest.orchestration import orchestrate_physics_publication
from sciona.physics_ingest.sources.foundational_physics import (
    FOUNDATIONAL_LAW_SEEDS,
    build_foundational_physics_backfill_bundle,
)
from sciona.physics_ingest.write_plan import build_publication_write_plan
from sciona.physics_ingest.writer import PublicationWriter


ARTIFACT_ID = "20000000-0000-0000-0000-000000000101"
VERSION_ID = "30000000-0000-0000-0000-000000000101"


def test_foundational_physics_publication_pipeline_is_offline_end_to_end() -> None:
    source_bundle = build_foundational_physics_backfill_bundle(
        retrieved_at="2026-04-30T00:00:00Z",
    )
    source_candidate_rows = [dict(row) for row in source_bundle.candidate_rows]

    snapshot_id_bindings, planned_bundles = plan_source_bundle_ids([source_bundle])
    planned_snapshot_id = snapshot_id_bindings[
        "sciona.physics_ingest.sources.foundational_physics"
    ]

    assert planned_snapshot_id == "9b71ba80-1c66-59ad-8056-7f23907b5dbd"
    assert source_bundle.candidate_rows[0] == source_candidate_rows[0]
    assert "candidate_id" not in source_bundle.candidate_rows[0]
    assert planned_bundles[0]["candidate_rows"][0]["candidate_id"] == (
        source_candidate_id(planned_snapshot_id, FOUNDATIONAL_LAW_SEEDS[0].source_id)
    )

    orchestrated = orchestrate_physics_publication(
        source_bundles=planned_bundles,
        publication_manifests=[_publication_manifest()],
        artifact_bindings={
            "local:sciona-atoms-physics:fixture.newton_second_law": {
                "artifact_id": ARTIFACT_ID,
                "version_id": VERSION_ID,
            }
        },
        snapshot_id_bindings=snapshot_id_bindings,
    )

    insert_rows = orchestrated.to_insert_rows()
    assert orchestrated.diagnostics == ()
    assert orchestrated.audit_summary.to_dict() == {
        "source_bundle_count": 1,
        "publication_manifest_count": 1,
        "input_row_counts": {
            "artifact_symbolic_expressions": 1,
            "artifact_symbolic_variables": 3,
            "artifact_validity_bounds": 1,
            "physics_equation_candidates": len(FOUNDATIONAL_LAW_SEEDS),
            "physics_ingest_snapshots": 1,
        },
        "insert_row_counts": {
            "artifact_symbolic_expressions": 1,
            "artifact_symbolic_variables": 3,
            "artifact_validity_bounds": 1,
            "physics_equation_candidates": len(FOUNDATIONAL_LAW_SEEDS),
            "physics_ingest_snapshots": 1,
        },
        "skipped_row_count": 0,
        "error_row_count": 0,
        "has_errors": False,
    }
    assert (
        insert_rows["physics_ingest_snapshots"][0]["snapshot_id"]
        == planned_snapshot_id
    )
    assert len(insert_rows["physics_equation_candidates"]) == len(FOUNDATIONAL_LAW_SEEDS)
    assert insert_rows["physics_equation_candidates"][0]["candidate_id"] == (
        source_candidate_id(planned_snapshot_id, FOUNDATIONAL_LAW_SEEDS[0].source_id)
    )
    assert (
        insert_rows["artifact_symbolic_expressions"][0]["artifact_id"] == ARTIFACT_ID
    )

    write_plan = build_publication_write_plan(insert_rows)

    assert write_plan.ordered_tables() == (
        "physics_ingest_snapshots",
        "physics_equation_candidates",
        "artifact_symbolic_expressions",
        "artifact_symbolic_variables",
        "artifact_validity_bounds",
    )
    assert write_plan.audit_summary.total_row_count == (
        1 + len(FOUNDATIONAL_LAW_SEEDS) + 1 + 3 + 1
    )

    client = FakePublicationClient()
    write_result = PublicationWriter(client).write(write_plan, dry_run=True)

    assert client.calls == []
    assert write_result.dry_run is True
    assert write_result.inserted_count == 0
    assert write_result.upserted_count == 0
    assert write_result.has_errors is False
    assert [
        (table.table, table.mode, table.planned_count, table.dry_run)
        for table in write_result.tables
    ] == [
        ("physics_ingest_snapshots", "insert", 1, True),
        ("physics_equation_candidates", "insert", len(FOUNDATIONAL_LAW_SEEDS), True),
        ("artifact_symbolic_expressions", "insert", 1, True),
        ("artifact_symbolic_variables", "insert", 3, True),
        ("artifact_validity_bounds", "insert", 1, True),
    ]
    assert {diagnostic.reason for diagnostic in write_result.diagnostics} == {"dry_run"}

    fake_write_result = PublicationWriter(client).write(write_plan)

    assert [(call.mode, call.table, len(call.rows)) for call in client.calls] == [
        ("insert", "physics_ingest_snapshots", 1),
        ("insert", "physics_equation_candidates", len(FOUNDATIONAL_LAW_SEEDS)),
        ("insert", "artifact_symbolic_expressions", 1),
        ("insert", "artifact_symbolic_variables", 3),
        ("insert", "artifact_validity_bounds", 1),
    ]
    assert fake_write_result.dry_run is False
    assert fake_write_result.inserted_count == write_plan.audit_summary.total_row_count
    assert fake_write_result.upserted_count == 0
    assert fake_write_result.diagnostics == ()


def _publication_manifest() -> dict[str, object]:
    artifact_key = "local:sciona-atoms-physics:fixture.newton_second_law"
    return {
        "provider": "sciona-atoms-physics",
        "modules": ["fixture"],
        "artifact_symbolic_expressions": [
            {
                "artifact_key": artifact_key,
                "local_artifact_key": artifact_key,
                "provider": "sciona-atoms-physics",
                "atom_name": "newton_second_law_atom",
                "atom_module": "fixture.newton_second_law",
                "registry_name": "newton_second_law_atom",
                "expression_srepr": "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))",
                "expression_text": "Eq(F, a*m)",
                "variables": {"F": "output", "m": "input", "a": "input"},
                "dim_signature": {"F": "M1L1T-2", "m": "M1", "a": "L1T-2"},
                "symbolic_dim_signature": {
                    "F": "M1L1T-2",
                    "m": "M1",
                    "a": "L1T-2",
                },
                "constants": {},
                "bibliography": [{"title": "fixture"}],
                "artifact_uuid": None,
            }
        ],
        "artifact_symbolic_variables": [
            {
                "artifact_key": artifact_key,
                "provider": "sciona-atoms-physics",
                "atom_name": "newton_second_law_atom",
                "symbol": "F",
                "role": "output",
                "dim_signature": "M1L1T-2",
            },
            {
                "artifact_key": artifact_key,
                "provider": "sciona-atoms-physics",
                "atom_name": "newton_second_law_atom",
                "symbol": "m",
                "role": "input",
                "dim_signature": "M1",
            },
            {
                "artifact_key": artifact_key,
                "provider": "sciona-atoms-physics",
                "atom_name": "newton_second_law_atom",
                "symbol": "a",
                "role": "input",
                "dim_signature": "L1T-2",
            },
        ],
        "artifact_validity_bounds": [
            {
                "artifact_key": artifact_key,
                "provider": "sciona-atoms-physics",
                "atom_name": "newton_second_law_atom",
                "symbol": "m",
                "min_value": 0.0,
                "max_value": None,
            }
        ],
    }


@dataclass(frozen=True)
class WriteCall:
    mode: str
    table: str
    rows: tuple[dict[str, Any], ...]


class FakePublicationClient:
    def __init__(self) -> None:
        self.calls: list[WriteCall] = []

    def insert(self, table: str, rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
        self.calls.append(WriteCall(mode="insert", table=table, rows=rows))
        return {"count": len(rows)}

    def upsert(self, table: str, rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
        self.calls.append(WriteCall(mode="upsert", table=table, rows=rows))
        return {"count": len(rows)}
