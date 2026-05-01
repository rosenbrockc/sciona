from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sciona.physics_ingest.writer import (
    PublicationWritePlan,
    PublicationWriter,
    write_publication_rows,
)


def test_publication_writer_inserts_rows_in_dependency_order() -> None:
    client = FakeTableClient()
    rows_by_table = {
        "artifact_symbolic_variables": [{"symbol_name": "F"}],
        "physics_equation_candidates": [{"source_candidate_id": "eq:force"}],
        "physics_ingest_snapshots": [{"source_system": "manual"}],
        "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
    }

    result = PublicationWriter(client).write(rows_by_table)

    assert [(call.mode, call.table) for call in client.calls] == [
        ("insert", "physics_ingest_snapshots"),
        ("insert", "physics_equation_candidates"),
        ("insert", "artifact_symbolic_expressions"),
        ("insert", "artifact_symbolic_variables"),
    ]
    assert [table.table for table in result.tables] == [
        "physics_ingest_snapshots",
        "physics_equation_candidates",
        "artifact_symbolic_expressions",
        "artifact_symbolic_variables",
    ]
    assert result.inserted_count == 4
    assert result.upserted_count == 0
    assert result.diagnostics == ()
    assert rows_by_table["physics_ingest_snapshots"][0] == {"source_system": "manual"}


def test_publication_writer_supports_per_table_upsert_counts() -> None:
    client = FakeTableClient(
        responses={
            ("upsert", "artifact_symbolic_expressions"): {"count": 2},
            ("insert", "artifact_symbolic_variables"): [{"symbol_name": "F"}],
        }
    )
    plan = PublicationWritePlan.from_rows(
        {
            "artifact_symbolic_expressions": [
                {"expression_id": "expr-1"},
                {"expression_id": "expr-2"},
            ],
            "artifact_symbolic_variables": [{"symbol_name": "F"}],
        },
        table_modes={"artifact_symbolic_expressions": "upsert"},
    )

    result = PublicationWriter(client).write(plan)

    assert [(call.mode, call.table, len(call.rows)) for call in client.calls] == [
        ("upsert", "artifact_symbolic_expressions", 2),
        ("insert", "artifact_symbolic_variables", 1),
    ]
    assert result.upserted_count == 2
    assert result.inserted_count == 1
    expression_result = result.table_result("artifact_symbolic_expressions")
    variable_result = result.table_result("artifact_symbolic_variables")
    assert expression_result is not None
    assert variable_result is not None
    assert expression_result.upserted_count == 2
    assert variable_result.inserted_count == 1


def test_publication_writer_dry_run_does_not_call_client() -> None:
    client = FakeTableClient()

    result = write_publication_rows(
        {
            "physics_ingest_snapshots": [{"source_system": "manual"}],
            "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
        },
        client=client,
        table_modes={"artifact_symbolic_expressions": "upsert"},
        dry_run=True,
    )

    assert client.calls == []
    assert result.dry_run is True
    assert result.inserted_count == 0
    assert result.upserted_count == 0
    assert [
        (row.table, row.mode, row.planned_count, row.dry_run)
        for row in result.tables
    ] == [
        ("physics_ingest_snapshots", "insert", 1, True),
        ("artifact_symbolic_expressions", "upsert", 1, True),
    ]
    assert [(row.table, row.reason, row.severity) for row in result.diagnostics] == [
        ("physics_ingest_snapshots", "dry_run", "info"),
        ("artifact_symbolic_expressions", "dry_run", "info"),
    ]


def test_publication_writer_stops_and_reports_failed_table() -> None:
    client = FakeTableClient(fail_on={("insert", "artifact_symbolic_expressions")})

    result = PublicationWriter(client).write(
        {
            "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
            "artifact_symbolic_variables": [{"symbol_name": "F"}],
        }
    )

    assert [(call.mode, call.table) for call in client.calls] == [
        ("insert", "artifact_symbolic_expressions")
    ]
    assert result.inserted_count == 0
    assert result.has_errors is True
    assert [(row.table, row.reason, row.severity) for row in result.diagnostics] == [
        ("artifact_symbolic_expressions", "write_error", "error")
    ]
    assert "configured failure" in result.diagnostics[0].detail


@dataclass(frozen=True)
class WriteCall:
    mode: str
    table: str
    rows: tuple[dict[str, Any], ...]


class FakeTableClient:
    def __init__(
        self,
        *,
        responses: dict[tuple[str, str], Any] | None = None,
        fail_on: set[tuple[str, str]] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.fail_on = fail_on or set()
        self.calls: list[WriteCall] = []

    def insert(self, table: str, rows: tuple[dict[str, Any], ...]) -> Any:
        return self._write("insert", table, rows)

    def upsert(self, table: str, rows: tuple[dict[str, Any], ...]) -> Any:
        return self._write("upsert", table, rows)

    def _write(self, mode: str, table: str, rows: tuple[dict[str, Any], ...]) -> Any:
        key = (mode, table)
        self.calls.append(WriteCall(mode=mode, table=table, rows=rows))
        if key in self.fail_on:
            raise RuntimeError(f"configured failure for {mode} {table}")
        return self.responses.get(key, {"count": len(rows)})
