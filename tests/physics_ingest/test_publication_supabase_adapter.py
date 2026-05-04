from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sciona.physics_ingest.supabase_adapter import (
    PostgrestPublicationTableClient,
    adapt_publication_supabase_client,
    apply_publication_supabase_write,
    preflight_publication_supabase_write,
)
from sciona.physics_ingest.write_plan import PublicationWritePlan
from sciona.physics_ingest.writer import PublicationWriter


def test_adapter_translates_writer_insert_to_postgrest_execute() -> None:
    client = FakePostgrestClient()
    adapter = PostgrestPublicationTableClient(client)
    rows = [{"source_system": "manual"}]

    response = adapter.insert("physics_ingest_snapshots", rows)

    assert response == {"count": 1}
    assert client.calls == [
        QueryCall(
            table="physics_ingest_snapshots",
            mode="insert",
            rows=({"source_system": "manual"},),
            kwargs={},
        )
    ]
    rows[0]["source_system"] = "mutated"
    assert client.calls[0].rows == ({"source_system": "manual"},)


def test_writer_upsert_uses_write_plan_mode_and_conflict_metadata() -> None:
    client = FakePostgrestClient(
        responses={"artifact_symbolic_expressions": {"count": 2}}
    )
    plan = PublicationWritePlan.from_rows(
        {
            "artifact_symbolic_expressions": [
                {"expression_id": "expr-1"},
                {"expression_id": "expr-2"},
            ],
        },
        table_modes={"artifact_symbolic_expressions": "upsert"},
    )
    adapter = adapt_publication_supabase_client(client, write_plan=plan)

    result = PublicationWriter(adapter).write(plan)

    assert result.upserted_count == 2
    assert result.inserted_count == 0
    assert client.calls == [
        QueryCall(
            table="artifact_symbolic_expressions",
            mode="upsert",
            rows=({"expression_id": "expr-1"}, {"expression_id": "expr-2"}),
            kwargs={"on_conflict": "expression_id"},
        )
    ]


def test_adapter_allows_explicit_conflict_metadata_for_unknown_tables() -> None:
    client = FakePostgrestClient()
    adapter = adapt_publication_supabase_client(
        client,
        conflict_keys_by_table={"custom_publications": ("source_id", "version")},
    )

    adapter.upsert("custom_publications", [{"source_id": "src", "version": 1}])

    assert client.calls == [
        QueryCall(
            table="custom_publications",
            mode="upsert",
            rows=({"source_id": "src", "version": 1},),
            kwargs={"on_conflict": "source_id,version"},
        )
    ]


def test_preflight_reports_plan_modes_conflicts_and_missing_upsert_metadata() -> None:
    plan = PublicationWritePlan.from_rows(
        {
            "artifact_symbolic_expressions": [
                {"expression_id": "expr-1"},
                {"expression_id": "expr-2"},
            ],
            "custom_publications": [{"source_id": "src"}],
        },
        table_modes={
            "artifact_symbolic_expressions": "upsert",
            "custom_publications": "upsert",
        },
    )

    report = preflight_publication_supabase_write(plan)

    assert report.missing_conflict_metadata_for_upserts == ("custom_publications",)
    assert report.to_dict() == {
        "table_count": 2,
        "total_row_count": 3,
        "tables": [
            {
                "table": "artifact_symbolic_expressions",
                "mode": "upsert",
                "row_count": 2,
                "conflict_keys": ["expression_id"],
                "missing_conflict_metadata": False,
            },
            {
                "table": "custom_publications",
                "mode": "upsert",
                "row_count": 1,
                "conflict_keys": [],
                "missing_conflict_metadata": True,
            },
        ],
        "missing_conflict_metadata_for_upserts": ["custom_publications"],
        "adapter_capabilities": {
            "imports_supabase": False,
            "requires_injected_client": True,
            "writes_during_preflight": False,
            "supports_insert": True,
            "supports_upsert": True,
            "supports_upsert_on_conflict": True,
        },
    }


def test_preflight_accepts_rows_modes_and_explicit_conflict_metadata() -> None:
    rows = {
        "custom_publications": [{"source_id": "src", "version": 1}],
        "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
    }

    report = preflight_publication_supabase_write(
        rows,
        table_modes={"custom_publications": "upsert"},
        conflict_keys_by_table={"custom_publications": ("source_id", "version")},
    )

    assert [table.to_dict() for table in report.tables] == [
        {
            "table": "physics_ingest_snapshots",
            "mode": "insert",
            "row_count": 1,
            "conflict_keys": ["snapshot_id"],
            "missing_conflict_metadata": False,
        },
        {
            "table": "custom_publications",
            "mode": "upsert",
            "row_count": 1,
            "conflict_keys": ["source_id", "version"],
            "missing_conflict_metadata": False,
        },
    ]


def test_apply_publication_supabase_write_dry_run_does_not_call_client() -> None:
    client = FakePostgrestClient()
    plan = PublicationWritePlan.from_rows(
        {
            "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
            "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
        },
        table_modes={"artifact_symbolic_expressions": "upsert"},
    )

    result = apply_publication_supabase_write(client, plan, dry_run=True)

    assert client.calls == []
    assert result.to_dict() == {
        "dry_run": True,
        "inserted_count": 0,
        "upserted_count": 0,
        "affected_count": 0,
        "has_errors": False,
        "tables": [
            {
                "table": "physics_ingest_snapshots",
                "mode": "insert",
                "planned_count": 1,
                "inserted_count": 0,
                "upserted_count": 0,
                "affected_count": 0,
                "dry_run": True,
            },
            {
                "table": "artifact_symbolic_expressions",
                "mode": "upsert",
                "planned_count": 1,
                "inserted_count": 0,
                "upserted_count": 0,
                "affected_count": 0,
                "dry_run": True,
            },
        ],
        "diagnostics": [
            {
                "table": "physics_ingest_snapshots",
                "reason": "dry_run",
                "severity": "info",
                "detail": "planned insert of 1 rows",
            },
            {
                "table": "artifact_symbolic_expressions",
                "reason": "dry_run",
                "severity": "info",
                "detail": "planned upsert of 1 rows",
            },
        ],
    }


def test_apply_publication_supabase_write_applies_insert_and_upsert_batches() -> None:
    client = FakePostgrestClient(
        responses={
            "physics_ingest_snapshots": {"count": 1},
            "artifact_symbolic_expressions": {"count": 2},
        }
    )

    result = apply_publication_supabase_write(
        client,
        {
            "artifact_symbolic_expressions": [
                {"expression_id": "expr-1"},
                {"expression_id": "expr-2"},
            ],
            "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
        },
        table_modes={"artifact_symbolic_expressions": "upsert"},
    )

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
            rows=({"expression_id": "expr-1"}, {"expression_id": "expr-2"}),
            kwargs={"on_conflict": "expression_id"},
        ),
    ]


def test_apply_publication_supabase_write_propagates_explicit_conflict_metadata() -> None:
    client = FakePostgrestClient()

    apply_publication_supabase_write(
        client,
        {"custom_publications": [{"source_id": "src", "version": 1}]},
        table_modes={"custom_publications": "upsert"},
        conflict_keys_by_table={"custom_publications": ("source_id", "version")},
    )

    assert client.calls == [
        QueryCall(
            table="custom_publications",
            mode="upsert",
            rows=({"source_id": "src", "version": 1},),
            kwargs={"on_conflict": "source_id,version"},
        )
    ]


def test_adapter_does_not_import_supabase_package() -> None:
    source = Path("sciona/physics_ingest/supabase_adapter.py").read_text()

    assert "import supabase" not in source
    assert "from supabase" not in source


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

    def table(self, name: str) -> "FakeTable":
        return FakeTable(self, name)


class FakeTable:
    def __init__(self, client: FakePostgrestClient, name: str) -> None:
        self._client = client
        self._name = name

    def insert(self, rows: tuple[dict[str, Any], ...], **kwargs: Any) -> "FakeQuery":
        return FakeQuery(self._client, self._name, "insert", rows, kwargs)

    def upsert(self, rows: tuple[dict[str, Any], ...], **kwargs: Any) -> "FakeQuery":
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
