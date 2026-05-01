from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sciona.physics_ingest.supabase_adapter import (
    PostgrestPublicationTableClient,
    adapt_publication_supabase_client,
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
