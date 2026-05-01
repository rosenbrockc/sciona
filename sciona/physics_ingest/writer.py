"""Dependency-ordered publication writes for physics ingestion.

This module intentionally knows nothing about Supabase. Callers inject a small
table client so publication writes can be tested with fakes and adapted to any
storage backend at the application boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from sciona.physics_ingest.write_plan import (
    PUBLICATION_TABLE_ORDER,
    PublicationWritePlan,
    WriteMode,
)


class PublicationTableClient(Protocol):
    """Minimal client protocol used by :class:`PublicationWriter`."""

    def insert(
        self, table: str, rows: Sequence[Mapping[str, Any]]
    ) -> Any:
        """Insert rows into ``table`` and return a backend-specific result."""

    def upsert(
        self, table: str, rows: Sequence[Mapping[str, Any]]
    ) -> Any:
        """Upsert rows into ``table`` and return a backend-specific result."""


@dataclass(frozen=True)
class PublicationWriteDiagnostic:
    """One non-fatal publication write diagnostic."""

    table: str
    reason: str
    severity: Literal["info", "error"] = "info"
    detail: str = ""


@dataclass(frozen=True)
class PublicationTableWriteResult:
    """Write accounting for one table."""

    table: str
    mode: WriteMode
    planned_count: int
    inserted_count: int = 0
    upserted_count: int = 0
    dry_run: bool = False

    @property
    def affected_count(self) -> int:
        return self.inserted_count + self.upserted_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "mode": self.mode,
            "planned_count": self.planned_count,
            "inserted_count": self.inserted_count,
            "upserted_count": self.upserted_count,
            "affected_count": self.affected_count,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class PublicationWriteResult:
    """Result returned after applying a publication write plan."""

    tables: tuple[PublicationTableWriteResult, ...] = ()
    diagnostics: tuple[PublicationWriteDiagnostic, ...] = ()
    dry_run: bool = False

    @property
    def inserted_count(self) -> int:
        return sum(table.inserted_count for table in self.tables)

    @property
    def upserted_count(self) -> int:
        return sum(table.upserted_count for table in self.tables)

    @property
    def affected_count(self) -> int:
        return self.inserted_count + self.upserted_count

    @property
    def has_errors(self) -> bool:
        return any(row.severity == "error" for row in self.diagnostics)

    def table_result(self, table: str) -> PublicationTableWriteResult | None:
        return next((row for row in self.tables if row.table == table), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "inserted_count": self.inserted_count,
            "upserted_count": self.upserted_count,
            "affected_count": self.affected_count,
            "has_errors": self.has_errors,
            "tables": [table.to_dict() for table in self.tables],
            "diagnostics": [
                {
                    "table": row.table,
                    "reason": row.reason,
                    "severity": row.severity,
                    "detail": row.detail,
                }
                for row in self.diagnostics
            ],
        }


class PublicationWriter:
    """Apply publication rows through an injected table client."""

    def __init__(self, client: PublicationTableClient) -> None:
        self._client = client

    def write(
        self,
        plan: PublicationWritePlan | Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        dry_run: bool = False,
    ) -> PublicationWriteResult:
        write_plan = (
            plan
            if isinstance(plan, PublicationWritePlan)
            else PublicationWritePlan.from_rows(plan)
        )
        diagnostics: list[PublicationWriteDiagnostic] = []
        table_results: list[PublicationTableWriteResult] = []
        batches = write_plan.batches_by_table()

        for table in write_plan.ordered_tables():
            rows = _copy_rows(batches[table].rows)
            mode = write_plan.mode_for(table)
            planned_count = len(rows)
            if planned_count == 0:
                continue

            if dry_run:
                diagnostics.append(
                    PublicationWriteDiagnostic(
                        table=table,
                        reason="dry_run",
                        detail=f"planned {mode} of {planned_count} rows",
                    )
                )
                table_results.append(
                    PublicationTableWriteResult(
                        table=table,
                        mode=mode,
                        planned_count=planned_count,
                        dry_run=True,
                    )
                )
                continue

            try:
                response = (
                    self._client.insert(table, rows)
                    if mode == "insert"
                    else self._client.upsert(table, rows)
                )
            except Exception as exc:  # pragma: no cover - detail covered by tests
                diagnostics.append(
                    PublicationWriteDiagnostic(
                        table=table,
                        reason="write_error",
                        severity="error",
                        detail=str(exc),
                    )
                )
                table_results.append(
                    PublicationTableWriteResult(
                        table=table,
                        mode=mode,
                        planned_count=planned_count,
                    )
                )
                break

            affected_count = _affected_count(response, planned_count)
            table_results.append(
                PublicationTableWriteResult(
                    table=table,
                    mode=mode,
                    planned_count=planned_count,
                    inserted_count=affected_count if mode == "insert" else 0,
                    upserted_count=affected_count if mode == "upsert" else 0,
                )
            )

        return PublicationWriteResult(
            tables=tuple(table_results),
            diagnostics=tuple(diagnostics),
            dry_run=dry_run,
        )


def write_publication_rows(
    rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    client: PublicationTableClient,
    table_modes: Mapping[str, WriteMode] | None = None,
    dry_run: bool = False,
    dependency_order: Sequence[str] = PUBLICATION_TABLE_ORDER,
) -> PublicationWriteResult:
    """Convenience wrapper for writing a table-to-rows mapping."""

    ordered_rows = {table: rows_by_table[table] for table in dependency_order if table in rows_by_table}
    ordered_rows.update(
        {
            table: rows_by_table[table]
            for table in sorted(rows_by_table)
            if table not in ordered_rows
        }
    )
    plan = PublicationWritePlan.from_rows(ordered_rows, table_modes=table_modes)
    return PublicationWriter(client).write(plan, dry_run=dry_run)


def _copy_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(row) for row in rows)


def _affected_count(response: Any, fallback: int) -> int:
    if isinstance(response, int):
        return response
    if isinstance(response, Sequence) and not isinstance(
        response, (str, bytes, bytearray)
    ):
        return len(response)
    if isinstance(response, Mapping):
        count = response.get("count")
        if isinstance(count, int):
            return count
        data = response.get("data")
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            return len(data)
    count = getattr(response, "count", None)
    if isinstance(count, int):
        return count
    data = getattr(response, "data", None)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return len(data)
    return fallback
