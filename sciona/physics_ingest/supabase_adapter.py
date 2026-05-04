"""PostgREST-style client adapter for physics publication writes.

The publication writer depends on a tiny table-client protocol and deliberately
does not import Supabase. This adapter keeps that boundary intact by wrapping an
already-created client with ``.table(name).insert(...).execute()`` and
``.table(name).upsert(...).execute()`` methods.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sciona.physics_ingest.write_plan import (
    CONFLICT_KEYS_BY_TABLE,
    PublicationWritePlan,
    WriteMode,
)
from sciona.physics_ingest.writer import PublicationWriteResult, write_publication_rows


POSTGREST_PUBLICATION_ADAPTER_CAPABILITIES: Mapping[str, bool] = {
    "imports_supabase": False,
    "requires_injected_client": True,
    "writes_during_preflight": False,
    "supports_insert": True,
    "supports_upsert": True,
    "supports_upsert_on_conflict": True,
}


@dataclass(frozen=True)
class PostgrestPublicationWritePreflightTable:
    """Side-effect-free write metadata for one planned PostgREST table call."""

    table: str
    mode: WriteMode
    row_count: int
    conflict_keys: tuple[str, ...] = ()
    missing_conflict_metadata: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "mode": self.mode,
            "row_count": self.row_count,
            "conflict_keys": list(self.conflict_keys),
            "missing_conflict_metadata": self.missing_conflict_metadata,
        }


@dataclass(frozen=True)
class PostgrestPublicationWritePreflight:
    """Side-effect-free report for the publication PostgREST adapter boundary."""

    tables: tuple[PostgrestPublicationWritePreflightTable, ...]
    adapter_capabilities: Mapping[str, bool] = field(
        default_factory=lambda: dict(POSTGREST_PUBLICATION_ADAPTER_CAPABILITIES)
    )

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def total_row_count(self) -> int:
        return sum(table.row_count for table in self.tables)

    @property
    def missing_conflict_metadata_for_upserts(self) -> tuple[str, ...]:
        return tuple(
            table.table for table in self.tables if table.missing_conflict_metadata
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_count": self.table_count,
            "total_row_count": self.total_row_count,
            "tables": [table.to_dict() for table in self.tables],
            "missing_conflict_metadata_for_upserts": list(
                self.missing_conflict_metadata_for_upserts
            ),
            "adapter_capabilities": dict(self.adapter_capabilities),
        }


class PostgrestPublicationTableClient:
    """Adapt an injected PostgREST/Supabase-style client to PublicationWriter."""

    def __init__(
        self,
        client: Any,
        *,
        write_plan: PublicationWritePlan | None = None,
        conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._client = client
        self._conflict_keys_by_table = _conflict_keys_by_table(
            write_plan=write_plan,
            conflict_keys_by_table=conflict_keys_by_table,
        )

    @classmethod
    def from_write_plan(
        cls,
        client: Any,
        write_plan: PublicationWritePlan,
        *,
        conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
    ) -> "PostgrestPublicationTableClient":
        """Build an adapter using conflict metadata carried by ``write_plan``."""

        return cls(
            client,
            write_plan=write_plan,
            conflict_keys_by_table=conflict_keys_by_table,
        )

    def insert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> Any:
        """Insert rows into ``table`` through the wrapped client."""

        return self._execute(self._client.table(table).insert(_copy_rows(rows)))

    def upsert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> Any:
        """Upsert rows into ``table`` using known conflict keys when available."""

        payload = _copy_rows(rows)
        conflict_keys = self._conflict_keys_by_table.get(table, ())
        query = (
            self._client.table(table).upsert(
                payload,
                on_conflict=",".join(conflict_keys),
            )
            if conflict_keys
            else self._client.table(table).upsert(payload)
        )
        return self._execute(query)

    @staticmethod
    def _execute(query: Any) -> Any:
        return query.execute()


def adapt_publication_supabase_client(
    client: Any,
    *,
    write_plan: PublicationWritePlan | None = None,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
) -> PostgrestPublicationTableClient:
    """Return a publication-writer client for an injected PostgREST client."""

    return PostgrestPublicationTableClient(
        client,
        write_plan=write_plan,
        conflict_keys_by_table=conflict_keys_by_table,
    )


def preflight_publication_postgrest_write(
    plan_or_rows: PublicationWritePlan | Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    table_modes: Mapping[str, WriteMode] | None = None,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
) -> PostgrestPublicationWritePreflight:
    """Report planned adapter calls without importing Supabase or writing rows."""

    write_plan = _write_plan_for_preflight(plan_or_rows, table_modes=table_modes)
    conflict_keys = _conflict_keys_by_table(
        write_plan=write_plan,
        conflict_keys_by_table=conflict_keys_by_table,
    )
    tables = tuple(
        PostgrestPublicationWritePreflightTable(
            table=batch.table,
            mode=write_plan.mode_for(batch.table),
            row_count=batch.row_count,
            conflict_keys=conflict_keys.get(batch.table, ()),
            missing_conflict_metadata=(
                write_plan.mode_for(batch.table) == "upsert"
                and not conflict_keys.get(batch.table)
            ),
        )
        for batch in write_plan.batches
    )
    return PostgrestPublicationWritePreflight(tables=tables)


def preflight_publication_supabase_write(
    plan_or_rows: PublicationWritePlan | Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    table_modes: Mapping[str, WriteMode] | None = None,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
) -> PostgrestPublicationWritePreflight:
    """Alias for the PostgREST preflight helper at the Supabase adapter boundary."""

    return preflight_publication_postgrest_write(
        plan_or_rows,
        table_modes=table_modes,
        conflict_keys_by_table=conflict_keys_by_table,
    )


def apply_publication_supabase_write(
    client: Any,
    plan_or_rows: PublicationWritePlan | Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    table_modes: Mapping[str, WriteMode] | None = None,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
    dry_run: bool = False,
) -> PublicationWriteResult:
    """Apply publication rows through an injected Supabase/PostgREST client.

    The helper keeps Supabase client construction outside this module. It
    performs the same side-effect-free preflight used by callers that want a
    report, adapts the injected client to the publication writer protocol, and
    delegates write accounting to the existing writer helper.
    """

    write_plan = _write_plan_for_preflight(plan_or_rows, table_modes=table_modes)
    preflight_publication_supabase_write(
        write_plan,
        conflict_keys_by_table=conflict_keys_by_table,
    )
    adapted_client = adapt_publication_supabase_client(
        client,
        write_plan=write_plan,
        conflict_keys_by_table=conflict_keys_by_table,
    )
    write_modes = {
        table: write_plan.mode_for(table)
        for table in write_plan.ordered_tables()
    }
    return write_publication_rows(
        write_plan.to_insert_rows(),
        client=adapted_client,
        table_modes=write_modes,
        dry_run=dry_run,
        dependency_order=write_plan.ordered_tables(),
    )


def _conflict_keys_by_table(
    *,
    write_plan: PublicationWritePlan | None,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    conflicts = {
        table: tuple(keys)
        for table, keys in CONFLICT_KEYS_BY_TABLE.items()
        if keys
    }
    if write_plan is not None:
        conflicts.update(
            {
                batch.table: tuple(batch.conflict_keys)
                for batch in write_plan.batches
                if batch.conflict_keys
            }
        )
    if conflict_keys_by_table is not None:
        conflicts.update(
            {
                table: tuple(keys)
                for table, keys in conflict_keys_by_table.items()
                if keys
            }
        )
    return conflicts


def _write_plan_for_preflight(
    plan_or_rows: PublicationWritePlan | Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    table_modes: Mapping[str, WriteMode] | None,
) -> PublicationWritePlan:
    if isinstance(plan_or_rows, PublicationWritePlan):
        if table_modes is None:
            return plan_or_rows
        return PublicationWritePlan(
            batches=plan_or_rows.batches,
            audit_summary=plan_or_rows.audit_summary,
            table_modes={**plan_or_rows.table_modes, **table_modes},
        )
    return PublicationWritePlan.from_rows(plan_or_rows, table_modes=table_modes)


def _copy_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(row) for row in rows)
