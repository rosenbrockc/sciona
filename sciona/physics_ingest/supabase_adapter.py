"""PostgREST-style client adapter for physics publication writes.

The publication writer depends on a tiny table-client protocol and deliberately
does not import Supabase. This adapter keeps that boundary intact by wrapping an
already-created client with ``.table(name).insert(...).execute()`` and
``.table(name).upsert(...).execute()`` methods.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sciona.physics_ingest.write_plan import (
    CONFLICT_KEYS_BY_TABLE,
    PublicationWritePlan,
)


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


def _copy_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(dict(row) for row in rows)
