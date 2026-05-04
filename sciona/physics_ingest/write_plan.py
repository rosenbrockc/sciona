"""Side-effect-free write planning for physics ingestion publication rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


WriteMode = Literal["insert", "upsert"]


PUBLICATION_TABLE_ORDER = (
    "physics_ingest_snapshots",
    "physics_equation_candidates",
    "artifacts",
    "artifact_versions",
    "artifact_symbolic_expressions",
    "artifact_symbolic_variables",
    "artifact_validity_bounds",
    "artifact_relationships",
    "artifact_cdg_nodes",
    "artifact_cdg_edges",
    "artifact_cdg_bindings",
    "catalog_cdg_artifacts",
    "catalog_cdg_versions",
    "catalog_cdg_nodes",
    "catalog_cdg_relationships",
    "catalog_symbolic_artifacts",
    "physics_review_queue_tasks",
    "physics_ingest_audit_artifacts",
)

CONFLICT_KEYS_BY_TABLE: Mapping[str, tuple[str, ...]] = {
    "physics_ingest_snapshots": ("snapshot_id",),
    "physics_equation_candidates": ("candidate_id",),
    "artifacts": ("artifact_id",),
    "artifact_versions": ("version_id",),
    "artifact_symbolic_expressions": ("expression_id",),
    "artifact_symbolic_variables": ("variable_id",),
    "artifact_validity_bounds": ("bound_id",),
    "artifact_relationships": ("relationship_id",),
    "artifact_cdg_nodes": ("version_id", "node_id"),
    "artifact_cdg_edges": (
        "version_id",
        "source_id",
        "target_id",
        "output_name",
        "input_name",
    ),
    "artifact_cdg_bindings": ("version_id", "node_id", "bound_artifact_fqdn"),
    "catalog_cdg_artifacts": ("artifact_id", "version_id", "projection_kind"),
    "catalog_cdg_versions": ("version_id", "projection_kind"),
    "catalog_cdg_nodes": ("version_id", "node_id", "projection_kind"),
    "catalog_cdg_relationships": ("relationship_id", "projection_kind"),
    "catalog_symbolic_artifacts": ("artifact_id", "version_id", "projection_kind"),
    "physics_review_queue_tasks": ("task_id",),
    "physics_ingest_audit_artifacts": ("artifact_key",),
}


@dataclass(frozen=True)
class PublicationWriteBatch:
    """Rows for one table plus loader metadata; no DB operation is performed."""

    table: str
    rows: tuple[dict[str, Any], ...]
    conflict_keys: tuple[str, ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "rows": [dict(row) for row in self.rows],
            "conflict_keys": list(self.conflict_keys),
            "row_count": self.row_count,
        }


@dataclass(frozen=True)
class PublicationWritePlanAuditSummary:
    """Compact counts for a side-effect-free publication write plan."""

    input_row_counts: Mapping[str, int] = field(default_factory=dict)
    planned_row_counts: Mapping[str, int] = field(default_factory=dict)
    table_order: tuple[str, ...] = ()
    batch_count: int = 0
    total_row_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_row_counts": dict(self.input_row_counts),
            "planned_row_counts": dict(self.planned_row_counts),
            "table_order": list(self.table_order),
            "batch_count": self.batch_count,
            "total_row_count": self.total_row_count,
        }


@dataclass(frozen=True)
class PublicationWritePlan:
    """Ordered publication batches ready for a caller-owned DB writer."""

    batches: tuple[PublicationWriteBatch, ...]
    audit_summary: PublicationWritePlanAuditSummary
    table_modes: Mapping[str, WriteMode] = field(default_factory=dict)

    def batches_by_table(self) -> dict[str, PublicationWriteBatch]:
        return {batch.table: batch for batch in self.batches}

    def ordered_tables(self) -> tuple[str, ...]:
        return tuple(batch.table for batch in self.batches)

    def mode_for(self, table: str) -> WriteMode:
        mode = self.table_modes.get(table, "insert")
        if mode not in ("insert", "upsert"):
            raise ValueError(f"unsupported write mode for {table}: {mode}")
        return mode

    def to_insert_rows(self) -> dict[str, list[dict[str, Any]]]:
        return {
            batch.table: [dict(row) for row in batch.rows]
            for batch in self.batches
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "batches": [batch.to_dict() for batch in self.batches],
            "audit_summary": self.audit_summary.to_dict(),
        }

    @classmethod
    def from_rows(
        cls,
        insert_rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
        *,
        table_modes: Mapping[str, WriteMode] | None = None,
    ) -> "PublicationWritePlan":
        plan = build_publication_write_plan(insert_rows_by_table)
        return cls(
            batches=plan.batches,
            audit_summary=plan.audit_summary,
            table_modes=dict(table_modes or {}),
        )


def build_publication_write_plan(
    insert_rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    table_modes: Mapping[str, WriteMode] | None = None,
) -> PublicationWritePlan:
    """Build ordered write batches from orchestrated insert rows.

    The returned plan is intentionally inert: it only copies, orders, and
    summarizes rows supplied by orchestration. It does not validate schema
    contracts and does not perform database IO.
    """

    normalized = _normalize_insert_rows_by_table(insert_rows_by_table)
    ordered_tables = _ordered_tables(normalized)
    batches = tuple(
        PublicationWriteBatch(
            table=table,
            rows=tuple(normalized[table]),
            conflict_keys=CONFLICT_KEYS_BY_TABLE.get(table, ()),
        )
        for table in ordered_tables
        if normalized[table]
    )
    planned_counts = {batch.table: batch.row_count for batch in batches}
    summary = PublicationWritePlanAuditSummary(
        input_row_counts={table: len(rows) for table, rows in normalized.items()},
        planned_row_counts=planned_counts,
        table_order=tuple(batch.table for batch in batches),
        batch_count=len(batches),
        total_row_count=sum(planned_counts.values()),
    )
    return PublicationWritePlan(
        batches=batches,
        audit_summary=summary,
        table_modes=dict(table_modes or {}),
    )


def merge_publication_insert_rows(
    *rows_by_table_values: Mapping[str, Iterable[Mapping[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Copy and concatenate publication rows from multiple producers."""

    merged: dict[str, list[dict[str, Any]]] = {}
    for rows_by_table in rows_by_table_values:
        if rows_by_table is None:
            continue
        normalized = _normalize_insert_rows_by_table(rows_by_table)
        for table, rows in normalized.items():
            merged.setdefault(table, []).extend(dict(row) for row in rows)
    return merged


def _normalize_insert_rows_by_table(
    insert_rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    if not isinstance(insert_rows_by_table, Mapping):
        raise ValueError("insert_rows_by_table must be a mapping")

    normalized: dict[str, tuple[dict[str, Any], ...]] = {}
    for table, rows in insert_rows_by_table.items():
        if not isinstance(table, str) or not table:
            raise ValueError("insert_rows_by_table table names must be non-empty strings")
        if isinstance(rows, Mapping) or isinstance(rows, (str, bytes)) or rows is None:
            raise ValueError(f"{table} rows must be an iterable of mappings")
        copied_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"{table} row {index} must be a mapping")
            copied_rows.append(dict(row))
        normalized[table] = tuple(copied_rows)
    return normalized


def _ordered_tables(
    rows_by_table: Mapping[str, tuple[dict[str, Any], ...]],
) -> tuple[str, ...]:
    known_tables = [table for table in PUBLICATION_TABLE_ORDER if table in rows_by_table]
    unknown_tables = sorted(
        table for table in rows_by_table if table not in PUBLICATION_TABLE_ORDER
    )
    return tuple([*known_tables, *unknown_tables])
