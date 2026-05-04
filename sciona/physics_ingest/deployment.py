"""Side-effect-free deployment storage composition for physics ingestion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Protocol

from sciona.physics_ingest.supabase_adapter import (
    PostgrestPublicationWritePreflight,
    preflight_publication_supabase_write,
)
from sciona.physics_ingest.write_plan import (
    PublicationWritePlan,
    WriteMode,
    build_publication_write_plan,
    merge_publication_insert_rows,
)


JSONDict = dict[str, Any]
RowsByTable = Mapping[str, Iterable[Mapping[str, Any]]]


class _InsertRowsProvider(Protocol):
    def to_insert_rows(self) -> RowsByTable:
        """Return rows grouped by destination table."""


@dataclass(frozen=True)
class PhysicsIngestDeploymentStorageComponent:
    """Rows contributed by one deployment storage family."""

    name: str
    rows_by_table: Mapping[str, tuple[JSONDict, ...]]
    summary: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    @property
    def table_row_counts(self) -> dict[str, int]:
        return {table: len(rows) for table, rows in self.rows_by_table.items()}

    @property
    def total_row_count(self) -> int:
        return sum(self.table_row_counts.values())

    def to_insert_rows(self) -> dict[str, list[JSONDict]]:
        return {
            table: [dict(row) for row in rows]
            for table, rows in self.rows_by_table.items()
            if rows
        }

    def to_dict(self) -> JSONDict:
        return {
            "name": self.name,
            "insert_rows": self.to_insert_rows(),
            "summary": dict(self.summary),
            "diagnostics": [dict(row) for row in self.diagnostics],
            "table_row_counts": self.table_row_counts,
            "total_row_count": self.total_row_count,
        }


@dataclass(frozen=True)
class PhysicsIngestDeploymentStorageBundle:
    """Merged production storage rows and inert write plan for deployment."""

    insert_rows_by_table: Mapping[str, tuple[JSONDict, ...]]
    write_plan: PublicationWritePlan
    components: tuple[PhysicsIngestDeploymentStorageComponent, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    def to_insert_rows(self) -> dict[str, list[JSONDict]]:
        return {
            table: [dict(row) for row in rows]
            for table, rows in self.insert_rows_by_table.items()
            if rows
        }

    def to_dict(self) -> JSONDict:
        return {
            "insert_rows": self.to_insert_rows(),
            "summary": dict(self.summary),
            "components": [component.to_dict() for component in self.components],
            "write_plan": self.write_plan.to_dict(),
        }

    def preflight_supabase_write(
        self,
        *,
        conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
    ) -> PostgrestPublicationWritePreflight:
        """Return an injected-client Supabase/PostgREST preflight report."""

        return preflight_publication_supabase_write(
            self.write_plan,
            conflict_keys_by_table=conflict_keys_by_table,
        )


def build_physics_ingest_deployment_storage_bundle(
    publication_rows: RowsByTable | _InsertRowsProvider | PublicationWritePlan,
    *,
    pdg_catalog_rows: RowsByTable | _InsertRowsProvider | None = None,
    review_queue_rows: RowsByTable | _InsertRowsProvider | None = None,
    audit_artifact_rows: RowsByTable | _InsertRowsProvider | None = None,
    table_modes: Mapping[str, WriteMode] | None = None,
) -> PhysicsIngestDeploymentStorageBundle:
    """Compose publication storage row families into one inert write bundle."""

    components = tuple(
        component
        for component in (
            _deployment_component("publication", publication_rows),
            _deployment_component("pdg_catalog", pdg_catalog_rows),
            _deployment_component("review_queue", review_queue_rows),
            _deployment_component("audit_artifacts", audit_artifact_rows),
        )
        if component is not None
    )
    merged_rows = merge_publication_insert_rows(
        *(component.to_insert_rows() for component in components)
    )
    safe_rows = {
        table: [_json_safe_value(row) for row in rows]
        for table, rows in merged_rows.items()
    }
    write_plan = build_publication_write_plan(safe_rows, table_modes=table_modes)
    insert_rows_by_table = {
        table: tuple(rows)
        for table, rows in safe_rows.items()
        if rows
    }
    summary = _deployment_summary(
        components=components,
        write_plan=write_plan,
    )
    return PhysicsIngestDeploymentStorageBundle(
        insert_rows_by_table=insert_rows_by_table,
        write_plan=write_plan,
        components=components,
        summary=summary,
    )


def preflight_physics_ingest_deployment_storage_bundle(
    bundle: PhysicsIngestDeploymentStorageBundle,
    *,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
) -> PostgrestPublicationWritePreflight:
    """Preflight a composed deployment bundle without constructing DB clients."""

    return bundle.preflight_supabase_write(
        conflict_keys_by_table=conflict_keys_by_table,
    )


def _deployment_component(
    name: str,
    value: RowsByTable | _InsertRowsProvider | PublicationWritePlan | None,
) -> PhysicsIngestDeploymentStorageComponent | None:
    if value is None:
        return None
    rows_by_table, summary, diagnostics = _extract_rows_summary_diagnostics(value)
    safe_rows = {
        table: tuple(_json_safe_value(row) for row in rows)
        for table, rows in merge_publication_insert_rows(rows_by_table).items()
        if rows
    }
    return PhysicsIngestDeploymentStorageComponent(
        name=name,
        rows_by_table=safe_rows,
        summary=summary,
        diagnostics=diagnostics,
    )


def _extract_rows_summary_diagnostics(
    value: RowsByTable | _InsertRowsProvider | PublicationWritePlan,
) -> tuple[RowsByTable, Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    if isinstance(value, PublicationWritePlan):
        return (
            value.to_insert_rows(),
            value.audit_summary.to_dict(),
            (),
        )
    if isinstance(value, Mapping):
        if isinstance(value.get("insert_rows"), Mapping):
            return (
                value["insert_rows"],  # type: ignore[index,return-value]
                _summary_from_mapping(value),
                _diagnostics_from_mapping(value),
            )
        return value, {}, ()
    to_insert_rows = getattr(value, "to_insert_rows", None)
    if callable(to_insert_rows):
        rows = to_insert_rows()
        return rows, _summary_from_object(value), _diagnostics_from_object(value)
    raise ValueError("deployment storage rows must be a mapping or to_insert_rows result")


def _summary_from_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = value.get("summary")
    return _json_safe_value(summary) if isinstance(summary, Mapping) else {}


def _diagnostics_from_mapping(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    diagnostics = value.get("diagnostics")
    if isinstance(diagnostics, Iterable) and not isinstance(
        diagnostics,
        (str, bytes, Mapping),
    ):
        return tuple(
            row
            for row in (_json_safe_value(diagnostic) for diagnostic in diagnostics)
            if isinstance(row, Mapping)
        )
    return ()


def _summary_from_object(value: Any) -> Mapping[str, Any]:
    summary = getattr(value, "summary", None)
    if isinstance(summary, Mapping):
        return _json_safe_value(summary)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _summary_from_mapping(to_dict())
    return {}


def _diagnostics_from_object(value: Any) -> tuple[Mapping[str, Any], ...]:
    diagnostics = getattr(value, "diagnostics", None)
    if diagnostics is not None:
        return _diagnostics_from_mapping({"diagnostics": diagnostics})
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _diagnostics_from_mapping(to_dict())
    return ()


def _deployment_summary(
    *,
    components: Sequence[PhysicsIngestDeploymentStorageComponent],
    write_plan: PublicationWritePlan,
) -> JSONDict:
    component_table_counts = {
        component.name: component.table_row_counts for component in components
    }
    conflict_keys = {
        batch.table: list(batch.conflict_keys)
        for batch in write_plan.batches
        if batch.conflict_keys
    }
    return {
        "summary_kind": "physics_ingest_deployment_storage_bundle.v1",
        "component_order": [component.name for component in components],
        "component_count": len(components),
        "component_table_row_counts": component_table_counts,
        "merged_table_row_counts": dict(write_plan.audit_summary.planned_row_counts),
        "total_row_count": write_plan.audit_summary.total_row_count,
        "write_plan_table_order": list(write_plan.ordered_tables()),
        "write_plan_table_modes": {
            table: write_plan.mode_for(table) for table in write_plan.ordered_tables()
        },
        "write_plan_conflict_keys": conflict_keys,
        "missing_conflict_metadata_for_upserts": [
            batch.table
            for batch in write_plan.batches
            if write_plan.mode_for(batch.table) == "upsert" and not batch.conflict_keys
        ],
        "diagnostic_count": sum(len(component.diagnostics) for component in components),
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_safe_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str | int | bool):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe_value(to_dict())
    return str(value)
