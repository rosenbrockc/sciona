"""PDG-derived CDG deployment storage composition.

This module is intentionally side-effect free by default. It composes the
existing PDG catalog projection rows with the production deployment storage
bundle and only writes when a caller explicitly opts into applying the bundle
through an injected client.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

from sciona.physics_ingest.deployment import (
    PhysicsIngestDeploymentStorageApplyResult,
    PhysicsIngestDeploymentStorageBundle,
    apply_physics_ingest_deployment_storage_bundle,
    build_physics_ingest_deployment_storage_bundle,
    preflight_physics_ingest_deployment_storage_bundle,
)
from sciona.physics_ingest.pdg_cdg import (
    PDGCDGCatalogWritePlanRows,
    PDGPublicationWriteRows,
    build_pdg_cdg_catalog_write_plan_rows,
)
from sciona.physics_ingest.write_plan import WriteMode


if TYPE_CHECKING:
    from sciona.physics_ingest.supabase_adapter import (
        PostgrestPublicationWritePreflight,
    )


JSONDict = dict[str, Any]
RowsByTable = Mapping[str, Iterable[Mapping[str, Any]]]


@dataclass(frozen=True)
class PDGDeploymentStoragePlan:
    """Catalog-aware PDG write rows wrapped in a deployment storage bundle."""

    catalog_write_plan_rows: PDGCDGCatalogWritePlanRows
    deployment_bundle: PhysicsIngestDeploymentStorageBundle
    preflight: "PostgrestPublicationWritePreflight | None" = None
    apply_result: PhysicsIngestDeploymentStorageApplyResult | None = None
    diagnostics: tuple[JSONDict, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        diagnostics = tuple(_json_safe_value(row) for row in self.diagnostics)
        object.__setattr__(self, "diagnostics", diagnostics)
        if self.summary:
            object.__setattr__(self, "summary", _json_safe_mapping(self.summary))
        else:
            object.__setattr__(
                self,
                "summary",
                _build_pdg_deployment_storage_summary(
                    catalog_write_plan_rows=self.catalog_write_plan_rows,
                    deployment_bundle=self.deployment_bundle,
                    preflight=self.preflight,
                    apply_result=self.apply_result,
                    diagnostics=diagnostics,
                    input_kind="unknown",
                    apply_requested=self.apply_result is not None,
                    dry_run=(
                        self.apply_result.write_result.dry_run
                        if self.apply_result is not None
                        else True
                    ),
                ),
            )

    def to_insert_rows(self) -> dict[str, list[JSONDict]]:
        """Return JSON-ready rows planned for production deployment storage."""

        return self.deployment_bundle.to_insert_rows()

    def to_dict(self) -> JSONDict:
        return {
            "catalog_write_plan_rows": self.catalog_write_plan_rows.to_dict(),
            "deployment_bundle": self.deployment_bundle.to_dict(),
            "preflight": None if self.preflight is None else self.preflight.to_dict(),
            "apply_result": None
            if self.apply_result is None
            else self.apply_result.to_dict(),
            "diagnostics": list(self.diagnostics),
            "summary": dict(self.summary),
        }


def build_pdg_deployment_storage_plan(
    rows: PDGPublicationWriteRows | RowsByTable,
    *,
    table_modes: Mapping[str, WriteMode] | None = None,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
    include_preflight: bool = True,
    apply_storage: bool = False,
    client: Any | None = None,
    dry_run: bool = True,
) -> PDGDeploymentStoragePlan:
    """Build and optionally apply a catalog-aware PDG deployment write plan.

    ``rows`` may be ``PDGPublicationWriteRows`` or a plain rows-by-table
    mapping. The function does not construct storage clients and defaults to
    ``dry_run=True`` when ``apply_storage`` is requested.
    """

    publication_rows = _publication_rows_value(rows)
    catalog_write_plan_rows = build_pdg_cdg_catalog_write_plan_rows(
        publication_rows,
        include_write_plan=True,
        table_modes=table_modes,
    )
    deployment_bundle = build_physics_ingest_deployment_storage_bundle(
        publication_rows,
        pdg_catalog_rows={
            "insert_rows": catalog_write_plan_rows.catalog_projection_rows.to_projection_rows(),
            "summary": dict(catalog_write_plan_rows.catalog_projection_rows.summary),
            "diagnostics": tuple(catalog_write_plan_rows.catalog_projection_rows.diagnostics),
        },
        table_modes=table_modes,
    )
    preflight = (
        preflight_physics_ingest_deployment_storage_bundle(
            deployment_bundle,
            conflict_keys_by_table=conflict_keys_by_table,
        )
        if include_preflight
        else None
    )
    apply_result = (
        apply_physics_ingest_deployment_storage_bundle(
            deployment_bundle,
            client,
            dry_run=dry_run,
            conflict_keys_by_table=conflict_keys_by_table,
        )
        if apply_storage
        else None
    )
    if preflight is None and apply_result is not None:
        preflight = apply_result.preflight

    diagnostics = tuple(catalog_write_plan_rows.diagnostics)
    summary = _build_pdg_deployment_storage_summary(
        catalog_write_plan_rows=catalog_write_plan_rows,
        deployment_bundle=deployment_bundle,
        preflight=preflight,
        apply_result=apply_result,
        diagnostics=diagnostics,
        input_kind=(
            "pdg_publication_write_rows"
            if isinstance(rows, PDGPublicationWriteRows)
            else "rows_by_table"
        ),
        apply_requested=apply_storage,
        dry_run=dry_run,
    )
    return PDGDeploymentStoragePlan(
        catalog_write_plan_rows=catalog_write_plan_rows,
        deployment_bundle=deployment_bundle,
        preflight=preflight,
        apply_result=apply_result,
        diagnostics=diagnostics,
        summary=summary,
    )


def _publication_rows_value(
    rows: PDGPublicationWriteRows | RowsByTable,
) -> PDGPublicationWriteRows | dict[str, list[JSONDict]]:
    if isinstance(rows, PDGPublicationWriteRows):
        return rows
    if not isinstance(rows, Mapping):
        raise ValueError("PDG deployment rows must be PDGPublicationWriteRows or rows by table")
    return {
        str(table): [_json_safe_mapping(dict(row)) for row in table_rows]
        for table, table_rows in rows.items()
    }


def _build_pdg_deployment_storage_summary(
    *,
    catalog_write_plan_rows: PDGCDGCatalogWritePlanRows,
    deployment_bundle: PhysicsIngestDeploymentStorageBundle,
    preflight: "PostgrestPublicationWritePreflight | None",
    apply_result: PhysicsIngestDeploymentStorageApplyResult | None,
    diagnostics: Sequence[Mapping[str, Any]],
    input_kind: str,
    apply_requested: bool,
    dry_run: bool,
) -> JSONDict:
    preflight_summary = (
        None
        if preflight is None
        else {
            "table_count": preflight.table_count,
            "total_row_count": preflight.total_row_count,
            "missing_conflict_metadata_for_upserts": list(
                preflight.missing_conflict_metadata_for_upserts
            ),
        }
    )
    return _json_safe_mapping(
        {
            "summary_kind": "pdg_deployment_storage_plan.v1",
            "input_kind": input_kind,
            "apply_requested": apply_requested,
            "dry_run": dry_run,
            "wrote": apply_result is not None and not apply_result.write_result.dry_run,
            "catalog_projection_summary": dict(
                catalog_write_plan_rows.catalog_projection_rows.summary
            ),
            "catalog_write_plan_summary": dict(catalog_write_plan_rows.summary),
            "deployment_bundle_summary": dict(deployment_bundle.summary),
            "write_plan_table_order": list(deployment_bundle.write_plan.ordered_tables()),
            "catalog_write_plan_table_order": list(
                catalog_write_plan_rows.write_plan.ordered_tables()
                if catalog_write_plan_rows.write_plan is not None
                else ()
            ),
            "table_row_counts": dict(
                deployment_bundle.write_plan.audit_summary.planned_row_counts
            ),
            "total_row_count": deployment_bundle.write_plan.audit_summary.total_row_count,
            "preflight": preflight_summary,
            "apply_summary": None
            if apply_result is None
            else dict(apply_result.summary),
            "diagnostic_count": len(diagnostics),
            "diagnostics_by_reason": _diagnostics_by_reason(diagnostics),
        }
    )


def _diagnostics_by_reason(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        reason = str(diagnostic.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _json_safe_mapping(value: Mapping[str, Any]) -> JSONDict:
    return {
        str(key): _json_safe_value(item)
        for key, item in value.items()
        if item is not None
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
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


__all__ = [
    "PDGDeploymentStoragePlan",
    "build_pdg_deployment_storage_plan",
]
