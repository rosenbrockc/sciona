"""High-level publication pipeline for physics ingestion rows.

The pipeline composes the lower-level, side-effect-free publication steps and
optionally applies the resulting write plan through an injected table client.
It intentionally does not import Supabase or create network clients.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from sciona.physics_ingest.ids import plan_source_bundle_ids
from sciona.physics_ingest.orchestration import (
    PublicationOrchestrationResult,
    orchestrate_physics_publication,
)
from sciona.physics_ingest.publication import ArtifactBinding
from sciona.physics_ingest.write_plan import (
    PublicationWritePlan,
    WriteMode,
    build_publication_write_plan,
    merge_publication_insert_rows,
)
from sciona.physics_ingest.writer import (
    PublicationTableClient,
    PublicationWriteResult,
    write_publication_rows,
)


@dataclass(frozen=True)
class PublicationPipelineSummary:
    """Compact accounting for a publication pipeline run."""

    dry_run: bool
    source_bundle_count: int
    publication_manifest_count: int
    snapshot_binding_count: int
    planned_batch_count: int
    planned_row_count: int
    wrote: bool
    affected_row_count: int = 0
    orchestration_error_count: int = 0
    write_error_count: int = 0

    @property
    def has_errors(self) -> bool:
        return self.orchestration_error_count > 0 or self.write_error_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "source_bundle_count": self.source_bundle_count,
            "publication_manifest_count": self.publication_manifest_count,
            "snapshot_binding_count": self.snapshot_binding_count,
            "planned_batch_count": self.planned_batch_count,
            "planned_row_count": self.planned_row_count,
            "wrote": self.wrote,
            "affected_row_count": self.affected_row_count,
            "orchestration_error_count": self.orchestration_error_count,
            "write_error_count": self.write_error_count,
            "has_errors": self.has_errors,
        }


@dataclass(frozen=True)
class PublicationPipelineResult:
    """All intermediate artifacts from a publication pipeline run."""

    snapshot_id_bindings: Mapping[str, str]
    planned_source_bundles: tuple[dict[str, Any], ...]
    orchestration_result: PublicationOrchestrationResult
    write_plan: PublicationWritePlan
    write_result: PublicationWriteResult | None
    diagnostics: tuple[dict[str, Any], ...]
    summary: PublicationPipelineSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id_bindings": dict(self.snapshot_id_bindings),
            "planned_source_bundles": [dict(bundle) for bundle in self.planned_source_bundles],
            "orchestration_result": {
                "insert_rows_by_table": self.orchestration_result.to_insert_rows(),
                "audit_summary": self.orchestration_result.audit_summary.to_dict(),
            },
            "write_plan": self.write_plan.to_dict(),
            "write_result": None
            if self.write_result is None
            else self.write_result.to_dict(),
            "diagnostics": [dict(row) for row in self.diagnostics],
            "summary": self.summary.to_dict(),
        }


def run_physics_publication_pipeline(
    *,
    source_bundles: Iterable[Any] = (),
    publication_manifests: Iterable[Mapping[str, Any]] = (),
    additional_insert_rows: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    additional_diagnostics: Iterable[Mapping[str, Any]] = (),
    artifact_bindings: Mapping[str, Mapping[str, Any] | ArtifactBinding] | None = None,
    client: PublicationTableClient | None = None,
    table_modes: Mapping[str, WriteMode] | None = None,
    dry_run: bool = True,
) -> PublicationPipelineResult:
    """Plan, orchestrate, and optionally execute physics publication writes.

    ``source_bundles`` receive deterministic snapshot and candidate IDs before
    orchestration. A write is executed only when ``client`` is supplied or
    ``dry_run`` is true. Dry runs use an internal inert client because the writer
    does not touch the client while ``dry_run=True``.
    """

    source_bundle_list = tuple(source_bundles)
    publication_manifest_list = tuple(publication_manifests)
    snapshot_id_bindings, planned_bundles = plan_source_bundle_ids(source_bundle_list)
    deterministic_bindings = dict(sorted(snapshot_id_bindings.items()))

    orchestration_result = orchestrate_physics_publication(
        source_bundles=planned_bundles,
        publication_manifests=publication_manifest_list,
        artifact_bindings=artifact_bindings or {},
        snapshot_id_bindings=deterministic_bindings,
    )
    insert_rows = merge_publication_insert_rows(
        orchestration_result.insert_rows_by_table,
        additional_insert_rows,
    )
    write_plan = build_publication_write_plan(
        insert_rows,
        table_modes=table_modes,
    )

    write_result: PublicationWriteResult | None = None
    if dry_run or client is not None:
        write_result = write_publication_rows(
            write_plan.to_insert_rows(),
            client=client if client is not None else _DryRunOnlyPublicationClient(),
            table_modes=table_modes,
            dry_run=dry_run,
        )

    diagnostics = _diagnostics(
        orchestration_result,
        write_result,
        additional_diagnostics=additional_diagnostics,
    )
    summary = PublicationPipelineSummary(
        dry_run=dry_run,
        source_bundle_count=len(planned_bundles),
        publication_manifest_count=len(publication_manifest_list),
        snapshot_binding_count=len(deterministic_bindings),
        planned_batch_count=write_plan.audit_summary.batch_count,
        planned_row_count=write_plan.audit_summary.total_row_count,
        wrote=write_result is not None,
        affected_row_count=0 if write_result is None else write_result.affected_count,
        orchestration_error_count=orchestration_result.audit_summary.error_row_count,
        write_error_count=0
        if write_result is None
        else sum(1 for row in write_result.diagnostics if row.severity == "error"),
    )
    return PublicationPipelineResult(
        snapshot_id_bindings=deterministic_bindings,
        planned_source_bundles=tuple(dict(bundle) for bundle in planned_bundles),
        orchestration_result=orchestration_result,
        write_plan=write_plan,
        write_result=write_result,
        diagnostics=diagnostics,
        summary=summary,
    )


class _DryRunOnlyPublicationClient:
    def insert(self, table: str, rows: object) -> object:
        raise RuntimeError(f"dry-run pipeline attempted insert into {table}")

    def upsert(self, table: str, rows: object) -> object:
        raise RuntimeError(f"dry-run pipeline attempted upsert into {table}")


def _diagnostics(
    orchestration_result: PublicationOrchestrationResult,
    write_result: PublicationWriteResult | None,
    *,
    additional_diagnostics: Iterable[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = [
        {
            "stage": "orchestration",
            "table": row.table,
            "reason": row.reason,
            "severity": row.severity,
            "artifact_key": row.artifact_key,
            "atom_name": row.atom_name,
            "detail": row.detail,
        }
        for row in orchestration_result.diagnostics
    ]
    if write_result is not None:
        rows.extend(
            {
                "stage": "write",
                "table": row.table,
                "reason": row.reason,
                "severity": row.severity,
                "detail": row.detail,
            }
            for row in write_result.diagnostics
        )
    rows.extend(
        {
            "stage": str(row.get("stage") or "publication_extension"),
            "table": str(row.get("table") or ""),
            "reason": str(row.get("reason") or ""),
            "severity": str(row.get("severity") or "info"),
            "artifact_key": str(row.get("artifact_key") or ""),
            "atom_name": str(row.get("atom_name") or ""),
            "detail": str(row.get("detail") or ""),
        }
        for row in additional_diagnostics
    )
    return tuple(rows)
