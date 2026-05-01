"""CLI-oriented dry-run helpers for physics ingestion publication.

The functions here intentionally avoid database clients and argument parsing.
Callers can decode JSON/YAML however they prefer, pass ordinary Python
structures in, and serialize the returned report directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
from typing import Any

from sciona.physics_ingest.ids import DeterministicIdError, plan_source_bundle_ids
from sciona.physics_ingest.orchestration import orchestrate_physics_publication
from sciona.physics_ingest.publication import ArtifactBinding, PublicationDiagnostic
from sciona.physics_ingest.write_plan import WriteMode, build_publication_write_plan


REPORT_KIND = "physics_ingest_publication_dry_run"


def build_publication_dry_run_report_from_payload(
    payload: Mapping[str, Any],
    *,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Build a dry-run report from a decoded CLI payload.

    Expected payload keys mirror :func:`build_publication_dry_run_report`:
    ``source_bundles``, ``publication_manifests``, ``artifact_bindings``,
    ``snapshot_id_bindings``, ``table_modes``, and ``plan_ids``.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")

    return build_publication_dry_run_report(
        source_bundles=_sequence(payload.get("source_bundles", ()), "source_bundles"),
        publication_manifests=_sequence(
            payload.get("publication_manifests", ()),
            "publication_manifests",
        ),
        artifact_bindings=_mapping(payload.get("artifact_bindings", {}), "artifact_bindings"),
        snapshot_id_bindings=_string_mapping(
            payload.get("snapshot_id_bindings", {}),
            "snapshot_id_bindings",
        ),
        table_modes=_table_modes(payload.get("table_modes", {})),
        plan_ids=bool(payload.get("plan_ids", True)),
        include_rows=include_rows,
    )


def build_publication_dry_run_report(
    *,
    source_bundles: Iterable[Any] = (),
    publication_manifests: Iterable[Mapping[str, Any]] = (),
    artifact_bindings: Mapping[str, Mapping[str, Any] | ArtifactBinding] | None = None,
    snapshot_id_bindings: Mapping[str, str] | None = None,
    table_modes: Mapping[str, WriteMode] | None = None,
    plan_ids: bool = True,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable dry-run report for publication inputs.

    The helper composes the side-effect-free publication pipeline:
    deterministic source IDs, publication orchestration, and dependency-ordered
    write planning. If deterministic ID planning fails, the report keeps the
    error as a diagnostic and continues with caller-provided snapshot bindings
    so validation can surface additional row-level problems.
    """

    source_bundle_list = list(source_bundles)
    publication_manifest_list = list(publication_manifests)
    diagnostics: list[PublicationDiagnostic] = []
    planned_source_bundles = source_bundle_list
    planned_snapshot_bindings = dict(snapshot_id_bindings or {})
    id_strategy = "provided_or_existing"

    if plan_ids and source_bundle_list:
        try:
            deterministic_bindings, planned_source_bundles = plan_source_bundle_ids(
                source_bundle_list,
            )
        except DeterministicIdError as exc:
            id_strategy = "deterministic_failed"
            diagnostics.append(
                PublicationDiagnostic(
                    table="physics_ingest_snapshots",
                    reason="deterministic_id_error",
                    severity="error",
                    detail=str(exc),
                )
            )
        else:
            id_strategy = "deterministic"
            planned_snapshot_bindings = {
                **deterministic_bindings,
                **planned_snapshot_bindings,
            }

    orchestration = orchestrate_physics_publication(
        source_bundles=planned_source_bundles,
        publication_manifests=publication_manifest_list,
        artifact_bindings=artifact_bindings or {},
        snapshot_id_bindings=planned_snapshot_bindings,
    )
    diagnostics.extend(orchestration.diagnostics)

    insert_rows = orchestration.to_insert_rows()
    write_plan = build_publication_write_plan(insert_rows, table_modes=table_modes)
    diagnostic_dicts = [_diagnostic_to_dict(row) for row in diagnostics]
    has_errors = any(row["severity"] == "error" for row in diagnostic_dicts)
    report: dict[str, Any] = {
        "report_kind": REPORT_KIND,
        "dry_run": True,
        "ok": not has_errors,
        "source_bundle_count": len(source_bundle_list),
        "publication_manifest_count": len(publication_manifest_list),
        "id_planning": {
            "enabled": plan_ids,
            "strategy": id_strategy,
            "snapshot_binding_count": len(planned_snapshot_bindings),
        },
        "audit_summary": orchestration.audit_summary.to_dict(),
        "write_plan": {
            "audit_summary": write_plan.audit_summary.to_dict(),
            "batches": [
                {
                    "table": batch.table,
                    "mode": write_plan.mode_for(batch.table),
                    "row_count": batch.row_count,
                    "conflict_keys": list(batch.conflict_keys),
                    "dry_run": True,
                }
                for batch in write_plan.batches
            ],
        },
        "diagnostics": diagnostic_dicts,
    }
    if include_rows:
        report["insert_rows_by_table"] = insert_rows

    _assert_json_serializable(report)
    return report


def _diagnostic_to_dict(row: PublicationDiagnostic) -> dict[str, Any]:
    return {
        "table": row.table,
        "reason": row.reason,
        "artifact_key": row.artifact_key,
        "atom_name": row.atom_name,
        "severity": row.severity,
        "detail": row.detail,
    }


def _assert_json_serializable(report: Mapping[str, Any]) -> None:
    try:
        json.dumps(report, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("dry-run report must be JSON serializable") from exc


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if value in (None, ""):
        return ()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _string_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    mapping = _mapping(value, field_name)
    return {str(key): str(item) for key, item in mapping.items()}


def _table_modes(value: Any) -> Mapping[str, WriteMode]:
    mapping = _mapping(value, "table_modes")
    modes: dict[str, WriteMode] = {}
    for table, mode in mapping.items():
        if mode not in ("insert", "upsert"):
            raise ValueError(f"unsupported write mode for {table}: {mode}")
        modes[str(table)] = mode
    return modes
