"""CLI-oriented dry-run helpers for physics ingestion publication.

The functions here intentionally avoid database clients and argument parsing.
Callers can decode JSON/YAML however they prefer, pass ordinary Python
structures in, and serialize the returned report directly.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

from sciona.physics_ingest.audit_artifacts import (
    build_backfill_audit_artifact_write_plan_rows,
)
from sciona.physics_ingest.backfill import build_physics_ingest_backfill_report
from sciona.physics_ingest.ids import DeterministicIdError, plan_source_bundle_ids
from sciona.physics_ingest.orchestration import orchestrate_physics_publication
from sciona.physics_ingest.publication import ArtifactBinding, PublicationDiagnostic
from sciona.physics_ingest.review import (
    REVIEW_QUEUE_TASKS_TABLE,
    build_review_queue_write_plan_rows,
)
from sciona.physics_ingest.sources.runtime_execution import (
    build_source_retrieval_runtime_execution_report_dict,
)
from sciona.physics_ingest.write_plan import WriteMode, build_publication_write_plan


REPORT_KIND = "physics_ingest_publication_dry_run"
COMPOSED_REPORT_KIND = "physics_ingest_publication_backfill_dry_run"


def main(argv: Sequence[str] | None = None) -> int:
    """Read a publication payload JSON file and print a dry-run report."""

    parser = argparse.ArgumentParser(
        description="Build a physics ingestion publication dry-run report.",
    )
    parser.add_argument(
        "payload_file",
        type=Path,
        help="Path to a JSON payload for the dry-run report.",
    )
    parser.add_argument(
        "--include-rows",
        action="store_true",
        help="Include planned insert rows in the printed report.",
    )
    args = parser.parse_args(argv)

    with args.payload_file.open(encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, Mapping) and _has_backfill_payload(payload):
        report = build_publication_backfill_dry_run_report_from_payload(
            payload,
            include_rows=args.include_rows,
        )
    else:
        report = build_publication_dry_run_report_from_payload(
            payload,
            include_rows=args.include_rows,
        )
    print(json.dumps(report, sort_keys=True), file=sys.stdout)
    return 0


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


def build_publication_backfill_dry_run_report_from_payload(
    payload: Mapping[str, Any],
    *,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Build a composed publication/backfill dry-run report from a JSON payload.

    This additive payload mode accepts the publication dry-run inputs plus an
    optional source retrieval run plan under either ``source_retrieval_run_plan``
    or the shorter ``retrieval_run_plan`` alias. It preserves the established
    publication report shape by embedding it unchanged under
    ``publication_dry_run_report``.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")

    retrieval_plan = _optional_retrieval_run_plan_payload(payload)
    table_modes = _table_modes(payload.get("table_modes", {}))
    include_execution_boundary_preflight = _payload_bool(
        payload,
        "include_execution_boundary_preflight",
        default=False,
    )
    include_audit_artifact_write_plan_rows = _payload_bool(
        payload,
        "include_audit_artifact_write_plan_rows",
        default=False,
    )
    include_review_queue_write_plan_rows = _payload_bool(
        payload,
        "include_review_queue_write_plan_rows",
        default=False,
    )
    source_runtime_execution_preflight = _source_runtime_execution_preflight_payload(
        payload
    )
    review_queue_rows = _review_queue_write_plan_rows_payload(payload)
    publication_report = build_publication_dry_run_report_from_payload(
        payload,
        include_rows=include_rows,
    )
    backfill_report = build_physics_ingest_backfill_report(
        source_bundles=_sequence(payload.get("source_bundles", ()), "source_bundles"),
        publication_manifests=_sequence(
            payload.get("publication_manifests", ()),
            "publication_manifests",
        ),
        artifact_bindings=_mapping(
            payload.get("artifact_bindings", {}),
            "artifact_bindings",
        ),
        table_modes=table_modes,
        source_retrieval_run_plan=retrieval_plan,
        include_source_request_envelopes=_payload_bool(
            payload,
            "include_source_request_envelopes",
            default=False,
        ),
        include_source_runtime_execution_preflight=(
            _payload_bool(
                payload,
                "include_source_runtime_execution_preflight",
                default=False,
            )
            and source_runtime_execution_preflight is None
        ),
        include_publication_write_preflight=_payload_bool(
            payload,
            "include_publication_write_preflight",
            default=False,
        ),
        include_execution_boundary_preflight=include_execution_boundary_preflight,
        include_audit_artifact_manifests=(
            _payload_bool(
                payload,
                "include_audit_artifact_manifests",
                default=False,
            )
            or include_audit_artifact_write_plan_rows
        ),
        include_rows=include_rows,
    )
    if source_runtime_execution_preflight is not None:
        backfill_report["source_runtime_execution_preflight"] = (
            source_runtime_execution_preflight
        )

    report: dict[str, Any] = {
        "report_kind": COMPOSED_REPORT_KIND,
        "dry_run": True,
        "ok": bool(publication_report["ok"] and backfill_report["ok"]),
        "publication_dry_run_report": publication_report,
        "backfill_report": backfill_report,
        "source_retrieval_run_plan": backfill_report["source_retrieval_run_plan"],
        "phase7_coverage_row_counts": backfill_report["phase7_coverage_row_counts"],
        "phase7_coverage_summary": backfill_report["phase7_coverage_summary"],
    }
    for summary_key in (
        "publication_readiness_summary",
        "source_retrieval_summary",
        "source_retrieval_readiness_summary",
    ):
        if summary_key in backfill_report:
            report[summary_key] = backfill_report[summary_key]
    for preflight_key in (
        "source_request_envelope_preflight",
        "source_runtime_execution_preflight",
        "publication_storage_write_preflight",
    ):
        if preflight_key in backfill_report:
            report[preflight_key] = backfill_report[preflight_key]
    if include_audit_artifact_write_plan_rows:
        audit_artifact_write_plan_rows = (
            build_backfill_audit_artifact_write_plan_rows(
                backfill_report,
                include_write_plan=True,
                table_modes=table_modes,
            )
        )
        report["audit_artifact_write_plan_rows"] = _write_plan_rows_section(
            audit_artifact_write_plan_rows,
            include_rows=include_rows,
        )
    if include_review_queue_write_plan_rows or review_queue_rows is not None:
        if review_queue_rows is None:
            raise ValueError(
                "review_queue_tasks or review_queue_rows is required when "
                "include_review_queue_write_plan_rows is true"
            )
        review_queue_write_plan_rows = build_review_queue_write_plan_rows(
            review_queue_rows,
            include_write_plan=True,
            table_modes=table_modes,
        )
        report["review_queue_write_plan_rows"] = _write_plan_rows_section(
            review_queue_write_plan_rows,
            include_rows=include_rows,
        )
    report["dashboard_summary"] = _composed_backfill_dashboard_summary(
        publication_report=publication_report,
        backfill_report=backfill_report,
        ok=report["ok"],
    )
    _assert_json_serializable(report)
    return report


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


def _composed_backfill_dashboard_summary(
    *,
    publication_report: Mapping[str, Any],
    backfill_report: Mapping[str, Any],
    ok: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "report_version": "physics-ingest-publication-backfill-dashboard.v1",
        "ok": bool(ok),
        "dry_run": True,
        "publication": _publication_dashboard_rollup(publication_report),
        "backfill": _json_safe_mapping(
            _mapping_or_empty(backfill_report.get("dashboard_summary"))
        ),
    }
    runtime_preflight = _mapping_or_empty(
        backfill_report.get("source_runtime_execution_preflight")
    )
    if runtime_preflight:
        summary["source_runtime_execution"] = (
            _source_runtime_execution_dashboard_rollup(runtime_preflight)
        )
    return _json_safe_mapping(summary)


def _publication_dashboard_rollup(report: Mapping[str, Any]) -> dict[str, Any]:
    write_plan = _mapping_or_empty(report.get("write_plan"))
    batches = [
        batch
        for batch in write_plan.get("batches", ())
        if isinstance(batch, Mapping)
    ]
    return {
        "ok": bool(report.get("ok")),
        "source_bundle_count": _int_value(report.get("source_bundle_count")),
        "publication_manifest_count": _int_value(
            report.get("publication_manifest_count")
        ),
        "diagnostic_count": len(
            [row for row in report.get("diagnostics", ()) if isinstance(row, Mapping)]
        ),
        "id_strategy": str(
            _mapping_or_empty(report.get("id_planning")).get("strategy") or ""
        ),
        "write_plan": {
            "batch_count": len(batches),
            "row_count": sum(_int_value(batch.get("row_count")) for batch in batches),
            "mode_counts": _count_values(
                str(batch.get("mode") or "unknown") for batch in batches
            ),
        },
    }


def _source_runtime_execution_dashboard_rollup(
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _mapping_or_empty(preflight.get("summary"))
    return {
        "step_count": _int_value(summary.get("total_steps")),
        "diagnostic_count": _int_value(summary.get("diagnostic_count")),
        "blocking_diagnostic_count": _int_value(
            summary.get("blocking_diagnostic_count")
        ),
        "execution_requested": bool(preflight.get("execution_requested")),
        "execution_performed": bool(preflight.get("execution_performed")),
        "side_effect_free": bool(preflight.get("side_effect_free", True)),
    }


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _assert_json_serializable(report: Mapping[str, Any]) -> None:
    try:
        json.dumps(report, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("dry-run report must be JSON serializable") from exc


def _has_backfill_payload(payload: Mapping[str, Any]) -> bool:
    return _has_retrieval_run_plan_payload(payload) or any(
        payload.get(key) not in (None, "", False)
        for key in (
            "include_source_request_envelopes",
            "include_source_runtime_execution_preflight",
            "include_publication_write_preflight",
            "include_execution_boundary_preflight",
            "include_audit_artifact_manifests",
            "include_audit_artifact_write_plan_rows",
            "include_review_queue_write_plan_rows",
            "review_queue_tasks",
            "review_queue_rows",
            "source_runtime_execution_plan",
            "source_runtime_execution_report",
            "source_runtime_execution_preflight",
        )
    )


def _has_retrieval_run_plan_payload(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("source_retrieval_run_plan") not in (None, "")
        or payload.get("retrieval_run_plan") not in (None, "")
    )


def _optional_retrieval_run_plan_payload(payload: Mapping[str, Any]) -> Any | None:
    source_plan = payload.get("source_retrieval_run_plan")
    alias_plan = payload.get("retrieval_run_plan")
    has_source_plan = source_plan not in (None, "")
    has_alias_plan = alias_plan not in (None, "")
    if has_source_plan and has_alias_plan:
        raise ValueError(
            "pass only one of source_retrieval_run_plan or retrieval_run_plan"
        )
    if has_source_plan:
        return source_plan
    if has_alias_plan:
        return alias_plan
    return None


def _payload_bool(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    default: bool,
) -> bool:
    if field_name not in payload:
        return default
    value = payload[field_name]
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _source_runtime_execution_preflight_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    values = {
        key: payload.get(key)
        for key in (
            "source_runtime_execution_plan",
            "source_runtime_execution_report",
            "source_runtime_execution_preflight",
        )
        if payload.get(key) not in (None, "")
    }
    if len(values) > 1:
        raise ValueError(
            "pass only one source runtime execution plan, report, or preflight payload"
        )
    if not values:
        return None

    field_name, value = next(iter(values.items()))
    mapping = _mapping(value, field_name)
    if field_name == "source_runtime_execution_plan":
        return _json_safe_mapping(
            build_source_retrieval_runtime_execution_report_dict(
                mapping,
                execute=True,
                preflight=True,
            )
        )
    return _validate_source_runtime_execution_preflight(mapping, field_name)


def _validate_source_runtime_execution_preflight(
    value: Mapping[str, Any],
    field_name: str,
) -> dict[str, Any]:
    preflight = _json_safe_mapping(value)
    if preflight.get("preflight") is False:
        raise ValueError(f"{field_name} must describe a preflight report")
    if preflight.get("execution_performed") is True:
        raise ValueError(f"{field_name} must not include performed execution")
    if preflight.get("side_effect_free") is False:
        raise ValueError(f"{field_name} must be side-effect-free")
    return preflight


def _review_queue_write_plan_rows_payload(payload: Mapping[str, Any]) -> Any | None:
    tasks = payload.get("review_queue_tasks")
    rows = payload.get("review_queue_rows")
    has_tasks = tasks not in (None, "")
    has_rows = rows not in (None, "")
    if has_tasks and has_rows:
        raise ValueError("pass only one of review_queue_tasks or review_queue_rows")
    if has_tasks:
        return tasks
    if has_rows:
        return _review_queue_rows_payload(rows)
    return None


def _review_queue_rows_payload(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if "tasks" in value:
        return value
    if "insert_rows" in value:
        insert_rows = _mapping(value["insert_rows"], "review_queue_rows.insert_rows")
        return _sequence(
            insert_rows.get(REVIEW_QUEUE_TASKS_TABLE, ()),
            f"review_queue_rows.insert_rows.{REVIEW_QUEUE_TASKS_TABLE}",
        )
    if REVIEW_QUEUE_TASKS_TABLE in value:
        return _sequence(
            value[REVIEW_QUEUE_TASKS_TABLE],
            f"review_queue_rows.{REVIEW_QUEUE_TASKS_TABLE}",
        )
    return value


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


def _write_plan_rows_section(value: Any, *, include_rows: bool) -> dict[str, Any]:
    section: dict[str, Any] = {
        "summary": _json_safe_mapping(getattr(value, "summary", {}) or {}),
        "diagnostics": [
            _json_safe_mapping(row)
            for row in tuple(getattr(value, "diagnostics", ()) or ())
            if isinstance(row, Mapping)
        ],
    }
    write_plan = getattr(value, "write_plan", None)
    if write_plan is not None:
        section["write_plan"] = {
            "audit_summary": write_plan.audit_summary.to_dict(),
            "batches": [
                {
                    "table": batch.table,
                    "mode": write_plan.mode_for(batch.table),
                    "row_count": batch.row_count,
                    "conflict_keys": list(batch.conflict_keys),
                    "dry_run": True,
                    **(
                        {"rows": [dict(row) for row in batch.rows]}
                        if include_rows
                        else {}
                    ),
                }
                for batch in write_plan.batches
            ],
        }
    if include_rows:
        section["insert_rows"] = value.to_insert_rows()
    return _json_safe_mapping(section)


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
