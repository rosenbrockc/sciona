"""Production deployment planning for physics ingest bulk backfills."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from sciona.physics_ingest.audit_artifacts import (
    BACKFILL_AUDIT_ARTIFACTS_TABLE,
    BackfillAuditArtifactWritePlanRows,
    build_backfill_audit_artifact_write_plan_rows,
)
from sciona.physics_ingest.backfill import (
    BACKFILL_REPORT_KIND,
    build_physics_ingest_backfill_report,
)
from sciona.physics_ingest.deployment import (
    PhysicsIngestDeploymentStorageApplyResult,
    PhysicsIngestDeploymentStorageBundle,
    apply_physics_ingest_deployment_storage_bundle,
    build_physics_ingest_deployment_storage_bundle,
)
from sciona.physics_ingest.deployment_runtime import (
    build_physics_ingest_deployment_runtime_report_dict,
)
from sciona.physics_ingest.write_plan import WriteMode


BACKFILL_DEPLOYMENT_REPORT_VERSION = (
    "physics-ingest-production-backfill-deployment-report.v1"
)
BACKFILL_DEPLOYMENT_REPORT_KIND = "physics_ingest_production_backfill_deployment"


@dataclass(frozen=True)
class PhysicsIngestBackfillDeploymentReport:
    """JSON-safe production deployment plan/report for a bulk backfill."""

    report: Mapping[str, Any]
    backfill_report: Mapping[str, Any]
    audit_artifact_rows: BackfillAuditArtifactWritePlanRows
    storage_bundle: PhysicsIngestDeploymentStorageBundle
    runtime_preflight_report: Mapping[str, Any]
    storage_apply_result: PhysicsIngestDeploymentStorageApplyResult | None = None

    @property
    def ok(self) -> bool:
        return bool(self.report["ok"])

    @property
    def dry_run(self) -> bool:
        return bool(self.report["dry_run"])

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self.report)


def build_physics_ingest_backfill_deployment_report(
    *,
    backfill_report: Mapping[str, Any] | None = None,
    client: Any | None = None,
    dry_run: bool = True,
    apply_storage: bool = True,
    table_modes: Mapping[str, WriteMode] | None = None,
    conflict_keys_by_table: Mapping[str, Sequence[str]] | None = None,
    runtime_source_retrieval_run_plan: Any | None = None,
    runtime_source_execution_preflight: Mapping[str, Any] | None = None,
    include_backfill_report: bool = True,
    **backfill_report_kwargs: Any,
) -> PhysicsIngestBackfillDeploymentReport:
    """Build and optionally apply a production storage plan for a bulk backfill.

    The helper constructs no clients.  All writes flow through the caller's
    injected PostgREST/Supabase-style client, and ``dry_run`` defaults to true.
    """

    normalized_backfill_report, source = _resolve_backfill_report(
        backfill_report=backfill_report,
        backfill_report_kwargs=backfill_report_kwargs,
    )
    audit_table_modes = _audit_table_modes(table_modes)
    audit_artifact_rows = build_backfill_audit_artifact_write_plan_rows(
        normalized_backfill_report,
        include_write_plan=True,
        table_modes={BACKFILL_AUDIT_ARTIFACTS_TABLE: "upsert"},
    )
    storage_bundle = build_physics_ingest_deployment_storage_bundle(
        _publication_rows(normalized_backfill_report),
        audit_artifact_rows=audit_artifact_rows,
        table_modes=audit_table_modes,
    )
    storage_preflight = storage_bundle.preflight_supabase_write(
        conflict_keys_by_table=conflict_keys_by_table,
    ).to_dict()
    runtime_preflight_report = build_physics_ingest_deployment_runtime_report_dict(
        source_retrieval_run_plan=(
            runtime_source_retrieval_run_plan
            if runtime_source_retrieval_run_plan is not None
            else _runtime_source_retrieval_run_plan(normalized_backfill_report)
        ),
        source_runtime_execution_preflight=(
            runtime_source_execution_preflight
            if runtime_source_execution_preflight is not None
            else _runtime_source_execution_preflight(normalized_backfill_report)
        ),
        storage_bundle=storage_bundle,
        storage_preflight=storage_preflight,
    )
    storage_apply_result = (
        apply_physics_ingest_deployment_storage_bundle(
            storage_bundle,
            client,
            dry_run=dry_run,
            conflict_keys_by_table=conflict_keys_by_table,
        )
        if apply_storage
        else None
    )
    report = _deployment_report_dict(
        backfill_report=normalized_backfill_report,
        backfill_report_source=source,
        audit_artifact_rows=audit_artifact_rows,
        storage_bundle=storage_bundle,
        storage_preflight=storage_preflight,
        runtime_preflight_report=runtime_preflight_report,
        storage_apply_result=storage_apply_result,
        dry_run=dry_run,
        apply_storage=apply_storage,
        client_supplied=client is not None,
        include_backfill_report=include_backfill_report,
    )
    return PhysicsIngestBackfillDeploymentReport(
        report=report,
        backfill_report=normalized_backfill_report,
        audit_artifact_rows=audit_artifact_rows,
        storage_bundle=storage_bundle,
        runtime_preflight_report=runtime_preflight_report,
        storage_apply_result=storage_apply_result,
    )


def _resolve_backfill_report(
    *,
    backfill_report: Mapping[str, Any] | None,
    backfill_report_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if backfill_report is not None:
        if backfill_report_kwargs:
            raise ValueError(
                "pass either backfill_report or backfill report kwargs, not both"
            )
        return _ensure_audit_artifact_manifests(backfill_report), "existing_report"

    kwargs = dict(backfill_report_kwargs)
    kwargs.update(
        {
            "include_rows": True,
            "include_audit_artifact_manifests": True,
            "include_publication_write_preflight": True,
            "include_source_request_envelopes": True,
            "include_source_runtime_execution_preflight": True,
        }
    )
    return _json_safe(build_physics_ingest_backfill_report(**kwargs)), "built_report"


def _ensure_audit_artifact_manifests(report: Mapping[str, Any]) -> dict[str, Any]:
    copied = _json_safe(report)
    manifests = copied.get("audit_artifact_manifests")
    if isinstance(manifests, list) and manifests:
        return copied

    generated = _audit_artifact_manifests(copied, include_payload=True)
    copied["audit_artifact_manifests"] = generated
    copied["audit_artifact_manifest_summary"] = _manifest_summary(generated)
    return copied


def _audit_artifact_manifests(
    report: Mapping[str, Any],
    *,
    include_payload: bool,
) -> list[dict[str, Any]]:
    sections = (
        "dashboard_summary",
        "audit_replay",
        "phase7_coverage_summary",
        "source_request_envelope_preflight",
        "source_runtime_execution_preflight",
        "publication_storage_write_preflight",
    )
    return [
        _audit_artifact_manifest_row(
            section_name=section_name,
            payload=payload,
            report=report,
            include_payload=include_payload,
        )
        for section_name in sections
        if isinstance((payload := report.get(section_name)), Mapping)
    ]


def _audit_artifact_manifest_row(
    *,
    section_name: str,
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    include_payload: bool,
) -> dict[str, Any]:
    safe_payload = _json_safe(payload)
    payload_sha256 = _stable_json_sha256(safe_payload)
    row = {
        "manifest_version": "physics-ingest-backfill-artifact-manifest-row.v1",
        "artifact_key": f"{BACKFILL_REPORT_KIND}/{section_name}/{payload_sha256}",
        "artifact_name": section_name,
        "name": section_name,
        "source_section": section_name,
        "source_report_kind": str(report.get("report_kind") or ""),
        "source_replay_key": (
            f"{str(report.get('report_kind') or '')}|{section_name}|"
            f"payload_sha256={payload_sha256}"
        ),
        "source_report_fingerprint_sha256": str(
            (report.get("audit_replay") or {}).get("input_fingerprint_sha256")
            if isinstance(report.get("audit_replay"), Mapping)
            else ""
        ),
        "content_type": "application/json",
        "payload_sha256": payload_sha256,
        "payload_count": len(safe_payload),
        "section_count": len(safe_payload),
    }
    if include_payload:
        row["payload"] = safe_payload
    return _json_safe(row)


def _manifest_summary(manifests: Any) -> dict[str, Any]:
    rows = [row for row in _json_safe(manifests) if isinstance(row, Mapping)]
    digest_source = [
        {
            "artifact_key": row.get("artifact_key", ""),
            "content_type": row.get("content_type", ""),
            "payload_sha256": row.get("payload_sha256", ""),
            "source_section": row.get("source_section", ""),
        }
        for row in rows
    ]
    return {
        "manifest_version": "physics-ingest-backfill-artifact-manifest-summary.v1",
        "artifact_count": len(rows),
        "content_type_counts": _count_by(rows, "content_type"),
        "payload_sha256_digest": _stable_sequence_digest(digest_source),
        "artifact_keys": [str(row.get("artifact_key") or "") for row in rows],
    }


def _publication_rows(report: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = report.get("insert_rows_by_table")
    return rows if isinstance(rows, Mapping) else {}


def _audit_table_modes(
    table_modes: Mapping[str, WriteMode] | None,
) -> dict[str, WriteMode]:
    return {
        **dict(table_modes or {}),
        BACKFILL_AUDIT_ARTIFACTS_TABLE: "upsert",
    }


def _runtime_source_execution_preflight(
    report: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    preflight = report.get("source_runtime_execution_preflight")
    return preflight if isinstance(preflight, Mapping) else None


def _runtime_source_retrieval_run_plan(
    report: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    run_plan = report.get("source_retrieval_run_plan")
    return run_plan if isinstance(run_plan, Mapping) else None


def _deployment_report_dict(
    *,
    backfill_report: Mapping[str, Any],
    backfill_report_source: str,
    audit_artifact_rows: BackfillAuditArtifactWritePlanRows,
    storage_bundle: PhysicsIngestDeploymentStorageBundle,
    storage_preflight: Mapping[str, Any],
    runtime_preflight_report: Mapping[str, Any],
    storage_apply_result: PhysicsIngestDeploymentStorageApplyResult | None,
    dry_run: bool,
    apply_storage: bool,
    client_supplied: bool,
    include_backfill_report: bool,
) -> dict[str, Any]:
    apply_result_dict = (
        storage_apply_result.to_dict() if storage_apply_result is not None else None
    )
    summary = _summary(
        backfill_report=backfill_report,
        audit_artifact_rows=audit_artifact_rows,
        storage_bundle=storage_bundle,
        storage_preflight=storage_preflight,
        runtime_preflight_report=runtime_preflight_report,
        storage_apply_result=storage_apply_result,
        dry_run=dry_run,
        apply_storage=apply_storage,
        client_supplied=client_supplied,
    )
    report = {
        "report_version": BACKFILL_DEPLOYMENT_REPORT_VERSION,
        "report_kind": BACKFILL_DEPLOYMENT_REPORT_KIND,
        "ok": summary["ok"],
        "dry_run": dry_run,
        "side_effect_free": dry_run,
        "input": {
            "backfill_report_source": backfill_report_source,
            "backfill_report_kind": str(backfill_report.get("report_kind") or ""),
            "client_supplied": client_supplied,
            "apply_storage": apply_storage,
        },
        "summary": summary,
        "storage_preflight_summary": _storage_preflight_summary(storage_preflight),
        "runtime_preflight_summary": _json_safe(
            runtime_preflight_report.get("summary") or {}
        ),
        "backfill_dashboard_summary": _json_safe(
            backfill_report.get("dashboard_summary") or {}
        ),
        "audit_artifact_manifest_summary": _json_safe(
            backfill_report.get("audit_artifact_manifest_summary") or {}
        ),
        "audit_artifact_storage": audit_artifact_rows.to_dict(),
        "storage_bundle": storage_bundle.to_dict(),
        "storage_preflight": storage_preflight,
        "runtime_preflight": runtime_preflight_report,
        "storage_apply_result": apply_result_dict,
    }
    report["dashboard_summary"] = _dashboard_summary(report)
    if include_backfill_report:
        report["backfill_report"] = backfill_report
    return _json_safe(report)


def _summary(
    *,
    backfill_report: Mapping[str, Any],
    audit_artifact_rows: BackfillAuditArtifactWritePlanRows,
    storage_bundle: PhysicsIngestDeploymentStorageBundle,
    storage_preflight: Mapping[str, Any],
    runtime_preflight_report: Mapping[str, Any],
    storage_apply_result: PhysicsIngestDeploymentStorageApplyResult | None,
    dry_run: bool,
    apply_storage: bool,
    client_supplied: bool,
) -> dict[str, Any]:
    audit_summary = audit_artifact_rows.summary
    apply_summary = (
        storage_apply_result.summary if storage_apply_result is not None else {}
    )
    ok = (
        bool(backfill_report.get("ok", True))
        and not _has_error_diagnostics(audit_artifact_rows.diagnostics)
        and bool(runtime_preflight_report.get("ok", True))
        and not bool(
            (storage_apply_result.write_result.has_errors)
            if storage_apply_result is not None
            else False
        )
    )
    return {
        "ok": ok,
        "dry_run": dry_run,
        "client_supplied": client_supplied,
        "apply_storage": apply_storage,
        "storage_write_performed": bool(apply_summary.get("wrote", False)),
        "backfill_ok": bool(backfill_report.get("ok", True)),
        "audit_artifact_manifest_count": len(
            backfill_report.get("audit_artifact_manifests") or ()
        ),
        "audit_artifact_storage_row_count": int(audit_summary.get("row_count") or 0),
        "audit_artifact_storage_diagnostic_count": len(
            audit_artifact_rows.diagnostics
        ),
        "storage_table_count": int(storage_preflight.get("table_count") or 0),
        "storage_total_row_count": int(storage_preflight.get("total_row_count") or 0),
        "storage_bundle_total_row_count": int(
            storage_bundle.summary.get("total_row_count") or 0
        ),
        "runtime_ok": bool(runtime_preflight_report.get("ok", True)),
        "runtime_blocked": bool(runtime_preflight_report.get("blocked", False)),
        "runtime_blocking_diagnostic_count": int(
            (runtime_preflight_report.get("summary") or {}).get(
                "blocking_diagnostic_count",
                0,
            )
            if isinstance(runtime_preflight_report.get("summary"), Mapping)
            else 0
        ),
    }


def _storage_preflight_summary(storage_preflight: Mapping[str, Any]) -> dict[str, Any]:
    tables = [
        row for row in storage_preflight.get("tables", ()) if isinstance(row, Mapping)
    ]
    return {
        "table_count": int(storage_preflight.get("table_count") or 0),
        "total_row_count": int(storage_preflight.get("total_row_count") or 0),
        "missing_conflict_metadata_for_upserts": list(
            storage_preflight.get("missing_conflict_metadata_for_upserts") or ()
        ),
        "table_modes": {
            str(row.get("table")): str(row.get("mode") or "")
            for row in tables
            if row.get("table")
        },
        "table_row_counts": {
            str(row.get("table")): int(row.get("row_count") or 0)
            for row in tables
            if row.get("table")
        },
    }


def _dashboard_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") or {}
    storage = report.get("storage_preflight_summary") or {}
    runtime = report.get("runtime_preflight_summary") or {}
    backfill = report.get("backfill_dashboard_summary") or {}
    audit_manifest = report.get("audit_artifact_manifest_summary") or {}
    return _json_safe(
        {
            "report_version": "physics-ingest-backfill-deployment-dashboard.v1",
            "ok": bool(report.get("ok")),
            "dry_run": bool(report.get("dry_run")),
            "side_effect_free": bool(report.get("side_effect_free")),
            "input": _json_safe(report.get("input") or {}),
            "backfill": {
                "ok": bool(backfill.get("ok", True))
                if isinstance(backfill, Mapping)
                else True,
                "write_plan_row_count": _nested_int(
                    backfill,
                    "write_plan",
                    "row_count",
                ),
                "phase7_row_count": _nested_int(
                    backfill,
                    "phase7_coverage",
                    "row_count",
                ),
                "source_retrieval_step_count": _nested_int(
                    backfill,
                    "source_retrieval",
                    "step_count",
                ),
            },
            "audit_artifacts": {
                "manifest_count": _int(
                    summary.get("audit_artifact_manifest_count")
                    if isinstance(summary, Mapping)
                    else 0
                ),
                "manifest_artifact_count": _int(
                    audit_manifest.get("artifact_count")
                    if isinstance(audit_manifest, Mapping)
                    else 0
                ),
                "storage_row_count": _int(
                    summary.get("audit_artifact_storage_row_count")
                    if isinstance(summary, Mapping)
                    else 0
                ),
                "storage_diagnostic_count": _int(
                    summary.get("audit_artifact_storage_diagnostic_count")
                    if isinstance(summary, Mapping)
                    else 0
                ),
            },
            "storage": {
                "table_count": _int(
                    storage.get("table_count") if isinstance(storage, Mapping) else 0
                ),
                "total_row_count": _int(
                    storage.get("total_row_count")
                    if isinstance(storage, Mapping)
                    else 0
                ),
                "write_performed": bool(
                    summary.get("storage_write_performed")
                    if isinstance(summary, Mapping)
                    else False
                ),
            },
            "runtime": {
                "ok": bool(summary.get("runtime_ok", True))
                if isinstance(summary, Mapping)
                else True,
                "blocked": bool(summary.get("runtime_blocked", False))
                if isinstance(summary, Mapping)
                else False,
                "blocking_diagnostic_count": _int(
                    summary.get("runtime_blocking_diagnostic_count")
                    if isinstance(summary, Mapping)
                    else 0
                ),
                "source_runtime_step_count": _int(
                    runtime.get("source_runtime_step_count")
                    if isinstance(runtime, Mapping)
                    else 0
                ),
            },
        }
    )


def _nested_int(value: Any, section: str, field: str) -> int:
    if not isinstance(value, Mapping):
        return 0
    nested = value.get(section) or {}
    if not isinstance(nested, Mapping):
        return 0
    return _int(nested.get(field))


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _has_error_diagnostics(diagnostics: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(row.get("severity") or "") == "error" for row in diagnostics)


def _count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _stable_sequence_digest(values: Sequence[Mapping[str, Any]]) -> str:
    return _stable_json_sha256({"items": list(values)})


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    return json.loads(
        json.dumps(
            _sanitize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_sanitize(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_sanitize(item) for item in sorted(value, key=repr)]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str | int | bool):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _sanitize(to_dict())
    return str(value)


__all__ = [
    "BACKFILL_DEPLOYMENT_REPORT_KIND",
    "BACKFILL_DEPLOYMENT_REPORT_VERSION",
    "PhysicsIngestBackfillDeploymentReport",
    "build_physics_ingest_backfill_deployment_report",
]
