"""Side-effect-free deployment runtime preflight report composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from sciona.physics_ingest.deployment import (
    PhysicsIngestDeploymentStorageBundle,
    preflight_physics_ingest_deployment_storage_bundle,
)
from sciona.physics_ingest.sources import (
    RetrievalRunPlan,
    build_physics_source_retrieval_run_plan,
    build_source_retrieval_runtime_execution_report_dict,
)


JSONDict = dict[str, Any]

DEPLOYMENT_RUNTIME_REPORT_VERSION = "physics-ingest-deployment-runtime-report.v1"


@dataclass(frozen=True)
class PhysicsIngestDeploymentRuntimeReport:
    """JSON-safe deployment runtime preflight report."""

    report: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.report["ok"])

    @property
    def blocked(self) -> bool:
        return bool(self.report["blocked"])

    def to_dict(self) -> JSONDict:
        return _json_safe(self.report)


def build_physics_ingest_deployment_runtime_report(
    *,
    source_retrieval_run_plan: RetrievalRunPlan | Mapping[str, Any] | None = None,
    source_runtime_execution_preflight: Mapping[str, Any] | None = None,
    source_runtime_execution_report: Mapping[str, Any] | None = None,
    storage_bundle: PhysicsIngestDeploymentStorageBundle | None = None,
    storage_preflight: Mapping[str, Any] | Any | None = None,
) -> PhysicsIngestDeploymentRuntimeReport:
    """Compose source runtime and storage preflights without network or DB writes.

    If no source runtime report is supplied, this helper builds one from the
    supplied run plan.  If neither is supplied, it builds the full production
    source retrieval run plan with ``dry_run=False`` so deployment blockers are
    visible before any external IO is attempted.
    """

    source_plan = _source_plan(
        source_retrieval_run_plan=source_retrieval_run_plan,
        source_runtime_execution_preflight=(
            source_runtime_execution_preflight
            if source_runtime_execution_preflight is not None
            else source_runtime_execution_report
        ),
    )
    source_preflight = _source_runtime_preflight(
        source_retrieval_run_plan=source_plan,
        source_runtime_execution_preflight=(
            source_runtime_execution_preflight
            if source_runtime_execution_preflight is not None
            else source_runtime_execution_report
        ),
    )
    storage_preflight_dict = _storage_preflight(
        storage_bundle=storage_bundle,
        storage_preflight=storage_preflight,
    )
    storage_bundle_summary = _storage_bundle_summary(storage_bundle)
    storage_table_counts = _storage_table_counts(
        storage_bundle=storage_bundle,
        storage_preflight=storage_preflight_dict,
    )
    diagnostics = _diagnostics(
        source_preflight=source_preflight,
        storage_bundle=storage_bundle,
        storage_preflight=storage_preflight_dict,
    )
    diagnostic_summary = _diagnostic_summary(diagnostics)
    source_counts = _source_runtime_counts(source_preflight)
    replay_keys = _source_runtime_replay_keys(source_preflight, source_plan)
    blocked = diagnostic_summary["blocking_diagnostic_count"] > 0

    report = {
        "report_version": DEPLOYMENT_RUNTIME_REPORT_VERSION,
        "ok": not blocked,
        "blocked": blocked,
        "side_effect_free": True,
        "preflight": True,
        "flags": {
            "side_effect_free": True,
            "preflight": True,
            "source_runtime_preflight": bool(source_preflight.get("preflight", True)),
            "storage_preflight": storage_preflight_dict is not None,
            "execution_performed": bool(
                source_preflight.get("execution_performed", False)
            ),
            "db_write_performed": False,
        },
        "summary": {
            "ok": not blocked,
            "blocked": blocked,
            "source_runtime_step_count": source_counts["total_steps"],
            "source_runtime_network_step_count": source_counts["network_step_count"],
            "source_runtime_manual_step_count": source_counts["manual_step_count"],
            "source_runtime_dry_run_step_count": source_counts["dry_run_step_count"],
            "source_runtime_requires_http_client_count": source_counts[
                "requires_http_client_count"
            ],
            "source_runtime_requires_snapshot_sink_count": source_counts[
                "requires_snapshot_sink_count"
            ],
            "source_runtime_requires_auth_count": source_counts[
                "requires_auth_count"
            ],
            "storage_table_count": len(storage_table_counts),
            "storage_total_row_count": sum(storage_table_counts.values()),
            "diagnostic_count": diagnostic_summary["diagnostic_count"],
            "blocking_diagnostic_count": diagnostic_summary[
                "blocking_diagnostic_count"
            ],
        },
        "source_runtime_counts": source_counts,
        "storage_table_counts": storage_table_counts,
        "diagnostics": diagnostics,
        "diagnostic_summary": diagnostic_summary,
        "replay_keys": {
            "source_runtime": replay_keys,
            "source_runtime_count": len(replay_keys),
            "source_runtime_digest": _digest(replay_keys),
        },
        "source_runtime_execution_preflight": source_preflight,
        "storage_bundle_summary": storage_bundle_summary,
        "storage_preflight": storage_preflight_dict,
    }
    report["dashboard_summary"] = _dashboard_summary(report)
    return PhysicsIngestDeploymentRuntimeReport(_json_safe(report))


def build_physics_ingest_deployment_runtime_report_dict(
    **kwargs: Any,
) -> JSONDict:
    """Return ``build_physics_ingest_deployment_runtime_report(...).to_dict()``."""

    return build_physics_ingest_deployment_runtime_report(**kwargs).to_dict()


def _source_runtime_preflight(
    *,
    source_retrieval_run_plan: RetrievalRunPlan | Mapping[str, Any],
    source_runtime_execution_preflight: Mapping[str, Any] | None,
) -> JSONDict:
    if source_runtime_execution_preflight is not None:
        return _json_safe(source_runtime_execution_preflight)
    return _json_safe(
        build_source_retrieval_runtime_execution_report_dict(
            source_retrieval_run_plan,
            execute=True,
            preflight=True,
        )
    )


def _source_plan(
    *,
    source_retrieval_run_plan: RetrievalRunPlan | Mapping[str, Any] | None,
    source_runtime_execution_preflight: Mapping[str, Any] | None,
) -> RetrievalRunPlan | Mapping[str, Any]:
    if source_retrieval_run_plan is not None:
        return source_retrieval_run_plan
    if source_runtime_execution_preflight is not None:
        return {}
    return build_physics_source_retrieval_run_plan(dry_run=False)


def _storage_preflight(
    *,
    storage_bundle: PhysicsIngestDeploymentStorageBundle | None,
    storage_preflight: Mapping[str, Any] | Any | None,
) -> JSONDict | None:
    if storage_preflight is not None:
        return _json_safe(_to_dict(storage_preflight))
    if storage_bundle is None:
        return None
    return _json_safe(
        preflight_physics_ingest_deployment_storage_bundle(storage_bundle).to_dict()
    )


def _storage_bundle_summary(
    storage_bundle: PhysicsIngestDeploymentStorageBundle | None,
) -> JSONDict:
    if storage_bundle is None:
        return {}
    return _json_safe(storage_bundle.summary)


def _source_runtime_counts(source_preflight: Mapping[str, Any]) -> dict[str, int]:
    summary = _mapping(source_preflight.get("summary"))
    return {
        "total_steps": _int(summary.get("total_steps")),
        "network_step_count": _int(summary.get("network_step_count")),
        "manual_step_count": _int(summary.get("manual_step_count")),
        "dry_run_step_count": _int(summary.get("dry_run_step_count")),
        "requires_http_client_count": _int(summary.get("requires_http_client_count")),
        "requires_snapshot_sink_count": _int(
            summary.get("requires_snapshot_sink_count")
        ),
        "requires_auth_count": _int(summary.get("requires_auth_count")),
        "diagnostic_count": _int(summary.get("diagnostic_count")),
        "blocking_diagnostic_count": _int(summary.get("blocking_diagnostic_count")),
        "execution_result_count": _int(summary.get("execution_result_count")),
    }


def _dashboard_summary(report: Mapping[str, Any]) -> JSONDict:
    summary = _mapping(report.get("summary"))
    source_counts = _mapping(report.get("source_runtime_counts"))
    diagnostic_summary = _mapping(report.get("diagnostic_summary"))
    replay_keys = _mapping(report.get("replay_keys"))
    return _json_safe(
        {
            "report_version": "physics-ingest-deployment-runtime-dashboard.v1",
            "ok": bool(report.get("ok")),
            "blocked": bool(report.get("blocked")),
            "side_effect_free": bool(report.get("side_effect_free")),
            "preflight": bool(report.get("preflight")),
            "source_runtime": {
                "step_count": _int(summary.get("source_runtime_step_count")),
                "network_step_count": _int(
                    summary.get("source_runtime_network_step_count")
                ),
                "manual_step_count": _int(
                    summary.get("source_runtime_manual_step_count")
                ),
                "dry_run_step_count": _int(
                    summary.get("source_runtime_dry_run_step_count")
                ),
                "requires_http_client_count": _int(
                    summary.get("source_runtime_requires_http_client_count")
                ),
                "requires_snapshot_sink_count": _int(
                    summary.get("source_runtime_requires_snapshot_sink_count")
                ),
                "requires_auth_count": _int(
                    summary.get("source_runtime_requires_auth_count")
                ),
                "execution_result_count": _int(
                    source_counts.get("execution_result_count")
                ),
            },
            "storage": {
                "table_count": _int(summary.get("storage_table_count")),
                "total_row_count": _int(summary.get("storage_total_row_count")),
                "table_row_counts": dict(
                    sorted(
                        (
                            str(table),
                            _int(row_count),
                        )
                        for table, row_count in _mapping(
                            report.get("storage_table_counts")
                        ).items()
                    )
                ),
            },
            "diagnostics": {
                "diagnostic_count": _int(summary.get("diagnostic_count")),
                "blocking_diagnostic_count": _int(
                    summary.get("blocking_diagnostic_count")
                ),
                "by_stage": _mapping(diagnostic_summary.get("by_stage")),
                "by_severity": _mapping(diagnostic_summary.get("by_severity")),
                "blocking_by_code": _mapping(
                    diagnostic_summary.get("blocking_by_code")
                ),
            },
            "replay": {
                "source_runtime_count": _int(
                    replay_keys.get("source_runtime_count")
                ),
                "source_runtime_digest": str(
                    replay_keys.get("source_runtime_digest") or ""
                ),
            },
        }
    )


def _source_runtime_replay_keys(
    source_preflight: Mapping[str, Any],
    source_retrieval_run_plan: RetrievalRunPlan | Mapping[str, Any],
) -> list[str]:
    plan_dict = _plan_to_dict(source_retrieval_run_plan)
    plan_replay_keys = [
        str(step.get("replay_key"))
        for step in _mapping_sequence(plan_dict.get("steps"))
        if step.get("replay_key")
    ]
    if plan_replay_keys:
        return plan_replay_keys
    replay_keys = [
        str(step.get("replay_key"))
        for step in _mapping_sequence(source_preflight.get("steps"))
        if step.get("replay_key")
    ]
    if replay_keys:
        return replay_keys
    execution_report = _mapping(source_preflight.get("execution_report"))
    return [
        str(result.get("replay_key"))
        for result in _mapping_sequence(execution_report.get("results"))
        if result.get("replay_key")
    ]


def _plan_to_dict(plan: RetrievalRunPlan | Mapping[str, Any]) -> JSONDict:
    if isinstance(plan, Mapping):
        return _json_safe(plan)
    return _json_safe(plan.to_dict())


def _storage_table_counts(
    *,
    storage_bundle: PhysicsIngestDeploymentStorageBundle | None,
    storage_preflight: Mapping[str, Any] | None,
) -> dict[str, int]:
    if storage_preflight is not None:
        tables = _mapping_sequence(storage_preflight.get("tables"))
        return dict(
            sorted(
                (
                    (str(table.get("table")), _int(table.get("row_count")))
                    for table in tables
                    if table.get("table")
                ),
                key=lambda item: item[0],
            )
        )
    if storage_bundle is None:
        return {}
    return dict(
        sorted(
            (table, len(rows))
            for table, rows in storage_bundle.insert_rows_by_table.items()
        )
    )


def _diagnostics(
    *,
    source_preflight: Mapping[str, Any],
    storage_bundle: PhysicsIngestDeploymentStorageBundle | None,
    storage_preflight: Mapping[str, Any] | None,
) -> list[JSONDict]:
    diagnostics = [
        _deployment_diagnostic(
            stage="source_runtime",
            severity=str(diagnostic.get("severity") or "info"),
            code=str(diagnostic.get("code") or "source_runtime_diagnostic"),
            message=str(diagnostic.get("message") or ""),
            job_id=str(diagnostic.get("job_id") or ""),
            endpoint_id=str(diagnostic.get("endpoint_id") or ""),
            step_index=diagnostic.get("step_index"),
            dependency=str(diagnostic.get("dependency") or ""),
        )
        for diagnostic in _mapping_sequence(source_preflight.get("diagnostics"))
    ]
    if storage_bundle is not None:
        diagnostics.extend(_storage_bundle_diagnostics(storage_bundle))
    if storage_preflight is not None:
        diagnostics.extend(_storage_preflight_diagnostics(storage_preflight))
    return sorted(
        diagnostics,
        key=lambda item: (
            str(item.get("stage")),
            str(item.get("table")),
            _int(item.get("step_index")),
            str(item.get("job_id")),
            str(item.get("code")),
            str(item.get("message")),
        ),
    )


def _storage_bundle_diagnostics(
    storage_bundle: PhysicsIngestDeploymentStorageBundle,
) -> list[JSONDict]:
    rows = []
    for component in storage_bundle.components:
        for diagnostic in component.diagnostics:
            item = _mapping(diagnostic)
            rows.append(
                _deployment_diagnostic(
                    stage="storage_bundle",
                    severity=str(item.get("severity") or "info"),
                    code=str(
                        item.get("code")
                        or item.get("reason")
                        or "storage_bundle_diagnostic"
                    ),
                    message=str(item.get("message") or item.get("detail") or ""),
                    component=component.name,
                    table=str(item.get("table") or ""),
                )
            )
    return rows


def _storage_preflight_diagnostics(
    storage_preflight: Mapping[str, Any],
) -> list[JSONDict]:
    missing = [
        str(table)
        for table in storage_preflight.get(
            "missing_conflict_metadata_for_upserts",
            (),
        )
        if table
    ]
    return [
        _deployment_diagnostic(
            stage="storage_preflight",
            severity="error",
            code="missing_storage_conflict_metadata",
            message="upsert table is missing conflict key metadata",
            table=table,
        )
        for table in missing
    ]


def _deployment_diagnostic(
    *,
    stage: str,
    severity: str,
    code: str,
    message: str,
    job_id: str = "",
    endpoint_id: str = "",
    step_index: Any = None,
    dependency: str = "",
    component: str = "",
    table: str = "",
) -> JSONDict:
    return {
        "stage": stage,
        "severity": severity,
        "code": code,
        "message": message,
        "job_id": job_id,
        "endpoint_id": endpoint_id,
        "step_index": _int_or_none(step_index),
        "dependency": dependency,
        "component": component,
        "table": table,
        "blocking": severity == "error",
    }


def _diagnostic_summary(diagnostics: Sequence[Mapping[str, Any]]) -> JSONDict:
    blocking = [diagnostic for diagnostic in diagnostics if diagnostic.get("blocking")]
    return {
        "diagnostic_count": len(diagnostics),
        "blocking_diagnostic_count": len(blocking),
        "by_stage": _count_by(diagnostics, "stage"),
        "by_severity": _count_by(diagnostics, "severity"),
        "by_code": _count_by(diagnostics, "code"),
        "blocking_by_code": _count_by(blocking, "code"),
    }


def _count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _to_dict(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    return to_dict() if callable(to_dict) else value


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _digest(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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
            default=str,
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
    "DEPLOYMENT_RUNTIME_REPORT_VERSION",
    "PhysicsIngestDeploymentRuntimeReport",
    "build_physics_ingest_deployment_runtime_report",
    "build_physics_ingest_deployment_runtime_report_dict",
]
