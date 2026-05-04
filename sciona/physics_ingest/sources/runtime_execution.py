"""Production execution/preflight surface for source retrieval runtime jobs.

This module keeps concrete HTTP and storage construction outside the ingestion
package.  Callers inject already-created objects, receive deterministic
preflight metadata, and opt in explicitly before the low-level executor is
invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from sciona.physics_ingest.sources._manifest import jsonable
from sciona.physics_ingest.sources.executor import (
    SourceRetrievalExecutionReport,
    execute_source_retrieval_plan,
)
from sciona.physics_ingest.sources.retrieval_plan import RetrievalRunPlan
from sciona.physics_ingest.sources.runtime_adapters import (
    SourceRetrievalRuntimeAdapterBundle,
    build_source_retrieval_runtime_adapters,
)


JSONDict = dict[str, Any]
SummaryCounts = dict[str, int]
SourceRetrievalRuntimeExecutionSummary = Mapping[str, bool | int | str | SummaryCounts]

RUNTIME_EXECUTION_REPORT_VERSION = "physics-source-runtime-execution.v1"


@dataclass(frozen=True)
class SourceRetrievalRuntimeExecutionDiagnostic:
    """Deterministic diagnostic for runtime dependency preflight."""

    severity: str
    code: str
    message: str
    step_index: int | None = None
    job_id: str = ""
    dependency: str = ""

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class SourceRetrievalRuntimeExecutionStep:
    """Runtime dependency metadata for one retrieval step."""

    step_index: int
    job_id: str
    endpoint_id: str
    mode: str
    dry_run: bool
    requires_http_client: bool
    requires_snapshot_sink: bool
    requires_auth: bool
    executor_kwargs_required: tuple[str, ...]
    diagnostics: tuple[SourceRetrievalRuntimeExecutionDiagnostic, ...] = ()

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class SourceRetrievalRuntimeExecutionReport:
    """JSON-safe report for source retrieval runtime preflight/execution."""

    report_version: str
    manifest_version: str
    snapshot_key_prefix: str
    dry_run: bool
    preflight: bool
    execution_requested: bool
    execution_performed: bool
    execution_skipped_reason: str
    side_effect_free: bool
    summary: SourceRetrievalRuntimeExecutionSummary
    executor_kwargs: Mapping[str, Any]
    adapter_report: Mapping[str, Any]
    steps: tuple[SourceRetrievalRuntimeExecutionStep, ...]
    diagnostics: tuple[SourceRetrievalRuntimeExecutionDiagnostic, ...] = ()
    execution_report: SourceRetrievalExecutionReport | None = None

    def to_dict(self) -> JSONDict:
        return _json_safe(
            {
                "report_version": self.report_version,
                "manifest_version": self.manifest_version,
                "snapshot_key_prefix": self.snapshot_key_prefix,
                "dry_run": self.dry_run,
                "preflight": self.preflight,
                "execution_requested": self.execution_requested,
                "execution_performed": self.execution_performed,
                "execution_skipped_reason": self.execution_skipped_reason,
                "side_effect_free": self.side_effect_free,
                "summary": self.summary,
                "executor_kwargs": self.executor_kwargs,
                "adapter_report": self.adapter_report,
                "steps": [step.to_dict() for step in self.steps],
                "diagnostics": [
                    diagnostic.to_dict() for diagnostic in self.diagnostics
                ],
                "execution_report": (
                    self.execution_report.to_dict()
                    if self.execution_report is not None
                    else None
                ),
            }
        )


def build_source_retrieval_runtime_execution_report(
    plan: RetrievalRunPlan | Mapping[str, Any],
    *,
    http_client: Any | None = None,
    snapshot_sink: Any | None = None,
    headers: Mapping[str, Any] | None = None,
    auth_headers: Mapping[str, Any] | None = None,
    auth: Any | None = None,
    execute: bool = False,
    preflight: bool = True,
) -> SourceRetrievalRuntimeExecutionReport:
    """Build adapter metadata and optionally execute a retrieval run plan.

    ``preflight=True`` always prevents executor invocation.  ``execute=True``
    invokes ``execute_source_retrieval_plan`` only after dependency diagnostics
    have no blocking errors.
    """

    plan_dict = _plan_to_dict(plan)
    dry_run = bool(plan_dict.get("dry_run"))
    bundle = build_source_retrieval_runtime_adapters(
        http_client=http_client,
        snapshot_sink=snapshot_sink,
        headers=headers,
        auth_headers=auth_headers,
        auth=auth,
        dry_run=dry_run,
        preflight=preflight,
    )
    raw_steps = _raw_steps(plan_dict)
    steps = tuple(
        _runtime_step(
            raw_step=raw_step,
            fallback_index=fallback_index,
            plan_dry_run=dry_run,
            http_client=http_client,
            snapshot_sink=snapshot_sink,
            auth_supplied=_auth_supplied(
                headers=headers,
                auth_headers=auth_headers,
                auth=auth,
            ),
        )
        for fallback_index, raw_step in enumerate(raw_steps, start=1)
        if isinstance(raw_step, Mapping)
    )
    diagnostics = tuple(
        diagnostic for step in steps for diagnostic in step.diagnostics
    )
    blocking_errors = tuple(
        diagnostic for diagnostic in diagnostics if diagnostic.severity == "error"
    )

    execution_report: SourceRetrievalExecutionReport | None = None
    execution_skipped_reason = _execution_skipped_reason(
        execute=execute,
        preflight=preflight,
        blocking_errors=blocking_errors,
    )
    if execute and not preflight and not blocking_errors:
        execution_report = execute_source_retrieval_plan(
            plan,
            **bundle.execute_kwargs(),
        )
        execution_skipped_reason = ""

    execution_performed = execution_report is not None
    return SourceRetrievalRuntimeExecutionReport(
        report_version=RUNTIME_EXECUTION_REPORT_VERSION,
        manifest_version=str(plan_dict.get("manifest_version", "")),
        snapshot_key_prefix=str(plan_dict.get("snapshot_key_prefix", "")),
        dry_run=dry_run,
        preflight=preflight,
        execution_requested=execute,
        execution_performed=execution_performed,
        execution_skipped_reason=execution_skipped_reason,
        side_effect_free=_side_effect_free(
            execution_performed=execution_performed,
            execution_report=execution_report,
        ),
        summary=_summary(
            steps=steps,
            diagnostics=diagnostics,
            execution_performed=execution_performed,
            execution_report=execution_report,
        ),
        executor_kwargs=_executor_kwargs_metadata(bundle),
        adapter_report=bundle.report.to_dict(),
        steps=steps,
        diagnostics=diagnostics,
        execution_report=execution_report,
    )


def build_source_retrieval_runtime_execution_report_dict(
    plan: RetrievalRunPlan | Mapping[str, Any],
    **kwargs: Any,
) -> JSONDict:
    """Return ``build_source_retrieval_runtime_execution_report(...).to_dict()``."""

    return build_source_retrieval_runtime_execution_report(plan, **kwargs).to_dict()


def _runtime_step(
    *,
    raw_step: Mapping[str, Any],
    fallback_index: int,
    plan_dry_run: bool,
    http_client: Any | None,
    snapshot_sink: Any | None,
    auth_supplied: bool,
) -> SourceRetrievalRuntimeExecutionStep:
    step = dict(raw_step)
    envelope = step.get("request_envelope", {})
    if not isinstance(envelope, Mapping):
        envelope = {}
    execution = envelope.get("execution", {})
    if not isinstance(execution, Mapping):
        execution = {}
    storage = envelope.get("storage", {})
    if not isinstance(storage, Mapping):
        storage = {}

    step_index = _int_value(step.get("step_index"), fallback_index)
    job_id = str(step.get("job_id", ""))
    endpoint_id = str(step.get("endpoint_id", ""))
    method = str(envelope.get("method") or step.get("method") or "")
    url = str(envelope.get("url") or step.get("url") or "")
    dry_run = bool(execution.get("dry_run", plan_dry_run))
    manual_source = _is_manual_step(method=method, url=url, execution=execution)
    requires_http_client = not dry_run and not manual_source and bool(
        execution.get("network_required", True)
    )
    requires_snapshot_sink = not dry_run and not manual_source and bool(
        storage.get("storage_required", True)
    )
    requires_auth = requires_http_client and bool(step.get("requires_auth", False))
    diagnostics = _dependency_diagnostics(
        step_index=step_index,
        job_id=job_id,
        requires_http_client=requires_http_client,
        requires_snapshot_sink=requires_snapshot_sink,
        requires_auth=requires_auth,
        http_client=http_client,
        snapshot_sink=snapshot_sink,
        auth_supplied=auth_supplied,
    )
    return SourceRetrievalRuntimeExecutionStep(
        step_index=step_index,
        job_id=job_id,
        endpoint_id=endpoint_id,
        mode="manual" if manual_source else ("dry_run" if dry_run else "network"),
        dry_run=dry_run,
        requires_http_client=requires_http_client,
        requires_snapshot_sink=requires_snapshot_sink,
        requires_auth=requires_auth,
        executor_kwargs_required=_executor_kwargs_required(
            requires_http_client=requires_http_client,
            requires_snapshot_sink=requires_snapshot_sink,
        ),
        diagnostics=diagnostics,
    )


def _dependency_diagnostics(
    *,
    step_index: int,
    job_id: str,
    requires_http_client: bool,
    requires_snapshot_sink: bool,
    requires_auth: bool,
    http_client: Any | None,
    snapshot_sink: Any | None,
    auth_supplied: bool,
) -> tuple[SourceRetrievalRuntimeExecutionDiagnostic, ...]:
    diagnostics: list[SourceRetrievalRuntimeExecutionDiagnostic] = []
    if requires_http_client and http_client is None:
        diagnostics.append(
            SourceRetrievalRuntimeExecutionDiagnostic(
                severity="error",
                code="missing_http_client",
                message="network retrieval step requires an injected HTTP client",
                step_index=step_index,
                job_id=job_id,
                dependency="http_client",
            )
        )
    elif requires_http_client and not _supports_http_client(http_client):
        diagnostics.append(
            SourceRetrievalRuntimeExecutionDiagnostic(
                severity="error",
                code="unsupported_http_client",
                message="injected HTTP client must expose request() or be callable",
                step_index=step_index,
                job_id=job_id,
                dependency="http_client",
            )
        )

    if requires_snapshot_sink and snapshot_sink is None:
        diagnostics.append(
            SourceRetrievalRuntimeExecutionDiagnostic(
                severity="error",
                code="missing_snapshot_sink",
                message="network retrieval step requires an injected snapshot sink",
                step_index=step_index,
                job_id=job_id,
                dependency="snapshot_sink",
            )
        )
    elif requires_snapshot_sink and not _supports_snapshot_sink(snapshot_sink):
        diagnostics.append(
            SourceRetrievalRuntimeExecutionDiagnostic(
                severity="error",
                code="unsupported_snapshot_sink",
                message=(
                    "injected snapshot sink must expose write/store/put or be callable"
                ),
                step_index=step_index,
                job_id=job_id,
                dependency="snapshot_sink",
            )
        )

    if requires_auth and not auth_supplied:
        diagnostics.append(
            SourceRetrievalRuntimeExecutionDiagnostic(
                severity="error",
                code="missing_auth",
                message="authenticated retrieval step requires injected auth options",
                step_index=step_index,
                job_id=job_id,
                dependency="auth",
            )
        )
    return tuple(diagnostics)


def _executor_kwargs_metadata(
    bundle: SourceRetrievalRuntimeAdapterBundle,
) -> JSONDict:
    return {
        "http_client": {
            "kwarg": "http_client",
            "supplied": bundle.http_client is not None,
            "adapter": (
                type(bundle.http_client).__name__
                if bundle.http_client is not None
                else ""
            ),
        },
        "snapshot_sink": {
            "kwarg": "snapshot_sink",
            "supplied": bundle.snapshot_sink is not None,
            "adapter": (
                type(bundle.snapshot_sink).__name__
                if bundle.snapshot_sink is not None
                else ""
            ),
        },
    }


def _summary(
    *,
    steps: tuple[SourceRetrievalRuntimeExecutionStep, ...],
    diagnostics: tuple[SourceRetrievalRuntimeExecutionDiagnostic, ...],
    execution_performed: bool,
    execution_report: SourceRetrievalExecutionReport | None,
) -> SourceRetrievalRuntimeExecutionSummary:
    execution_summary = execution_report.summary if execution_report is not None else {}
    return {
        "total_steps": len(steps),
        "network_step_count": sum(1 for step in steps if step.mode == "network"),
        "manual_step_count": sum(1 for step in steps if step.mode == "manual"),
        "dry_run_step_count": sum(1 for step in steps if step.dry_run),
        "requires_http_client_count": sum(
            1 for step in steps if step.requires_http_client
        ),
        "requires_snapshot_sink_count": sum(
            1 for step in steps if step.requires_snapshot_sink
        ),
        "requires_auth_count": sum(1 for step in steps if step.requires_auth),
        "diagnostic_count": len(diagnostics),
        "blocking_diagnostic_count": sum(
            1 for diagnostic in diagnostics if diagnostic.severity == "error"
        ),
        "execution_performed": execution_performed,
        "execution_result_count": int(execution_summary.get("total_steps", 0)),
        "by_diagnostic_code": _count_by_string(
            diagnostic.code for diagnostic in diagnostics
        ),
        "by_diagnostic_severity": _count_by_string(
            diagnostic.severity for diagnostic in diagnostics
        ),
    }


def _execution_skipped_reason(
    *,
    execute: bool,
    preflight: bool,
    blocking_errors: tuple[SourceRetrievalRuntimeExecutionDiagnostic, ...],
) -> str:
    if not execute:
        return "execution_not_requested"
    if preflight:
        return "preflight_only"
    if blocking_errors:
        return "blocking_diagnostics"
    return ""


def _side_effect_free(
    *,
    execution_performed: bool,
    execution_report: SourceRetrievalExecutionReport | None,
) -> bool:
    if not execution_performed or execution_report is None:
        return True
    return not any(
        bool(result.execution.get("network_io_performed"))
        or bool(result.execution.get("storage_io_performed"))
        for result in execution_report.results
    )


def _executor_kwargs_required(
    *,
    requires_http_client: bool,
    requires_snapshot_sink: bool,
) -> tuple[str, ...]:
    required = []
    if requires_http_client:
        required.append("http_client")
    if requires_snapshot_sink:
        required.append("snapshot_sink")
    return tuple(required)


def _auth_supplied(
    *,
    headers: Mapping[str, Any] | None,
    auth_headers: Mapping[str, Any] | None,
    auth: Any | None,
) -> bool:
    if auth is not None or auth_headers:
        return True
    header_keys = {str(key).lower() for key in (headers or {})}
    return bool(header_keys & {"authorization", "x-api-key", "api-key"})


def _supports_http_client(target: Any | None) -> bool:
    return callable(getattr(target, "request", None)) or callable(target)


def _supports_snapshot_sink(target: Any | None) -> bool:
    return (
        callable(getattr(target, "write", None))
        or callable(getattr(target, "store", None))
        or callable(getattr(target, "put", None))
        or callable(target)
    )


def _is_manual_step(
    *,
    method: str,
    url: str,
    execution: Mapping[str, Any],
) -> bool:
    return (
        bool(execution.get("manual_source"))
        or str(execution.get("mode", "")) == "manual"
        or method == "MANUAL"
        or url.startswith("manual://")
    )


def _raw_steps(plan_dict: Mapping[str, Any]) -> tuple[Any, ...]:
    raw_steps = plan_dict.get("steps", ())
    if isinstance(raw_steps, list | tuple):
        return tuple(raw_steps)
    return ()


def _plan_to_dict(plan: RetrievalRunPlan | Mapping[str, Any]) -> JSONDict:
    if isinstance(plan, Mapping):
        return jsonable(plan)
    return plan.to_dict()


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_by_string(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _json_safe(value: Any) -> Any:
    return json.loads(
        json.dumps(
            jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
    )


__all__ = [
    "SourceRetrievalRuntimeExecutionDiagnostic",
    "SourceRetrievalRuntimeExecutionReport",
    "SourceRetrievalRuntimeExecutionStep",
    "build_source_retrieval_runtime_execution_report",
    "build_source_retrieval_runtime_execution_report_dict",
]
