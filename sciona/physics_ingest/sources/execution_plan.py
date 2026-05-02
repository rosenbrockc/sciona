"""Side-effect-free source adapter execution readiness reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from sciona.physics_ingest.sources._manifest import jsonable
from sciona.physics_ingest.sources.retrieval_plan import (
    RetrievalRunPlan,
    build_physics_source_retrieval_manifest,
)


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class SourceExecutionDiagnostic:
    """Deterministic readiness diagnostic for a source retrieval step."""

    severity: str
    code: str
    message: str
    step_index: int | None = None
    job_id: str = ""

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class SourceExecutionReadinessStep:
    """Source adapter execution readiness for one retrieval run step."""

    step_index: int
    job_id: str
    status: str
    status_reason: str
    source_system: str
    source_family: str
    phase7_ring: str
    phase7_ring_order: int
    phase7_rings: tuple[str, ...]
    snapshot_key: str
    required_adapter: Mapping[str, str]
    endpoint: Mapping[str, Any]
    replay_key: str
    payload_expectation: Mapping[str, Any]
    storage_expectation: Mapping[str, Any]
    diagnostics: tuple[SourceExecutionDiagnostic, ...] = ()

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class SourceExecutionReadinessReport:
    """JSON-safe readiness report for source adapter execution."""

    report_version: str
    manifest_version: str
    snapshot_key_prefix: str
    dry_run: bool
    summary: Mapping[str, int]
    steps: tuple[SourceExecutionReadinessStep, ...]
    diagnostics: tuple[SourceExecutionDiagnostic, ...] = ()

    def to_dict(self) -> JSONDict:
        return jsonable(self)


def build_source_execution_readiness_report(
    plan: RetrievalRunPlan | Mapping[str, Any],
) -> SourceExecutionReadinessReport:
    """Describe source adapter execution readiness without network or DB IO."""

    plan_dict = _plan_to_dict(plan)
    supported_job_ids = {
        job.job_id for job in build_physics_source_retrieval_manifest().jobs
    }
    dry_run = bool(plan_dict.get("dry_run"))
    report_diagnostics: list[SourceExecutionDiagnostic] = []
    if not dry_run:
        report_diagnostics.append(
            SourceExecutionDiagnostic(
                severity="error",
                code="non_dry_run_plan",
                message="source execution readiness requires a dry-run retrieval plan",
            )
        )

    readiness_steps: list[SourceExecutionReadinessStep] = []
    raw_steps = plan_dict.get("steps", ())
    if not isinstance(raw_steps, list | tuple):
        raw_steps = ()
    for fallback_index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, Mapping):
            continue
        step = dict(raw_step)
        readiness_step = _readiness_step(
            step=step,
            fallback_index=fallback_index,
            plan_dry_run=dry_run,
            supported_job_ids=supported_job_ids,
        )
        readiness_steps.append(readiness_step)
        report_diagnostics.extend(readiness_step.diagnostics)

    summary = {
        "total_steps": len(readiness_steps),
        "executable": _count_status(readiness_steps, "executable"),
        "offline_blocked": _count_status(readiness_steps, "offline_blocked"),
        "manual": _count_status(readiness_steps, "manual"),
        "diagnostic_count": len(report_diagnostics),
    }
    return SourceExecutionReadinessReport(
        report_version="physics-source-execution-readiness.v1",
        manifest_version=str(plan_dict.get("manifest_version", "")),
        snapshot_key_prefix=str(plan_dict.get("snapshot_key_prefix", "")),
        dry_run=dry_run,
        summary=summary,
        steps=tuple(readiness_steps),
        diagnostics=tuple(report_diagnostics),
    )


def build_source_execution_readiness_report_dict(
    plan: RetrievalRunPlan | Mapping[str, Any],
) -> JSONDict:
    """Return ``build_source_execution_readiness_report(plan).to_dict()``."""

    return build_source_execution_readiness_report(plan).to_dict()


def _readiness_step(
    *,
    step: Mapping[str, Any],
    fallback_index: int,
    plan_dry_run: bool,
    supported_job_ids: set[str],
) -> SourceExecutionReadinessStep:
    step_index = _int_value(step.get("step_index"), fallback_index)
    job_id = str(step.get("job_id", ""))
    method = str(step.get("method", ""))
    url = str(step.get("url", ""))
    adapter_name = str(step.get("adapter_module") or step.get("adapter_name") or "")
    adapter_version = str(step.get("adapter_version", ""))
    target_adapter_input = str(step.get("target_adapter_input", ""))
    endpoint_kind = str(step.get("endpoint_kind", ""))
    is_manual = method == "MANUAL" or url.startswith("manual://")
    diagnostics = list(
        _step_diagnostics(
            step_index=step_index,
            job_id=job_id,
            plan_dry_run=plan_dry_run,
            supported_job_ids=supported_job_ids,
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            url=url,
        )
    )
    status, status_reason = _status(
        diagnostics=diagnostics,
        is_manual=is_manual,
    )
    return SourceExecutionReadinessStep(
        step_index=step_index,
        job_id=job_id,
        status=status,
        status_reason=status_reason,
        source_system=str(step.get("source_system", "")),
        source_family=str(step.get("source_family", "")),
        phase7_ring=str(step.get("phase7_ring", "")),
        phase7_ring_order=_int_value(step.get("phase7_ring_order"), 0),
        phase7_rings=_string_tuple(step.get("phase7_rings", ())),
        snapshot_key=str(step.get("snapshot_key", "")),
        required_adapter={
            "name": adapter_name,
            "version": adapter_version,
            "target_input": target_adapter_input,
        },
        endpoint={
            "endpoint_id": str(step.get("endpoint_id", "")),
            "method": method,
            "url": url,
            "kind": endpoint_kind,
            "content_type": str(step.get("content_type", "")),
            "requires_auth": bool(step.get("requires_auth", False)),
            "auth_hint": str(step.get("auth_hint", "")),
            "params": jsonable(step.get("params", {})),
            "paging": jsonable(step.get("paging", {})),
            "headers": jsonable(step.get("headers", {})),
        },
        replay_key=str(step.get("replay_key", "")),
        payload_expectation={
            "target_adapter_input": target_adapter_input,
            "payload_required": not is_manual,
            "offline_payload_available": is_manual,
            "content_type": str(step.get("content_type", "")),
            "body_template": jsonable(step.get("body_template", {})),
        },
        storage_expectation={
            "snapshot_key": str(step.get("snapshot_key", "")),
            "write_required": False,
            "storage_required": not is_manual,
            "side_effect_free": True,
        },
        diagnostics=tuple(diagnostics),
    )


def _step_diagnostics(
    *,
    step_index: int,
    job_id: str,
    plan_dry_run: bool,
    supported_job_ids: set[str],
    adapter_name: str,
    adapter_version: str,
    url: str,
) -> tuple[SourceExecutionDiagnostic, ...]:
    diagnostics: list[SourceExecutionDiagnostic] = []
    if not plan_dry_run:
        diagnostics.append(
            SourceExecutionDiagnostic(
                severity="error",
                code="non_dry_run_plan",
                message="step belongs to a non-dry-run retrieval plan",
                step_index=step_index,
                job_id=job_id,
            )
        )
    if job_id not in supported_job_ids:
        diagnostics.append(
            SourceExecutionDiagnostic(
                severity="error",
                code="unsupported_job_id",
                message="retrieval job id is not supported by the current source manifest",
                step_index=step_index,
                job_id=job_id,
            )
        )
    if not adapter_name:
        diagnostics.append(
            SourceExecutionDiagnostic(
                severity="error",
                code="missing_adapter_name",
                message="required adapter name is missing",
                step_index=step_index,
                job_id=job_id,
            )
        )
    if not adapter_version:
        diagnostics.append(
            SourceExecutionDiagnostic(
                severity="error",
                code="missing_adapter_version",
                message="required adapter version is missing",
                step_index=step_index,
                job_id=job_id,
            )
        )
    if not url:
        diagnostics.append(
            SourceExecutionDiagnostic(
                severity="error",
                code="missing_endpoint_url",
                message="endpoint URL is missing",
                step_index=step_index,
                job_id=job_id,
            )
        )
    return tuple(diagnostics)


def _status(
    *,
    diagnostics: list[SourceExecutionDiagnostic],
    is_manual: bool,
) -> tuple[str, str]:
    if any(diagnostic.severity == "error" for diagnostic in diagnostics):
        return "offline_blocked", "diagnostics must be resolved before execution"
    if is_manual:
        return "manual", "manual source adapter can emit curated offline payloads"
    return "executable", "adapter metadata and endpoint contract are complete"


def _plan_to_dict(plan: RetrievalRunPlan | Mapping[str, Any]) -> JSONDict:
    if isinstance(plan, Mapping):
        return jsonable(plan)
    return plan.to_dict()


def _count_status(
    steps: list[SourceExecutionReadinessStep],
    status: str,
) -> int:
    return sum(1 for step in steps if step.status == status)


def _int_value(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return ()
