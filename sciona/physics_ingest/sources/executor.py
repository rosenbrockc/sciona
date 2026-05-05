"""Executor boundary for physics source retrieval plans.

The executor is side-effect-free unless callers inject the effects:
HTTP/fetch clients for network requests and snapshot sinks for persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sciona.physics_ingest.sources._manifest import jsonable, stable_payload_sha256
from sciona.physics_ingest.sources.retrieval_plan import RetrievalRunPlan


JSONDict = dict[str, Any]
SummaryCounts = dict[str, int]
SourceRetrievalExecutorSummary = Mapping[str, int | SummaryCounts]


@dataclass(frozen=True)
class SourceRetrievalExecutionResult:
    """JSON-safe execution result for one retrieval run step."""

    step_index: int
    job_id: str
    endpoint_id: str
    status: str
    status_reason: str
    source_system: str
    source_family: str
    snapshot_key: str
    replay_key: str
    adapter_target: Mapping[str, Any]
    execution: Mapping[str, Any]
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    payload: Mapping[str, Any]
    storage: Mapping[str, Any]

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class SourceRetrievalExecutionReport:
    """JSON-safe source retrieval execution report."""

    report_version: str
    manifest_version: str
    snapshot_key_prefix: str
    dry_run: bool
    summary: SourceRetrievalExecutorSummary
    results: tuple[SourceRetrievalExecutionResult, ...]

    def to_dict(self) -> JSONDict:
        return jsonable(self)


def execute_source_retrieval_plan(
    plan: RetrievalRunPlan | Mapping[str, Any],
    *,
    http_client: Any | None = None,
    snapshot_sink: Any | None = None,
) -> SourceRetrievalExecutionReport:
    """Execute or simulate a retrieval plan through injected IO boundaries.

    Dry-run plans never call ``http_client`` or ``snapshot_sink``. Manual
    source steps are reported as offline work and also do not require injected
    IO. Network steps in non-dry-run plans require an injected object exposing
    ``request(method, url, headers=..., params=..., json=.../data=...)``.
    """

    plan_dict = _plan_to_dict(plan)
    plan_dry_run = bool(plan_dict.get("dry_run"))
    raw_steps = plan_dict.get("steps", ())
    if not isinstance(raw_steps, list | tuple):
        raw_steps = ()

    results: list[SourceRetrievalExecutionResult] = []
    for fallback_index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, Mapping):
            continue
        results.append(
            _execute_step(
                step=dict(raw_step),
                fallback_index=fallback_index,
                plan_dry_run=plan_dry_run,
                http_client=http_client,
                snapshot_sink=snapshot_sink,
            )
        )

    return SourceRetrievalExecutionReport(
        report_version="physics-source-retrieval-executor.v1",
        manifest_version=str(plan_dict.get("manifest_version", "")),
        snapshot_key_prefix=str(plan_dict.get("snapshot_key_prefix", "")),
        dry_run=plan_dry_run,
        summary=_summary(results),
        results=tuple(results),
    )


def execute_source_retrieval_plan_dict(
    plan: RetrievalRunPlan | Mapping[str, Any],
    *,
    http_client: Any | None = None,
    snapshot_sink: Any | None = None,
) -> JSONDict:
    """Return ``execute_source_retrieval_plan(...).to_dict()``."""

    return execute_source_retrieval_plan(
        plan,
        http_client=http_client,
        snapshot_sink=snapshot_sink,
    ).to_dict()


def _execute_step(
    *,
    step: Mapping[str, Any],
    fallback_index: int,
    plan_dry_run: bool,
    http_client: Any | None,
    snapshot_sink: Any | None,
) -> SourceRetrievalExecutionResult:
    envelope = _request_envelope(step)
    execution = _execution_contract(envelope, plan_dry_run=plan_dry_run)
    adapter_target = _adapter_target(step=step, envelope=envelope)
    replay_key = str(envelope.get("replay_key") or step.get("replay_key") or "")
    snapshot_key = str(envelope.get("snapshot_key") or step.get("snapshot_key") or "")
    method = str(envelope.get("method") or step.get("method") or "")
    url = str(envelope.get("url") or step.get("url") or "")
    is_manual = _is_manual_step(method=method, url=url, execution=execution)

    base_kwargs = {
        "step_index": _int_value(step.get("step_index"), fallback_index),
        "job_id": str(step.get("job_id", "")),
        "endpoint_id": str(step.get("endpoint_id", "")),
        "source_system": str(step.get("source_system", "")),
        "source_family": str(step.get("source_family", "")),
        "snapshot_key": snapshot_key,
        "replay_key": replay_key,
        "adapter_target": adapter_target,
    }

    if is_manual:
        return SourceRetrievalExecutionResult(
            **base_kwargs,
            status="manual_offline",
            status_reason="manual source step is offline; no HTTP or storage required",
            execution={
                **execution,
                "mode": "manual/offline",
                "io_performed": False,
                "network_io_performed": False,
                "storage_io_performed": False,
            },
            request=_request_summary(envelope=envelope, method=method, url=url, status="offline"),
            response={},
            payload={
                "available": False,
                "payload_sha256": "",
                "status_metadata": {},
            },
            storage={
                "status": "not_required",
                "snapshot_key": snapshot_key,
                "side_effect_free": True,
            },
        )

    if bool(execution.get("dry_run")):
        return SourceRetrievalExecutionResult(
            **base_kwargs,
            status="dry_run",
            status_reason="dry run; HTTP and storage were not executed",
            execution={
                **execution,
                "mode": "dry_run",
                "io_performed": False,
                "network_io_performed": False,
                "storage_io_performed": False,
            },
            request=_request_summary(envelope=envelope, method=method, url=url, status="planned"),
            response={},
            payload={
                "available": False,
                "payload_sha256": "",
                "status_metadata": {},
            },
            storage={
                "status": "dry_run",
                "snapshot_key": snapshot_key,
                "side_effect_free": True,
            },
        )

    if http_client is None:
        return SourceRetrievalExecutionResult(
            **base_kwargs,
            status="blocked_missing_client",
            status_reason="network step requires an injected HTTP client",
            execution={
                **execution,
                "mode": "network",
                "io_performed": False,
                "network_io_performed": False,
                "storage_io_performed": False,
            },
            request=_request_summary(envelope=envelope, method=method, url=url, status="blocked"),
            response={},
            payload={
                "available": False,
                "payload_sha256": "",
                "status_metadata": {},
            },
            storage={
                "status": "not_attempted",
                "snapshot_key": snapshot_key,
                "side_effect_free": True,
            },
        )

    request_kwargs = _request_kwargs(envelope=envelope, step=step)
    response_obj = http_client.request(method, url, **request_kwargs)
    response_payload = _response_payload(response_obj)
    response_metadata = _response_metadata(response_obj)
    payload_sha256 = stable_payload_sha256(response_payload)
    payload_metadata = {
        "payload_sha256": payload_sha256,
        "response_status": response_metadata.get("status_code"),
        "content_type": response_metadata.get("content_type", ""),
    }
    storage_result = _store_payload(
        snapshot_sink=snapshot_sink,
        snapshot_key=snapshot_key,
        replay_key=replay_key,
        payload=response_payload,
        metadata={
            **payload_metadata,
            "adapter_target": adapter_target,
            "request": _request_summary(
                envelope=envelope,
                method=method,
                url=url,
                status="executed",
            ),
        },
    )

    return SourceRetrievalExecutionResult(
        **base_kwargs,
        status="executed",
        status_reason="HTTP client returned a response",
        execution={
            **execution,
            "mode": "network",
            "io_performed": True,
            "network_io_performed": True,
            "storage_io_performed": storage_result["status"] == "written",
        },
        request=_request_summary(envelope=envelope, method=method, url=url, status="executed"),
        response=response_metadata,
        payload={
            "available": True,
            "payload_sha256": payload_sha256,
            "status_metadata": payload_metadata,
        },
        storage=storage_result,
    )


def _request_envelope(step: Mapping[str, Any]) -> JSONDict:
    raw_envelope = step.get("request_envelope", {})
    if isinstance(raw_envelope, Mapping):
        return jsonable(raw_envelope)
    return {}


def _execution_contract(
    envelope: Mapping[str, Any],
    *,
    plan_dry_run: bool,
) -> JSONDict:
    execution = envelope.get("execution", {})
    if isinstance(execution, Mapping):
        result = dict(jsonable(execution))
    else:
        result = {}
    result["dry_run"] = bool(result.get("dry_run", plan_dry_run))
    result.setdefault("network_required", True)
    result.setdefault("network_io_allowed", not result["dry_run"])
    result.setdefault("manual_source", False)
    return result


def _adapter_target(
    *,
    step: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> JSONDict:
    envelope_target = envelope.get("adapter_target", {})
    if isinstance(envelope_target, Mapping):
        return jsonable(envelope_target)
    return {
        "module": str(step.get("adapter_module") or step.get("adapter_name") or ""),
        "version": str(step.get("adapter_version", "")),
        "target_input": str(step.get("target_adapter_input", "")),
    }


def _request_kwargs(
    *,
    envelope: Mapping[str, Any],
    step: Mapping[str, Any],
) -> JSONDict:
    kwargs: JSONDict = {
        "headers": jsonable(envelope.get("headers", {})),
        "params": jsonable(envelope.get("params", {})),
    }
    method = str(envelope.get("method") or step.get("method") or "").upper()
    body_template = jsonable(envelope.get("body_template", {}))
    if method in {"POST", "PUT", "PATCH"} and body_template:
        rendered_body = _render_body_template(
            body_template=body_template,
            envelope=envelope,
            step=step,
        )
        if isinstance(rendered_body, Mapping) and str(
            rendered_body.get("body_kind") or ""
        ) == "form":
            kwargs["data"] = rendered_body.get("data", {})
            return kwargs
        content_type = str(step.get("content_type") or "").lower()
        if content_type and "json" not in content_type:
            kwargs["data"] = rendered_body
        else:
            kwargs["json"] = rendered_body
    return kwargs


def _render_body_template(
    *,
    body_template: Mapping[str, Any],
    envelope: Mapping[str, Any],
    step: Mapping[str, Any],
) -> Any:
    query_builder = str(body_template.get("query_builder") or "")
    if query_builder == "build_physics_ingestion_candidate_query":
        from sciona.physics_ingest.sources.wikidata import (
            build_physics_ingestion_candidate_query,
        )

        return {
            "body_kind": "form",
            "data": {
                "query": build_physics_ingestion_candidate_query(
                    limit=_request_limit(envelope=envelope, step=step)
                ),
                "format": "json",
            },
        }
    if query_builder == "build_physical_equation_candidates_query":
        from sciona.physics_ingest.sources.wikidata import (
            build_physical_equation_candidates_query,
        )

        return {
            "body_kind": "form",
            "data": {
                "query": build_physical_equation_candidates_query(
                    limit=_request_limit(envelope=envelope, step=step)
                ),
                "format": "json",
            },
        }
    return body_template


def _request_limit(*, envelope: Mapping[str, Any], step: Mapping[str, Any]) -> int:
    paging = envelope.get("paging")
    if isinstance(paging, Mapping):
        value = paging.get("limit")
        if value is not None:
            return max(_int_value(value, 500), 1)
    params = envelope.get("params")
    if isinstance(params, Mapping):
        value = params.get("LIMIT") or params.get("limit")
        if value is not None:
            return max(_int_value(value, 500), 1)
    value = step.get("limit")
    if value is not None:
        return max(_int_value(value, 500), 1)
    return 500


def _request_summary(
    *,
    envelope: Mapping[str, Any],
    method: str,
    url: str,
    status: str,
) -> JSONDict:
    return {
        "status": status,
        "method": method,
        "url": url,
        "headers": jsonable(envelope.get("headers", {})),
        "params": jsonable(envelope.get("params", {})),
        "body_template": jsonable(envelope.get("body_template", {})),
    }


def _response_payload(response: Any) -> Any:
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            return json_method()
        except Exception:
            pass
    if isinstance(response, Mapping | list | tuple | str | int | float | bool) or response is None:
        return response
    text = getattr(response, "text", None)
    if text is not None:
        return text
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if content is not None:
        return content
    return {"repr": repr(response)}


def _response_metadata(response: Any) -> JSONDict:
    status_code = getattr(response, "status_code", getattr(response, "status", None))
    headers = getattr(response, "headers", {})
    content_type = ""
    if isinstance(headers, Mapping):
        content_type = str(headers.get("content-type") or headers.get("Content-Type") or "")
    return {
        "status_code": status_code,
        "ok": _response_ok(status_code),
        "headers": jsonable(headers) if isinstance(headers, Mapping) else {},
        "content_type": content_type,
    }


def _response_ok(status_code: Any) -> bool | None:
    if status_code is None:
        return None
    try:
        return int(status_code) < 400
    except (TypeError, ValueError):
        return None


def _store_payload(
    *,
    snapshot_sink: Any | None,
    snapshot_key: str,
    replay_key: str,
    payload: Any,
    metadata: Mapping[str, Any],
) -> JSONDict:
    if snapshot_sink is None:
        return {
            "status": "not_persisted",
            "reason": "snapshot_sink_not_supplied",
            "snapshot_key": snapshot_key,
            "side_effect_free": True,
        }

    writer = _snapshot_writer(snapshot_sink)
    receipt = writer(
        snapshot_key=snapshot_key,
        replay_key=replay_key,
        payload=payload,
        metadata=jsonable(metadata),
    )
    return {
        "status": "written",
        "snapshot_key": snapshot_key,
        "receipt": jsonable(receipt) if receipt is not None else {},
        "side_effect_free": False,
    }


def _snapshot_writer(snapshot_sink: Any) -> Any:
    for method_name in ("write", "store", "put"):
        writer = getattr(snapshot_sink, method_name, None)
        if callable(writer):
            return writer
    if callable(snapshot_sink):
        return snapshot_sink
    raise TypeError("snapshot_sink must expose write/store/put or be callable")


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


def _plan_to_dict(plan: RetrievalRunPlan | Mapping[str, Any]) -> JSONDict:
    if isinstance(plan, Mapping):
        return jsonable(plan)
    return plan.to_dict()


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summary(results: list[SourceRetrievalExecutionResult]) -> dict[str, int | SummaryCounts]:
    return {
        "total_steps": len(results),
        "dry_run": _count_status(results, "dry_run"),
        "manual_offline": _count_status(results, "manual_offline"),
        "blocked_missing_client": _count_status(results, "blocked_missing_client"),
        "executed": _count_status(results, "executed"),
        "with_payload": sum(1 for result in results if result.payload.get("available")),
        "persisted": sum(1 for result in results if result.storage.get("status") == "written"),
        "by_status": _count_by_string(result.status for result in results),
        "by_source_system": _count_by_string(
            result.source_system or "unknown" for result in results
        ),
    }


def _count_status(
    results: list[SourceRetrievalExecutionResult],
    status: str,
) -> int:
    return sum(1 for result in results if result.status == status)


def _count_by_string(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
