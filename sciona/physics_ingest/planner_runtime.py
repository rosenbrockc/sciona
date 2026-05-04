"""Batch runtime boundary for symbolic retrieval planner service calls.

This module owns orchestration around the production runtime planner service
boundary.  It never constructs a network client; callers inject one when they
want non-dry-run execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any

from sciona.physics_ingest.ids import stable_payload_sha256
from sciona.physics_ingest.retrieval import SymbolicRetrievalQuery
from sciona.physics_ingest.retrieval_io import (
    SYMBOLIC_RETRIEVAL_PLANNER_REQUEST_KIND,
    build_symbolic_retrieval_planner_request,
    build_symbolic_retrieval_planner_service_invocation,
    execute_symbolic_retrieval_planner_service_invocation,
)


PLANNER_RUNTIME_SERVICE_BATCH_REPORT_KIND = (
    "symbolic_retrieval_planner_runtime_service_batch"
)
DEFAULT_PLANNER_SERVICE_NAME = "symbolic_retrieval_planner_service"

JSONDict = dict[str, Any]
PlannerRuntimeInput = SymbolicRetrievalQuery | Mapping[str, Any]


@dataclass(frozen=True)
class SymbolicRetrievalPlannerRuntimeBatchReport:
    """JSON-safe batch report for planner runtime service invocations."""

    report: Mapping[str, Any]

    @property
    def blocked(self) -> bool:
        return bool(self.report["blocked"])

    def to_dict(self) -> JSONDict:
        return _json_safe(self.report)


async def execute_symbolic_retrieval_planner_runtime_service_batch(
    inputs: PlannerRuntimeInput | Sequence[PlannerRuntimeInput] | None = None,
    *,
    queries: PlannerRuntimeInput | Sequence[PlannerRuntimeInput] | None = None,
    planner_requests: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    client: Any | None = None,
    planner_client: Any | None = None,
    service_name: str = DEFAULT_PLANNER_SERVICE_NAME,
    dry_run: bool = False,
    preflight: bool = False,
    limit: int = 50,
    include_artifact_documents: bool = False,
    document_fqdns: Sequence[str] = (),
    report_mode: str = "synthesis",
    report_limit: int | None = None,
) -> JSONDict:
    """Execute a batch of planner service invocations with an injected client.

    ``inputs`` and ``queries`` accept symbolic retrieval queries.  A mapping with
    ``request_kind == SYMBOLIC_RETRIEVAL_PLANNER_REQUEST_KIND`` is treated as a
    prebuilt planner request.  ``planner_requests`` is passed through as prebuilt
    requests, including invalid envelopes so retrieval_io can report diagnostics.
    """

    runtime_client = planner_client if planner_client is not None else client
    items = [
        *_runtime_inputs(inputs, source="inputs"),
        *_runtime_inputs(queries, source="queries"),
        *_prebuilt_requests(planner_requests),
    ]
    invocation_reports: list[JSONDict] = []
    for index, item in enumerate(items):
        invocation_reports.append(
            await _execute_runtime_item(
                item,
                index=index,
                client=runtime_client,
                service_name=service_name,
                dry_run=dry_run,
                preflight=preflight,
                limit=limit,
                include_artifact_documents=include_artifact_documents,
                document_fqdns=document_fqdns,
                report_mode=report_mode,
                report_limit=report_limit,
            )
        )

    report = _batch_report(
        invocation_reports,
        service_name=service_name,
        dry_run=dry_run,
        preflight=preflight,
    )
    return SymbolicRetrievalPlannerRuntimeBatchReport(report).to_dict()


async def execute_symbolic_retrieval_planner_runtime_service(
    inputs: PlannerRuntimeInput | Sequence[PlannerRuntimeInput] | None = None,
    **kwargs: Any,
) -> JSONDict:
    """Alias for ``execute_symbolic_retrieval_planner_runtime_service_batch``."""

    return await execute_symbolic_retrieval_planner_runtime_service_batch(
        inputs,
        **kwargs,
    )


async def _execute_runtime_item(
    item: Mapping[str, Any],
    *,
    index: int,
    client: Any | None,
    service_name: str,
    dry_run: bool,
    preflight: bool,
    limit: int,
    include_artifact_documents: bool,
    document_fqdns: Sequence[str],
    report_mode: str,
    report_limit: int | None,
) -> JSONDict:
    input_kind = str(item["input_kind"])
    raw_payload = item["payload"]
    planner_request = (
        _json_safe(raw_payload)
        if input_kind == "planner_request"
        else build_symbolic_retrieval_planner_request(
            raw_payload,
            limit=limit,
            include_artifact_documents=include_artifact_documents,
            document_fqdns=document_fqdns,
            dry_run=dry_run,
            report_mode=report_mode,  # type: ignore[arg-type]
            report_limit=report_limit,
        )
    )
    planned_invocation = _planned_invocation(
        planner_request,
        dry_run=dry_run,
        preflight=preflight,
        service_name=service_name,
    )
    response = await execute_symbolic_retrieval_planner_service_invocation(
        planner_request,
        client=client,
        dry_run=dry_run,
        preflight=preflight,
        service_name=service_name,
    )
    replay_hashes = _invocation_replay_hashes(
        planner_request=planner_request,
        planned_invocation=planned_invocation,
        response=response,
    )
    return _json_safe(
        {
            "index": index,
            "source": item["source"],
            "input_kind": input_kind,
            "request_valid": bool(planned_invocation),
            "planner_request": planner_request,
            "planned_service_invocation": planned_invocation,
            "response": response,
            "summary": _response_summary(response),
            "replay_hashes": replay_hashes,
        }
    )


def _batch_report(
    invocation_reports: Sequence[Mapping[str, Any]],
    *,
    service_name: str,
    dry_run: bool,
    preflight: bool,
) -> JSONDict:
    summaries = [_mapping(report.get("summary")) for report in invocation_reports]
    replay_hashes = _batch_replay_hashes(invocation_reports)
    diagnostics = [
        diagnostic
        for report in invocation_reports
        for diagnostic in _mapping_list(_mapping(report.get("response")).get("diagnostics"))
    ]
    summary = _batch_summary(summaries)
    stable = _json_safe(
        {
            "report_kind": PLANNER_RUNTIME_SERVICE_BATCH_REPORT_KIND,
            "report_version": 1,
            "service_name": service_name,
            "dry_run": bool(dry_run),
            "preflight": bool(preflight),
            "blocked": bool(summary["blocked"]),
            "summary": summary,
            "summaries": list(summaries),
            "blocker_counts": {
                "blocked_response_count": summary["blocked_response_count"],
                "blocked_candidate_count": summary["blocked_candidate_count"],
                "compiler_blocker_count": summary["compiler_blocker_count"],
                "blocking_diagnostic_count": summary["blocking_diagnostic_count"],
            },
            "replay_hashes": replay_hashes,
            "diagnostics": diagnostics,
            "invocation_responses": list(invocation_reports),
        }
    )
    report_hash = stable_payload_sha256(stable)
    return _json_safe(
        {
            **stable,
            "report_hash": report_hash,
            "replay_key": f"symbolic-retrieval-planner-runtime-batch:{report_hash}",
        }
    )


def _batch_summary(summaries: Sequence[Mapping[str, Any]]) -> JSONDict:
    blocked_response_count = sum(1 for summary in summaries if summary["blocked"])
    return {
        "input_count": len(summaries),
        "invocation_response_count": len(summaries),
        "blocked": blocked_response_count > 0,
        "blocked_response_count": blocked_response_count,
        "executable_candidate_count": sum(
            int(summary["executable_candidate_count"]) for summary in summaries
        ),
        "external_knowledge_candidate_count": sum(
            int(summary["external_knowledge_candidate_count"])
            for summary in summaries
        ),
        "blocked_candidate_count": sum(
            int(summary["blocked_candidate_count"]) for summary in summaries
        ),
        "compiler_blocker_count": sum(
            int(summary["compiler_blocker_count"]) for summary in summaries
        ),
        "diagnostic_count": sum(
            int(summary["diagnostic_count"]) for summary in summaries
        ),
        "blocking_diagnostic_count": sum(
            int(summary["blocking_diagnostic_count"]) for summary in summaries
        ),
    }


def _response_summary(response: Mapping[str, Any]) -> JSONDict:
    executable = _mapping_list(response.get("executable_candidates"))
    external = _mapping_list(response.get("external_knowledge_suggestions"))
    blocked = _mapping_list(response.get("blocked_candidates"))
    diagnostics = _mapping_list(response.get("diagnostics"))
    compiler_blockers = [
        blocker
        for candidate in blocked
        for blocker in _compiler_blockers(candidate)
    ]
    return {
        "blocked": bool(response.get("blocked")),
        "dry_run": bool(response.get("dry_run")),
        "preflight": bool(response.get("preflight")),
        "executable_candidate_count": len(executable),
        "external_knowledge_candidate_count": len(external),
        "blocked_candidate_count": len(blocked),
        "compiler_blocker_count": len(compiler_blockers),
        "diagnostic_count": len(diagnostics),
        "blocking_diagnostic_count": sum(
            1
            for diagnostic in diagnostics
            if str(diagnostic.get("severity", "")).lower() == "error"
        ),
    }


def _invocation_replay_hashes(
    *,
    planner_request: Mapping[str, Any],
    planned_invocation: Mapping[str, Any],
    response: Mapping[str, Any],
) -> JSONDict:
    service_invocation = _mapping(response.get("service_invocation"))
    request_replay_metadata = _mapping(response.get("request_replay_metadata"))
    fetch_plan = _mapping(planner_request.get("fetch_plan"))
    service_invocation_hash = str(
        planned_invocation.get("invocation_hash")
        or service_invocation.get("invocation_hash")
        or ""
    )
    service_invocation_replay_key = str(
        planned_invocation.get("replay_key")
        or service_invocation.get("invocation_replay_key")
        or ""
    )
    return {
        "planner_request_hash": str(planner_request.get("request_hash", "") or ""),
        "planner_request_replay_key": str(planner_request.get("replay_key", "") or ""),
        "fetch_plan_hash": str(fetch_plan.get("plan_hash", "") or ""),
        "fetch_plan_replay_key": str(fetch_plan.get("replay_key", "") or ""),
        "service_invocation_hash": service_invocation_hash,
        "service_invocation_replay_key": service_invocation_replay_key,
        "response_hash": stable_payload_sha256(_json_safe(response)),
        "executed_fetch_plan_hash": str(
            request_replay_metadata.get("executed_fetch_plan_hash", "") or ""
        ),
    }


def _batch_replay_hashes(invocation_reports: Sequence[Mapping[str, Any]]) -> JSONDict:
    replay_hashes = [_mapping(report.get("replay_hashes")) for report in invocation_reports]
    response_hashes = [
        str(replay_hash.get("response_hash", "") or "")
        for replay_hash in replay_hashes
    ]
    service_hashes = [
        str(replay_hash.get("service_invocation_hash", "") or "")
        for replay_hash in replay_hashes
        if replay_hash.get("service_invocation_hash")
    ]
    planner_request_hashes = [
        str(replay_hash.get("planner_request_hash", "") or "")
        for replay_hash in replay_hashes
        if replay_hash.get("planner_request_hash")
    ]
    fetch_plan_hashes = [
        str(replay_hash.get("fetch_plan_hash", "") or "")
        for replay_hash in replay_hashes
        if replay_hash.get("fetch_plan_hash")
    ]
    stable = {
        "planner_request_hashes": planner_request_hashes,
        "fetch_plan_hashes": fetch_plan_hashes,
        "service_invocation_hashes": service_hashes,
        "response_hashes": response_hashes,
    }
    return {
        **stable,
        "batch_replay_hash": stable_payload_sha256(_json_safe(stable)),
    }


def _planned_invocation(
    planner_request: Mapping[str, Any],
    *,
    dry_run: bool,
    preflight: bool,
    service_name: str,
) -> JSONDict:
    try:
        return build_symbolic_retrieval_planner_service_invocation(
            planner_request,
            dry_run=dry_run,
            preflight=preflight,
            service_name=service_name,
        )
    except (TypeError, ValueError):
        return {}


def _runtime_inputs(
    value: PlannerRuntimeInput | Sequence[PlannerRuntimeInput] | None,
    *,
    source: str,
) -> list[JSONDict]:
    if value is None:
        return []
    return [
        {
            "source": source,
            "input_kind": _input_kind(payload),
            "payload": payload,
        }
        for payload in _as_input_sequence(value)
    ]


def _prebuilt_requests(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[JSONDict]:
    if value is None:
        return []
    return [
        {
            "source": "planner_requests",
            "input_kind": "planner_request",
            "payload": payload,
        }
        for payload in _as_mapping_sequence(value)
    ]


def _input_kind(value: PlannerRuntimeInput) -> str:
    if isinstance(value, Mapping) and (
        value.get("request_kind") == SYMBOLIC_RETRIEVAL_PLANNER_REQUEST_KIND
    ):
        return "planner_request"
    return "query"


def _as_input_sequence(
    value: PlannerRuntimeInput | Sequence[PlannerRuntimeInput],
) -> list[PlannerRuntimeInput]:
    if isinstance(value, Mapping) or isinstance(value, SymbolicRetrievalQuery):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _as_mapping_sequence(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    return list(value)


def _compiler_blockers(candidate: Mapping[str, Any]) -> list[str]:
    contract = _mapping(candidate.get("compiler_contract"))
    return [
        str(blocker)
        for blocker in _sequence(contract.get("blockers"))
        if str(blocker)
    ]


def _mapping(value: Any) -> JSONDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[JSONDict]:
    return [
        dict(item)
        for item in _sequence(value)
        if isinstance(item, Mapping)
    ]


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(payload)
            for key, payload in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value
