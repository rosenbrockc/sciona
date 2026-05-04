"""Injected-client fetch boundary for symbolic physics retrieval.

This module plans and executes catalog/RPC fetches, but it never constructs a
real database or network client. Callers inject a duck-typed runtime client and
the fetched rows are handed to the side-effect-free rankers in ``retrieval``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import inspect
import json
import math
from typing import Any, Literal

from sciona.physics_ingest.ids import stable_payload_sha256
from sciona.physics_ingest.retrieval import (
    SymbolicArtifactCandidate,
    SymbolicRetrievalQuery,
    build_symbolic_retrieval_report,
    build_symbolic_synthesis_retrieval_report,
    candidates_from_rows,
)


CATALOG_SYMBOLIC_ARTIFACTS_TABLE = "catalog_symbolic_artifacts"
ARTIFACT_DOCUMENT_RPC = "get_artifact_document"
SYMBOLIC_RETRIEVAL_FETCH_REPORT_KIND = "symbolic_retrieval_fetch"
SYMBOLIC_RETRIEVAL_PLANNER_REQUEST_KIND = "symbolic_retrieval_planner_request"
SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND = "symbolic_retrieval_planner_response"

ReportMode = Literal["none", "retrieval", "synthesis"]

_CATALOG_SELECT = (
    "artifact_id, version_id, expression_id, fqdn, artifact_kind, expression_kind, "
    "raw_formula, topology_hash, dimensional_hash, dim_signatures, mechanism_tags, "
    "behavioral_archetypes, validity_bounds, relationships, symbolic_variables, "
    "source_system, source_kind, source_domains, known_analogues, "
    "data_artifact_dependencies, review_status, validation_status, publish_status, "
    "candidate_status, trust_readiness, is_publishable"
)

_PLANNER_COMPILER_BLOCKER_KINDS = (
    "blocked_status",
    "not_published_or_reviewed",
    "missing_dimensional_metadata",
    "missing_required_validity_bounds",
    "missing_reviewed_validity_bounds",
    "missing_required_validity_match",
    "missing_required_data_artifact_dependencies",
    "raw_excluded_by_policy",
)


@dataclass(frozen=True)
class SymbolicRetrievalPlannerFetchOptions:
    """Planner-facing fetch controls for symbolic retrieval."""

    limit: int = 50
    include_artifact_documents: bool = False
    document_fqdns: tuple[str, ...] = ()
    report_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "limit": max(0, int(self.limit)),
                "include_artifact_documents": self.include_artifact_documents,
                "document_fqdns": _unique_sorted(self.document_fqdns),
                "report_limit": self.report_limit,
            }
        )


@dataclass(frozen=True)
class SymbolicRetrievalPlannerExecutionPolicy:
    """Deterministic execution policy advertised to a runtime planner."""

    dry_run: bool = False
    client_required: bool = True
    report_mode: ReportMode = "synthesis"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "dry_run": self.dry_run,
                "client_required": self.client_required,
                "report_mode": self.report_mode,
            }
        )

_DOCUMENT_FQDN_KEYS = (
    "artifact_fqdns",
    "candidate_fqdns",
    "document_fqdns",
    "fqdns",
    "fqdn",
    "artifact_fqdn",
)

_FILTER_FIELDS = (
    ("topology_hash", "topology_hashes", "in"),
    ("dimensional_hash", "dimensional_hashes", "in"),
    ("dim_signatures", "dim_signatures", "overlaps"),
    ("mechanism_tags", "mechanism_tags", "overlaps"),
    ("behavioral_archetypes", "behavioral_archetypes", "overlaps"),
    ("relationship_kinds", "relationship_kinds", "overlaps"),
    ("validity_regimes", "validity_regimes", "overlaps"),
    ("validity_variables", "validity_variables", "overlaps"),
    ("source_system", "source_systems", "in"),
    ("source_kind", "source_kinds", "in"),
    ("source_domains", "source_domains", "overlaps"),
)


def build_symbolic_retrieval_fetch_plan(
    query: SymbolicRetrievalQuery | Mapping[str, Any],
    *,
    limit: int = 50,
    include_artifact_documents: bool = False,
    document_fqdns: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic JSON-safe catalog/RPC fetch plan."""

    request = _query(query)
    query_payload = _query_payload(request)
    explicit_fqdns = _unique_sorted(
        [*document_fqdns, *_mapping_document_fqdns(query)]
    )
    catalog_request = _request_row(
        operation="table_select",
        source_table=CATALOG_SYMBOLIC_ARTIFACTS_TABLE,
        params={
            "select": _CATALOG_SELECT,
            "limit": max(0, int(limit)),
            "filters": _query_filters(query_payload),
            "query": query_payload,
        },
    )
    request_rows: list[dict[str, Any]] = [catalog_request]
    if include_artifact_documents:
        if explicit_fqdns:
            request_rows.extend(_document_request_row(fqdn) for fqdn in explicit_fqdns)
        else:
            request_rows.append(
                _request_row(
                    operation="rpc_deferred",
                    source_rpc=ARTIFACT_DOCUMENT_RPC,
                    params={
                        "request_fqdn_source": (
                            f"{CATALOG_SYMBOLIC_ARTIFACTS_TABLE}.fqdn"
                        )
                    },
                )
            )
    plan_hash = stable_payload_sha256(
        {
            "version": 1,
            "request_rows": request_rows,
            "include_artifact_documents": include_artifact_documents,
        }
    )
    return _json_safe(
        {
            "plan_kind": "symbolic_retrieval_fetch_plan",
            "plan_hash": plan_hash,
            "replay_key": f"symbolic-retrieval-fetch-plan:{plan_hash}",
            "source_tables": [CATALOG_SYMBOLIC_ARTIFACTS_TABLE],
            "source_rpcs": (
                [ARTIFACT_DOCUMENT_RPC] if include_artifact_documents else []
            ),
            "include_artifact_documents": include_artifact_documents,
            "request_rows": request_rows,
            "summary": {
                "planned_request_count": len(request_rows),
                "planned_catalog_request_count": 1,
                "planned_document_request_count": sum(
                    1
                    for row in request_rows
                    if row.get("source_rpc") == ARTIFACT_DOCUMENT_RPC
                ),
                "has_deferred_document_request": any(
                    row.get("operation") == "rpc_deferred" for row in request_rows
                ),
            },
        }
    )


def build_symbolic_retrieval_planner_request(
    query: SymbolicRetrievalQuery | Mapping[str, Any],
    *,
    limit: int = 50,
    include_artifact_documents: bool = False,
    document_fqdns: Sequence[str] = (),
    dry_run: bool = False,
    client_required: bool | None = None,
    report_mode: ReportMode = "synthesis",
    report_limit: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-safe request envelope for runtime planners."""

    request = _query(query)
    query_payload = _query_payload(request)
    fetch_options = SymbolicRetrievalPlannerFetchOptions(
        limit=limit,
        include_artifact_documents=include_artifact_documents,
        document_fqdns=tuple(document_fqdns),
        report_limit=report_limit,
    ).to_dict()
    policy = SymbolicRetrievalPlannerExecutionPolicy(
        dry_run=dry_run,
        client_required=(not dry_run if client_required is None else client_required),
        report_mode=_report_mode(report_mode),
    ).to_dict()
    fetch_plan = build_symbolic_retrieval_fetch_plan(
        request,
        limit=int(fetch_options["limit"]),
        include_artifact_documents=include_artifact_documents,
        document_fqdns=fetch_options["document_fqdns"],
    )
    raw_trust_policy = str(query_payload.get("raw_trust_policy") or "prefer_reviewed")
    stable = _json_safe(
        {
            "request_kind": SYMBOLIC_RETRIEVAL_PLANNER_REQUEST_KIND,
            "request_version": 1,
            "query": query_payload,
            "fetch_options": fetch_options,
            "fetch_plan": fetch_plan,
            "compiler_contract_expectations": _planner_compiler_contract_expectations(
                query_payload
            ),
            "allowed_candidate_trust_statuses": _planner_allowed_trust_statuses(
                raw_trust_policy
            ),
            "trust_policy": {
                "raw_trust_policy": raw_trust_policy,
                "raw_candidate_execution": "external_knowledge_only",
            },
            "execution_policy": policy,
        }
    )
    request_hash = stable_payload_sha256(stable)
    return _json_safe(
        {
            **stable,
            "request_hash": request_hash,
            "replay_key": f"symbolic-retrieval-planner:{request_hash}",
        }
    )


async def fetch_symbolic_retrieval(
    query: SymbolicRetrievalQuery | Mapping[str, Any],
    *,
    client: Any | None = None,
    dry_run: bool = False,
    limit: int = 50,
    include_artifact_documents: bool = False,
    document_fqdns: Sequence[str] = (),
    report_mode: ReportMode = "none",
    report_limit: int | None = None,
) -> dict[str, Any]:
    """Plan and optionally execute symbolic retrieval fetches with an injected client."""

    plan = build_symbolic_retrieval_fetch_plan(
        query,
        limit=limit,
        include_artifact_documents=include_artifact_documents,
        document_fqdns=document_fqdns,
    )
    diagnostics: list[dict[str, str]] = []
    if dry_run:
        return _result(
            dry_run=True,
            blocked=False,
            plan=plan,
            diagnostics=diagnostics,
        )
    if client is None:
        diagnostics.append(
            {
                "severity": "error",
                "code": "missing_client",
                "message": "non-dry-run symbolic retrieval fetch requires an injected client",
            }
        )
        return _result(
            dry_run=False,
            blocked=True,
            plan=plan,
            diagnostics=diagnostics,
        )

    executed_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    catalog_request = plan["request_rows"][0]
    try:
        catalog_payload = await _execute_request(client, catalog_request)
        catalog_rows = _rows(catalog_payload)
        executed_rows.append(_executed_request_row(catalog_request, len(catalog_rows)))
    except Exception as exc:  # pragma: no cover - exact client exception is runtime-specific
        diagnostics.append(_client_error("catalog_fetch_failed", exc))
        executed_rows.append(_executed_request_row(catalog_request, 0, error=exc))

    if include_artifact_documents:
        doc_requests = _runtime_document_requests(plan, catalog_rows)
        if not doc_requests:
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "missing_document_fqdns",
                    "message": "artifact document fetch was requested but no FQDNs were available",
                }
            )
        for request_row in doc_requests:
            try:
                document_payload = await _execute_request(client, request_row)
                fetched_documents = _documents(document_payload)
                document_rows.extend(fetched_documents)
                executed_rows.append(
                    _executed_request_row(request_row, len(fetched_documents))
                )
            except Exception as exc:  # pragma: no cover - exact client exception is runtime-specific
                diagnostics.append(_client_error("document_fetch_failed", exc))
                executed_rows.append(_executed_request_row(request_row, 0, error=exc))

    normalization_rows = _normalization_rows(catalog_rows, document_rows)
    candidates = candidates_from_rows(normalization_rows)
    payload = _result(
        dry_run=False,
        blocked=False,
        plan=plan,
        diagnostics=diagnostics,
        executed_request_rows=executed_rows,
        catalog_rows=catalog_rows,
        document_rows=document_rows,
        candidates=candidates,
    )
    if report_mode == "retrieval":
        payload["retrieval_report"] = build_symbolic_retrieval_report(
            query,
            candidates,
            limit=report_limit,
        )
    elif report_mode == "synthesis":
        payload["synthesis_report"] = build_symbolic_synthesis_retrieval_report(
            query,
            candidates,
            limit=report_limit,
        )
    return _json_safe(payload)


async def execute_symbolic_retrieval_planner_request(
    planner_request: Mapping[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Execute a planner retrieval envelope through the injected-client fetch boundary."""

    request = _planner_request(planner_request)
    fetch_options = _planner_fetch_options(request)
    execution_policy = _planner_execution_policy(request)
    report_mode = _report_mode(execution_policy.get("report_mode", "synthesis"))
    fetch_result = await fetch_symbolic_retrieval(
        request["query"],
        client=client,
        dry_run=bool(execution_policy.get("dry_run")),
        limit=int(fetch_options.get("limit", 50)),
        include_artifact_documents=bool(
            fetch_options.get("include_artifact_documents")
        ),
        document_fqdns=_strings(fetch_options.get("document_fqdns")),
        report_mode=report_mode,
        report_limit=_optional_int(fetch_options.get("report_limit")),
    )
    candidate_sections = _planner_candidate_sections(fetch_result)
    diagnostics = list(fetch_result.get("diagnostics", ()))
    request_replay_metadata = _planner_replay_metadata(request, fetch_result)
    return _json_safe(
        {
            "report_kind": SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND,
            "request_replay_metadata": request_replay_metadata,
            "execution_policy": execution_policy,
            "blocked": bool(fetch_result.get("blocked")),
            "dry_run": bool(fetch_result.get("dry_run")),
            "executable_candidates": candidate_sections["executable_candidates"],
            "external_knowledge_suggestions": candidate_sections[
                "external_knowledge_suggestions"
            ],
            "blocked_candidates": candidate_sections["blocked_candidates"],
            "diagnostics": diagnostics,
            "fetch_summary": dict(fetch_result.get("summary", {})),
            "fetch_plan": dict(fetch_result.get("fetch_plan", {})),
        }
    )


def _planner_compiler_contract_expectations(
    query_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return _json_safe(
        {
            "required_response_sections": [
                "executable_candidates",
                "external_knowledge_suggestions",
                "blocked_candidates",
                "diagnostics",
            ],
            "executable_candidate_requires": [
                "eligible",
                "published_or_reviewed",
                "dimensional_metadata",
                "no_compiler_blockers",
            ],
            "candidate_contract_fields": [
                "candidate_key",
                "raw_formula",
                "trust_status",
                "score",
                "dimensions",
                "compiler_contract",
            ],
            "required_dimensional_checks": _planner_dimensional_checks(query_payload),
            "blocker_kinds": list(_PLANNER_COMPILER_BLOCKER_KINDS),
        }
    )


def _planner_dimensional_checks(
    query_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for query_key, check_kind in (
        ("topology_hashes", "topology_hash_match"),
        ("dimensional_hashes", "dimensional_hash_match"),
        ("dim_signatures", "dim_signature_overlap"),
    ):
        values = _strings(query_payload.get(query_key))
        if values:
            checks.append({"check_kind": check_kind, "values": values})
    checks.append(
        {
            "check_kind": "candidate_dimensional_metadata_present",
            "required": True,
        }
    )
    return checks


def _planner_allowed_trust_statuses(raw_trust_policy: str) -> dict[str, Any]:
    external_statuses = ["needs_human", "unreviewed"]
    return {
        "executable_candidates": ["automated_pass", "human_reviewed"],
        "external_knowledge_suggestions": external_statuses,
        "blocked_candidates": ["blocked"],
        "raw_trust_policy": raw_trust_policy,
        "raw_candidate_execution_allowed": False,
    }


def _planner_request(planner_request: Mapping[str, Any]) -> dict[str, Any]:
    request = _json_safe(planner_request)
    if not isinstance(request, Mapping):
        raise TypeError("planner_request must be a mapping")
    if request.get("request_kind") != SYMBOLIC_RETRIEVAL_PLANNER_REQUEST_KIND:
        raise ValueError("planner_request must be a symbolic retrieval planner request")
    stable = {
        key: value
        for key, value in request.items()
        if key not in {"request_hash", "replay_key"}
    }
    expected_hash = stable_payload_sha256(stable)
    if not request.get("request_hash"):
        request = {
            **dict(request),
            "request_hash": expected_hash,
            "replay_key": f"symbolic-retrieval-planner:{expected_hash}",
        }
    return dict(request)


def _planner_fetch_options(request: Mapping[str, Any]) -> dict[str, Any]:
    options = _mapping(request.get("fetch_options"))
    return {
        "limit": max(0, int(options.get("limit", 50))),
        "include_artifact_documents": bool(options.get("include_artifact_documents")),
        "document_fqdns": _strings(options.get("document_fqdns")),
        "report_limit": _optional_int(options.get("report_limit")),
    }


def _planner_execution_policy(request: Mapping[str, Any]) -> dict[str, Any]:
    policy = _mapping(request.get("execution_policy"))
    return _json_safe(
        {
            "dry_run": bool(policy.get("dry_run")),
            "client_required": bool(policy.get("client_required")),
            "report_mode": _report_mode(policy.get("report_mode", "synthesis")),
        }
    )


def _planner_candidate_sections(fetch_result: Mapping[str, Any]) -> dict[str, list[Any]]:
    synthesis_report = _mapping(fetch_result.get("synthesis_report"))
    executable = _mapping_list(synthesis_report.get("executable_candidates"))
    external = _mapping_list(synthesis_report.get("external_knowledge_suggestions"))
    blocked = _mapping_list(synthesis_report.get("blocked_candidates"))
    if not synthesis_report:
        retrieval_report = _mapping(fetch_result.get("retrieval_report"))
        external = _mapping_list(
            retrieval_report.get("raw_candidate_external_knowledge_suggestions")
        )
    if fetch_result.get("blocked") and not blocked:
        blocked.append(_planner_blocked_fetch_candidate(fetch_result))
    return {
        "executable_candidates": executable,
        "external_knowledge_suggestions": external,
        "blocked_candidates": blocked,
    }


def _planner_blocked_fetch_candidate(fetch_result: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _mapping_list(fetch_result.get("diagnostics"))
    codes = [
        str(diagnostic.get("code"))
        for diagnostic in diagnostics
        if diagnostic.get("code")
    ]
    return {
        "candidate_key": "<planner_fetch>",
        "eligible": False,
        "trust_status": "blocked",
        "score": 0.0,
        "score_reasons": codes or ["planner_fetch_blocked"],
        "compiler_contract": {
            "blockers": codes or ["planner_fetch_blocked"],
            "can_compile": False,
            "requires_human_review": False,
        },
    }


def _planner_replay_metadata(
    request: Mapping[str, Any],
    fetch_result: Mapping[str, Any],
) -> dict[str, Any]:
    requested_plan = _mapping(request.get("fetch_plan"))
    summary = _mapping(fetch_result.get("summary"))
    return {
        "request_hash": str(request.get("request_hash", "") or ""),
        "request_replay_key": str(request.get("replay_key", "") or ""),
        "fetch_plan_hash": str(requested_plan.get("plan_hash", "") or ""),
        "fetch_plan_replay_key": str(requested_plan.get("replay_key", "") or ""),
        "executed_fetch_plan_hash": str(summary.get("plan_hash", "") or ""),
        "executed_fetch_plan_replay_key": str(summary.get("replay_key", "") or ""),
    }


def _query(query: SymbolicRetrievalQuery | Mapping[str, Any]) -> SymbolicRetrievalQuery:
    if isinstance(query, SymbolicRetrievalQuery):
        return query
    return SymbolicRetrievalQuery.from_mapping(query)


def _query_payload(query: SymbolicRetrievalQuery) -> dict[str, Any]:
    return _json_safe(asdict(query))


def _query_filters(query_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    for field, query_key, op in _FILTER_FIELDS:
        values = _strings(query_payload.get(query_key))
        if values:
            filters.append({"field": field, "op": op, "values": values})
    if query_payload.get("require_validity_bounds"):
        filters.append({"field": "validity_bounds", "op": "present"})
    if query_payload.get("require_data_artifact_dependencies"):
        filters.append({"field": "data_artifact_dependencies", "op": "present"})
    return filters


def _request_row(
    *,
    operation: str,
    params: Mapping[str, Any],
    source_table: str = "",
    source_rpc: str = "",
) -> dict[str, Any]:
    stable = {
        "operation": operation,
        "source_table": source_table,
        "source_rpc": source_rpc,
        "params": _json_safe(params),
    }
    request_hash = stable_payload_sha256(stable)
    return {
        **stable,
        "request_hash": request_hash,
        "replay_key": f"symbolic-retrieval-fetch:{request_hash}",
    }


def _document_request_row(fqdn: str) -> dict[str, Any]:
    return _request_row(
        operation="rpc",
        source_rpc=ARTIFACT_DOCUMENT_RPC,
        params={"request_fqdn": fqdn},
    )


async def _execute_request(client: Any, request_row: Mapping[str, Any]) -> Any:
    operation = str(request_row.get("operation", ""))
    if operation in {"rpc", "rpc_deferred"}:
        return await _execute_rpc_request(client, request_row)
    return await _execute_table_request(client, request_row)


async def _execute_table_request(client: Any, request_row: Mapping[str, Any]) -> Any:
    delegated = await _execute_delegated_client_call(client, request_row)
    if delegated is not _NOT_HANDLED:
        return delegated
    if not hasattr(client, "table"):
        raise TypeError("client does not expose fetch/call or table")
    table = client.table(request_row["source_table"])
    query = table.select(str(_params(request_row).get("select", "*")))
    limit = _params(request_row).get("limit")
    if isinstance(limit, int) and hasattr(query, "limit"):
        query = query.limit(limit)
    return await _execute_query(query)


async def _execute_rpc_request(client: Any, request_row: Mapping[str, Any]) -> Any:
    delegated = await _execute_delegated_client_call(client, request_row)
    if delegated is not _NOT_HANDLED:
        return delegated
    if not hasattr(client, "rpc"):
        raise TypeError("client does not expose fetch/call or rpc")
    return await _execute_query(
        client.rpc(str(request_row["source_rpc"]), dict(_params(request_row)))
    )


class _NotHandled:
    pass


_NOT_HANDLED = _NotHandled()


async def _execute_delegated_client_call(
    client: Any,
    request_row: Mapping[str, Any],
) -> Any:
    for method_name in ("fetch", "call"):
        method = getattr(client, method_name, None)
        if method is None:
            continue
        try:
            return await _maybe_await(method(dict(request_row)))
        except TypeError:
            source = request_row.get("source_table") or request_row.get("source_rpc")
            return await _maybe_await(method(source, dict(_params(request_row))))
    return _NOT_HANDLED


async def _execute_query(query: Any) -> Any:
    execute = getattr(query, "execute", None)
    if execute is None:
        return await _maybe_await(query)
    return await _maybe_await(execute())


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _runtime_document_requests(
    plan: Mapping[str, Any],
    catalog_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    planned_fqdns = [
        str(row.get("params", {}).get("request_fqdn", "") or "")
        for row in plan.get("request_rows", [])
        if row.get("source_rpc") == ARTIFACT_DOCUMENT_RPC
        and row.get("operation") == "rpc"
    ]
    if planned_fqdns:
        return [_document_request_row(fqdn) for fqdn in _unique_sorted(planned_fqdns)]
    catalog_fqdns = [
        str(row.get("fqdn") or row.get("artifact_fqdn") or "")
        for row in catalog_rows
    ]
    return [_document_request_row(fqdn) for fqdn in _unique_sorted(catalog_fqdns)]


def _result(
    *,
    dry_run: bool,
    blocked: bool,
    plan: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    executed_request_rows: Sequence[Mapping[str, Any]] = (),
    catalog_rows: Sequence[Mapping[str, Any]] = (),
    document_rows: Sequence[Mapping[str, Any]] = (),
    candidates: Sequence[SymbolicArtifactCandidate] = (),
) -> dict[str, Any]:
    catalog_count = len(catalog_rows)
    document_count = len(document_rows)
    candidate_keys = [_candidate_key(candidate) for candidate in candidates]
    return _json_safe(
        {
            "report_kind": SYMBOLIC_RETRIEVAL_FETCH_REPORT_KIND,
            "dry_run": dry_run,
            "blocked": blocked,
            "diagnostics": list(diagnostics),
            "fetch_plan": dict(plan),
            "planned_request_rows": list(plan.get("request_rows", ())),
            "executed_request_rows": list(executed_request_rows),
            "fetched_rows": {
                "catalog": list(catalog_rows),
                "documents": list(document_rows),
            },
            "summary": {
                "plan_hash": plan.get("plan_hash", ""),
                "replay_key": plan.get("replay_key", ""),
                "planned_request_count": len(plan.get("request_rows", ())),
                "executed_request_count": len(executed_request_rows),
                "catalog_row_count": catalog_count,
                "document_row_count": document_count,
                "candidate_count": len(candidates),
                "candidate_keys": candidate_keys,
                "source_tables": list(plan.get("source_tables", ())),
                "source_rpcs": list(plan.get("source_rpcs", ())),
            },
        }
    )


def _executed_request_row(
    request_row: Mapping[str, Any],
    row_count: int,
    *,
    error: Exception | None = None,
) -> dict[str, Any]:
    executed = dict(request_row)
    executed["row_count"] = row_count
    executed["status"] = "error" if error else "ok"
    if error:
        executed["error"] = f"{type(error).__name__}: {error}"
    return executed


def _client_error(code: str, exc: Exception) -> dict[str, str]:
    return {
        "severity": "error",
        "code": code,
        "message": f"{type(exc).__name__}: {exc}",
    }


def _params(request_row: Mapping[str, Any]) -> Mapping[str, Any]:
    params = request_row.get("params", {})
    if isinstance(params, Mapping):
        return params
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _report_mode(value: Any) -> ReportMode:
    if value in {"none", "retrieval", "synthesis"}:
        return value
    raise ValueError("report_mode must be none, retrieval, or synthesis")


def _rows(payload: Any) -> list[dict[str, Any]]:
    value = _unwrap_data(payload)
    if value is None:
        return []
    if isinstance(value, Mapping):
        if isinstance(value.get("data"), list):
            return [dict(row) for row in value["data"] if isinstance(row, Mapping)]
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _documents(payload: Any) -> list[dict[str, Any]]:
    return _rows(payload)


def _unwrap_data(payload: Any) -> Any:
    if hasattr(payload, "data"):
        return payload.data
    if isinstance(payload, Mapping) and set(payload.keys()) == {"data"}:
        return payload.get("data")
    return payload


def _normalization_rows(
    catalog_rows: Sequence[Mapping[str, Any]],
    document_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    document_fqdns = {
        str(_artifact(document).get("fqdn", "") or "")
        for document in document_rows
        if _artifact(document).get("fqdn")
    }
    rows = [dict(document) for document in document_rows]
    rows.extend(
        dict(row)
        for row in catalog_rows
        if str(row.get("fqdn") or row.get("artifact_fqdn") or "") not in document_fqdns
    )
    return rows


def _artifact(document: Mapping[str, Any]) -> Mapping[str, Any]:
    artifact = document.get("artifact", {})
    if isinstance(artifact, Mapping):
        return artifact
    return {}


def _candidate_key(candidate: SymbolicArtifactCandidate) -> str:
    return (
        candidate.expression_id
        or candidate.artifact_id
        or candidate.fqdn
        or "<missing>"
    )


def _mapping_document_fqdns(query: SymbolicRetrievalQuery | Mapping[str, Any]) -> list[str]:
    if not isinstance(query, Mapping):
        return []
    values: list[str] = []
    for key in _DOCUMENT_FQDN_KEYS:
        values.extend(_strings(query.get(key)))
    return values


def _unique_sorted(values: Sequence[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        for key in ("fqdn", "artifact_fqdn", "id", "label", "value"):
            text = str(value.get(key, "") or "").strip()
            if text:
                return [text]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[str] = []
        for item in value:
            values.extend(_strings(item))
        return values
    text = str(value).strip()
    return [text] if text else []


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
