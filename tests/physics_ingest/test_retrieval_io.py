from __future__ import annotations

import json
from types import SimpleNamespace

from sciona.physics_ingest.retrieval import (
    SymbolicRetrievalQuery,
    SymbolicValidityBound,
)
from sciona.physics_ingest.retrieval_io import (
    ARTIFACT_DOCUMENT_RPC,
    CATALOG_SYMBOLIC_ARTIFACTS_TABLE,
    SYMBOLIC_RETRIEVAL_PLANNER_REQUEST_KIND,
    SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND,
    SYMBOLIC_RETRIEVAL_PLANNER_SERVICE_INVOCATION_KIND,
    SYMBOLIC_RETRIEVAL_PLANNER_SERVICE_RESPONSE_KIND,
    build_symbolic_retrieval_planner_service_invocation,
    build_symbolic_retrieval_planner_request,
    build_symbolic_retrieval_fetch_plan,
    execute_symbolic_retrieval_planner_service_invocation,
    execute_symbolic_retrieval_planner_request,
    fetch_symbolic_retrieval,
)


class _FakeTableQuery:
    def __init__(self, client: "_FakeClient", table_name: str) -> None:
        self._client = client
        self._table_name = table_name
        self._select = ""
        self._limit: int | None = None

    def select(self, fields: str) -> "_FakeTableQuery":
        self._select = fields
        return self

    def limit(self, value: int) -> "_FakeTableQuery":
        self._limit = value
        return self

    async def execute(self) -> SimpleNamespace:
        self._client.table_calls.append(
            {
                "table_name": self._table_name,
                "select": self._select,
                "limit": self._limit,
            }
        )
        return SimpleNamespace(data=list(self._client.catalog_rows))


class _FakeRpcQuery:
    def __init__(self, client: "_FakeClient", rpc_name: str, params: dict[str, object]) -> None:
        self._client = client
        self._rpc_name = rpc_name
        self._params = dict(params)

    async def execute(self) -> SimpleNamespace:
        self._client.rpc_calls.append(
            {"rpc_name": self._rpc_name, "params": dict(self._params)}
        )
        return SimpleNamespace(
            data=self._client.documents.get(str(self._params.get("request_fqdn", "")))
        )


class _FakeClient:
    def __init__(
        self,
        *,
        catalog_rows: list[dict[str, object]] | None = None,
        documents: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self.catalog_rows = list(catalog_rows or [])
        self.documents = dict(documents or {})
        self.table_calls: list[dict[str, object]] = []
        self.rpc_calls: list[dict[str, object]] = []

    def table(self, table_name: str) -> _FakeTableQuery:
        return _FakeTableQuery(self, table_name)

    def rpc(self, rpc_name: str, params: dict[str, object]) -> _FakeRpcQuery:
        return _FakeRpcQuery(self, rpc_name, params)


class _FakeAsyncPlannerClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def plan(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(request)
        return self.response


class _FakeSyncInvokePlannerClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def invoke(self, request: dict[str, object]) -> dict[str, object]:
        self.calls.append(request)
        return self.response


def test_symbolic_retrieval_fetch_plan_dry_run_is_deterministic() -> None:
    query = {
        "topology_hash": "topo-wave",
        "dimensional_hashes": ["dim-wave"],
        "mechanism_tags": ["dispersion"],
    }

    plan = build_symbolic_retrieval_fetch_plan(
        query,
        include_artifact_documents=True,
        limit=25,
    )
    plan_again = build_symbolic_retrieval_fetch_plan(
        query,
        include_artifact_documents=True,
        limit=25,
    )

    assert plan == plan_again
    assert plan["source_tables"] == [CATALOG_SYMBOLIC_ARTIFACTS_TABLE]
    assert plan["source_rpcs"] == [ARTIFACT_DOCUMENT_RPC]
    assert plan["summary"]["has_deferred_document_request"] is True
    assert [row["operation"] for row in plan["request_rows"]] == [
        "table_select",
        "rpc_deferred",
    ]
    assert len(plan["plan_hash"]) == 64
    assert json.loads(json.dumps(plan, sort_keys=True)) == plan


async def test_symbolic_retrieval_fetch_dry_run_makes_no_client_calls() -> None:
    client = _FakeClient(catalog_rows=[{"artifact_id": "would-not-fetch"}])

    result = await fetch_symbolic_retrieval(
        SymbolicRetrievalQuery(topology_hashes=("topo",)),
        client=client,
        dry_run=True,
        include_artifact_documents=True,
    )

    assert result["dry_run"] is True
    assert result["blocked"] is False
    assert result["summary"]["executed_request_count"] == 0
    assert result["summary"]["candidate_count"] == 0
    assert client.table_calls == []
    assert client.rpc_calls == []


async def test_symbolic_retrieval_fetch_missing_client_is_blocked() -> None:
    result = await fetch_symbolic_retrieval(
        {"topology_hash": "topo"},
        dry_run=False,
    )

    assert result["blocked"] is True
    assert result["summary"]["executed_request_count"] == 0
    assert result["diagnostics"] == [
        {
            "severity": "error",
            "code": "missing_client",
            "message": "non-dry-run symbolic retrieval fetch requires an injected client",
        }
    ]


async def test_symbolic_retrieval_fetch_uses_fake_client_and_document_rows() -> None:
    client = _FakeClient(
        catalog_rows=[
            {
                "artifact_id": "catalog-wave",
                "expression_id": "expr-catalog-wave",
                "fqdn": "physics.wave.reviewed",
                "topology_hash": "topo-wave",
            }
        ],
        documents={
            "physics.wave.reviewed": {
                "artifact": {
                    "artifact_id": "artifact-wave",
                    "fqdn": "physics.wave.reviewed",
                    "artifact_kind": "symbolic_equation",
                    "source_system": "theoria",
                    "source_kind": "curated_publication",
                    "review_status": "human_reviewed",
                    "validation_status": "passed",
                    "publish_status": "published",
                    "is_publishable": True,
                },
                "symbolic_expressions": [
                    {
                        "expression_id": "expr-doc-wave",
                        "raw_formula": "v = f lambda",
                        "topology_hash": "topo-wave",
                        "dimensional_hash": "dim-wave",
                        "dim_signatures": ["L1", "T-1"],
                        "mechanism_tags": ["dispersion"],
                    }
                ],
            }
        },
    )

    result = await fetch_symbolic_retrieval(
        {
            "topology_hash": "topo-wave",
            "document_fqdns": ["physics.wave.reviewed"],
        },
        client=client,
        include_artifact_documents=True,
        report_mode="retrieval",
    )

    assert client.table_calls[0]["table_name"] == CATALOG_SYMBOLIC_ARTIFACTS_TABLE
    assert client.rpc_calls == [
        {
            "rpc_name": ARTIFACT_DOCUMENT_RPC,
            "params": {"request_fqdn": "physics.wave.reviewed"},
        }
    ]
    assert result["summary"]["catalog_row_count"] == 1
    assert result["summary"]["document_row_count"] == 1
    assert result["summary"]["candidate_count"] == 1
    assert result["summary"]["candidate_keys"] == ["expr-doc-wave"]
    assert result["executed_request_rows"][0]["row_count"] == 1
    assert result["executed_request_rows"][1]["source_rpc"] == ARTIFACT_DOCUMENT_RPC
    assert result["retrieval_report"]["result_count"] == 1


async def test_symbolic_retrieval_fetch_result_is_json_safe() -> None:
    result = await fetch_symbolic_retrieval(
        SymbolicRetrievalQuery(
            topology_hashes=("topo",),
            validity_bounds=(
                SymbolicValidityBound(
                    variable_name="Re",
                    lower_value=0.0,
                    upper_value=1.0,
                ),
            ),
            require_validity_matches=True,
        ),
        client=_FakeClient(
            catalog_rows=[
                {
                    "artifact_id": "catalog",
                    "fqdn": "physics.safe",
                    "topology_hash": "topo",
                    "dim_signatures": ["L1"],
                    "review_status": "unreviewed",
                }
            ]
        ),
    )

    assert json.loads(json.dumps(result, sort_keys=True)) == result


async def test_symbolic_retrieval_fetch_builds_synthesis_report_from_rows() -> None:
    result = await fetch_symbolic_retrieval(
        {
            "topology_hash": "topo-wave",
            "dimensional_hash": "dim-wave",
            "mechanism_tags": ["dispersion"],
            "require_reviewed_bounds": True,
        },
        client=_FakeClient(
            catalog_rows=[
                {
                    "artifact_id": "reviewed-wave",
                    "expression_id": "expr-reviewed-wave",
                    "fqdn": "physics.wave.reviewed",
                    "raw_formula": "v = f lambda",
                    "topology_hash": "topo-wave",
                    "dimensional_hash": "dim-wave",
                    "dim_signatures": ["L1", "T-1"],
                    "mechanism_tags": ["dispersion"],
                    "review_status": "human_reviewed",
                    "validation_status": "passed",
                    "publish_status": "published",
                    "is_publishable": True,
                    "validity_bounds": [
                        {
                            "variable_name": "f",
                            "lower_value": 0,
                            "review_status": "human_reviewed",
                        }
                    ],
                }
            ]
        ),
        report_mode="synthesis",
    )

    assert result["synthesis_report"]["report_kind"] == "symbolic_synthesis_retrieval"
    assert result["synthesis_report"]["executable_candidate_count"] == 1
    assert result["synthesis_report"]["executable_candidates"][0][
        "candidate_key"
    ] == "expr-reviewed-wave"


def test_symbolic_retrieval_planner_request_envelope_is_deterministic() -> None:
    query = {
        "topology_hash": "topo-wave",
        "dimensional_hash": "dim-wave",
        "mechanism_tags": ["dispersion"],
        "raw_trust_policy": "reviewed_only",
    }

    request = build_symbolic_retrieval_planner_request(
        query,
        include_artifact_documents=True,
        limit=10,
        dry_run=True,
        report_mode="synthesis",
    )
    request_again = build_symbolic_retrieval_planner_request(
        query,
        include_artifact_documents=True,
        limit=10,
        dry_run=True,
        report_mode="synthesis",
    )

    assert request == request_again
    assert request["request_kind"] == SYMBOLIC_RETRIEVAL_PLANNER_REQUEST_KIND
    assert len(request["request_hash"]) == 64
    assert request["replay_key"] == (
        f"symbolic-retrieval-planner:{request['request_hash']}"
    )
    assert request["fetch_plan"]["plan_kind"] == "symbolic_retrieval_fetch_plan"
    assert request["execution_policy"] == {
        "client_required": False,
        "dry_run": True,
        "report_mode": "synthesis",
    }
    assert request["trust_policy"]["raw_trust_policy"] == "reviewed_only"
    assert request["allowed_candidate_trust_statuses"]["executable_candidates"] == [
        "automated_pass",
        "human_reviewed",
    ]
    assert request["compiler_contract_expectations"]["required_response_sections"] == [
        "executable_candidates",
        "external_knowledge_suggestions",
        "blocked_candidates",
        "diagnostics",
    ]
    assert json.loads(json.dumps(request, sort_keys=True)) == request


async def test_symbolic_retrieval_planner_dry_run_without_client_is_not_blocked() -> None:
    client = _FakeClient(catalog_rows=[{"artifact_id": "would-not-fetch"}])
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=True,
        include_artifact_documents=True,
    )

    response = await execute_symbolic_retrieval_planner_request(request, client=client)

    assert response["report_kind"] == SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND
    assert response["dry_run"] is True
    assert response["blocked"] is False
    assert response["executable_candidates"] == []
    assert response["external_knowledge_suggestions"] == []
    assert response["blocked_candidates"] == []
    assert client.table_calls == []
    assert client.rpc_calls == []


async def test_symbolic_retrieval_planner_missing_client_blocks_response() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )

    response = await execute_symbolic_retrieval_planner_request(request)

    assert response["blocked"] is True
    assert response["diagnostics"][0]["code"] == "missing_client"
    assert response["blocked_candidates"][0]["candidate_key"] == "<planner_fetch>"
    assert response["blocked_candidates"][0]["compiler_contract"]["blockers"] == [
        "missing_client"
    ]
    assert (
        response["request_replay_metadata"]["request_hash"] == request["request_hash"]
    )
    assert response["request_replay_metadata"]["fetch_plan_hash"] == (
        request["fetch_plan"]["plan_hash"]
    )


async def test_symbolic_retrieval_planner_executes_fake_client_synthesis_sections() -> None:
    client = _FakeClient(
        catalog_rows=[
            {
                "artifact_id": "reviewed-wave",
                "expression_id": "expr-reviewed-wave",
                "fqdn": "physics.wave.reviewed",
                "raw_formula": "v = f lambda",
                "topology_hash": "topo-wave",
                "dimensional_hash": "dim-wave",
                "mechanism_tags": ["dispersion"],
                "review_status": "human_reviewed",
                "validation_status": "passed",
                "publish_status": "published",
                "is_publishable": True,
            },
            {
                "artifact_id": "raw-wave",
                "expression_id": "expr-raw-wave",
                "fqdn": "physics.wave.raw",
                "raw_formula": "v approx f lambda",
                "topology_hash": "topo-wave",
                "dimensional_hash": "dim-wave",
                "mechanism_tags": ["dispersion"],
                "review_status": "unreviewed",
                "candidate_status": "parsed",
            },
            {
                "artifact_id": "blocked-wave",
                "expression_id": "expr-blocked-wave",
                "fqdn": "physics.wave.blocked",
                "raw_formula": "bad wave",
                "topology_hash": "topo-wave",
                "dimensional_hash": "dim-wave",
                "review_status": "human_reviewed",
                "validation_status": "failed",
            },
        ]
    )
    request = build_symbolic_retrieval_planner_request(
        {
            "topology_hash": "topo-wave",
            "dimensional_hash": "dim-wave",
            "mechanism_tags": ["dispersion"],
            "raw_trust_policy": "prefer_reviewed",
        },
        dry_run=False,
        limit=5,
    )

    response = await execute_symbolic_retrieval_planner_request(request, client=client)

    assert response["blocked"] is False
    assert response["fetch_summary"]["candidate_count"] == 3
    assert [row["candidate_key"] for row in response["executable_candidates"]] == [
        "expr-reviewed-wave"
    ]
    assert [
        row["candidate_key"] for row in response["external_knowledge_suggestions"]
    ] == ["expr-raw-wave"]
    assert response["external_knowledge_suggestions"][0]["suggestion"][
        "reason"
    ] == "raw_candidate_needs_external_knowledge"
    assert response["external_knowledge_suggestions"][0]["suggestion"][
        "raw_formula"
    ] == "v approx f lambda"
    assert [row["candidate_key"] for row in response["blocked_candidates"]] == [
        "expr-blocked-wave"
    ]
    assert response["request_replay_metadata"]["executed_fetch_plan_hash"] == request[
        "fetch_plan"
    ]["plan_hash"]
    assert response["dashboard_summary"]["synthesis_candidate_counts"] == {
        "executable": 1,
        "external": 1,
        "blocked": 1,
    }
    assert response["query_coverage_summary"] == {
        "requested": {
            "topology_hash": True,
            "dimensional_hash": True,
            "dim_signature": False,
            "mechanism": True,
            "behavioral_archetype": False,
            "relationship": False,
            "validity": False,
            "source": False,
            "known_analogue": False,
            "data_artifact": False,
        },
        "candidate_match_counts": {
            "topology_hash": 3,
            "dimensional_hash": 3,
            "dim_signature": 0,
            "mechanism": 2,
            "behavioral_archetype": 0,
            "relationship": 0,
            "validity": 0,
            "source": 0,
            "known_analogue": 0,
            "data_artifact": 0,
        },
        "requested_feature_count": 3,
        "matched_requested_feature_count": 3,
        "unmatched_requested_features": [],
    }


async def test_symbolic_retrieval_planner_response_is_json_safe() -> None:
    request = build_symbolic_retrieval_planner_request(
        SymbolicRetrievalQuery(
            topology_hashes=("topo",),
            validity_bounds=(
                SymbolicValidityBound(
                    variable_name="Re",
                    lower_value=float("nan"),
                    upper_value=float("inf"),
                ),
            ),
        ),
        dry_run=True,
        report_limit=1,
    )

    response = await execute_symbolic_retrieval_planner_request(request)

    assert request["query"]["validity_bounds"][0]["lower_value"] == "nan"
    assert request["query"]["validity_bounds"][0]["upper_value"] == "inf"
    assert json.loads(json.dumps(request, sort_keys=True)) == request
    assert json.loads(json.dumps(response, sort_keys=True)) == response


def test_symbolic_retrieval_planner_service_invocation_is_deterministic() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )

    invocation = build_symbolic_retrieval_planner_service_invocation(
        request,
        preflight=True,
    )
    invocation_again = build_symbolic_retrieval_planner_service_invocation(
        request,
        preflight=True,
    )

    assert invocation == invocation_again
    assert (
        invocation["invocation_kind"]
        == SYMBOLIC_RETRIEVAL_PLANNER_SERVICE_INVOCATION_KIND
    )
    assert len(invocation["invocation_hash"]) == 64
    assert invocation["planner_request_hash"] == request["request_hash"]
    assert invocation["planner_request_replay_key"] == request["replay_key"]
    assert invocation["execution_policy"] == {
        "dry_run": False,
        "preflight": True,
    }
    assert json.loads(json.dumps(invocation, sort_keys=True)) == invocation


async def test_symbolic_retrieval_planner_service_dry_run_does_not_call_client() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )
    client = _FakeAsyncPlannerClient(
        {"report_kind": SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND}
    )

    response = await execute_symbolic_retrieval_planner_service_invocation(
        request,
        client=client,
        dry_run=True,
    )

    assert response["report_kind"] == SYMBOLIC_RETRIEVAL_PLANNER_SERVICE_RESPONSE_KIND
    assert response["dry_run"] is True
    assert response["preflight"] is False
    assert response["blocked"] is False
    assert response["diagnostics"] == []
    assert response["planner_response"] == {}
    assert client.calls == []


async def test_symbolic_retrieval_planner_service_preflight_does_not_call_client() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )
    client = _FakeSyncInvokePlannerClient(
        {"report_kind": SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND}
    )

    response = await execute_symbolic_retrieval_planner_service_invocation(
        request,
        client=client,
        preflight=True,
    )

    assert response["dry_run"] is False
    assert response["preflight"] is True
    assert response["blocked"] is False
    assert client.calls == []


async def test_symbolic_retrieval_planner_service_async_plan_normalizes_response() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )
    planner_response = {
        "report_kind": SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND,
        "blocked": False,
        "dry_run": False,
        "executable_candidates": [{"candidate_key": "expr-reviewed-wave"}],
        "external_knowledge_suggestions": [{"candidate_key": "expr-raw-wave"}],
        "diagnostics": [{"severity": "info", "code": "ok", "message": "ok"}],
    }
    client = _FakeAsyncPlannerClient(planner_response)

    response = await execute_symbolic_retrieval_planner_service_invocation(
        request,
        client=client,
    )

    assert len(client.calls) == 1
    assert (
        client.calls[0]["invocation_kind"]
        == SYMBOLIC_RETRIEVAL_PLANNER_SERVICE_INVOCATION_KIND
    )
    assert client.calls[0]["planner_request"] == request
    assert response["blocked"] is False
    assert response["executable_candidates"] == [
        {"candidate_key": "expr-reviewed-wave"}
    ]
    assert response["external_knowledge_suggestions"] == [
        {"candidate_key": "expr-raw-wave"}
    ]
    assert response["blocked_candidates"] == []
    assert response["diagnostics"][0]["code"] == "ok"
    assert response["request_replay_metadata"]["request_hash"] == request[
        "request_hash"
    ]
    assert response["request_replay_metadata"][
        "service_invocation_hash"
    ] == response["service_invocation"]["invocation_hash"]
    assert json.loads(json.dumps(response, sort_keys=True)) == response


async def test_symbolic_retrieval_planner_service_sync_invoke_normalizes_response() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )
    client = _FakeSyncInvokePlannerClient(
        {
            "blocked": False,
            "executable_candidates": [{"candidate_key": "expr-sync"}],
        }
    )

    response = await execute_symbolic_retrieval_planner_service_invocation(
        request,
        client=client,
    )

    assert client.calls[0]["planner_request_hash"] == request["request_hash"]
    assert response["planner_response"]["report_kind"] == (
        SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND
    )
    assert response["executable_candidates"] == [{"candidate_key": "expr-sync"}]


async def test_symbolic_retrieval_planner_service_callable_client() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )
    calls: list[dict[str, object]] = []

    def planner(request_payload: dict[str, object]) -> dict[str, object]:
        calls.append(request_payload)
        return {
            "blocked": False,
            "external_knowledge_suggestions": [{"candidate_key": "expr-callable"}],
        }

    response = await execute_symbolic_retrieval_planner_service_invocation(
        request,
        client=planner,
    )

    assert calls[0]["planner_request_replay_key"] == request["replay_key"]
    assert response["external_knowledge_suggestions"] == [
        {"candidate_key": "expr-callable"}
    ]


async def test_symbolic_retrieval_planner_service_missing_client_blocks() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )

    response = await execute_symbolic_retrieval_planner_service_invocation(request)

    assert response["blocked"] is True
    assert response["diagnostics"] == [
        {
            "severity": "error",
            "code": "missing_planner_client",
            "message": (
                "non-dry-run planner service invocation requires an injected "
                "planner client"
            ),
        }
    ]
    assert response["blocked_candidates"][0]["candidate_key"] == "<planner_service>"
    assert response["blocked_candidates"][0]["compiler_contract"]["blockers"] == [
        "missing_planner_client"
    ]


async def test_symbolic_retrieval_planner_service_invalid_request_is_reported() -> None:
    response = await execute_symbolic_retrieval_planner_service_invocation(
        {"request_kind": "wrong"},
        client=_FakeAsyncPlannerClient({}),
    )

    assert response["blocked"] is True
    assert response["diagnostics"] == [
        {
            "severity": "error",
            "code": "invalid_planner_request",
            "message": "planner_request must be a symbolic retrieval planner request",
        }
    ]
    assert response["service_invocation"]["planner_request_hash"] == ""
    assert response["blocked_candidates"][0]["compiler_contract"]["blockers"] == [
        "invalid_planner_request"
    ]
    assert json.loads(json.dumps(response, sort_keys=True)) == response
