from __future__ import annotations

import json

from sciona.physics_ingest.planner_runtime import (
    PLANNER_RUNTIME_SERVICE_BATCH_REPORT_KIND,
    execute_symbolic_retrieval_planner_runtime_service_batch,
)
from sciona.physics_ingest.retrieval import (
    SymbolicRetrievalQuery,
    SymbolicValidityBound,
)
from sciona.physics_ingest.retrieval_io import (
    SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND,
    build_symbolic_retrieval_planner_request,
)


class _FakeAsyncPlannerClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def plan(self, invocation: dict[str, object]) -> dict[str, object]:
        self.calls.append(invocation)
        return self.response


async def test_planner_runtime_dry_run_batch_is_deterministic() -> None:
    client = _FakeAsyncPlannerClient({"would": "not-call"})
    inputs = [
        {"topology_hash": "topo-wave", "mechanism_tags": ["dispersion"]},
        {"dimensional_hash": "dim-wave", "raw_trust_policy": "reviewed_only"},
    ]

    report = await execute_symbolic_retrieval_planner_runtime_service_batch(
        inputs,
        client=client,
        dry_run=True,
        include_artifact_documents=True,
        service_name="runtime_planner",
    )
    report_again = await execute_symbolic_retrieval_planner_runtime_service_batch(
        inputs,
        client=client,
        dry_run=True,
        include_artifact_documents=True,
        service_name="runtime_planner",
    )

    assert report == report_again
    assert report["report_kind"] == PLANNER_RUNTIME_SERVICE_BATCH_REPORT_KIND
    assert report["blocked"] is False
    assert report["summary"]["input_count"] == 2
    assert report["summary"]["invocation_response_count"] == 2
    assert report["summary"]["blocked_response_count"] == 0
    assert report["summary"]["executable_candidate_count"] == 0
    assert len(report["report_hash"]) == 64
    assert len(report["replay_hashes"]["batch_replay_hash"]) == 64
    assert client.calls == []
    assert json.loads(json.dumps(report, sort_keys=True)) == report


async def test_planner_runtime_preflight_batch_makes_no_client_calls() -> None:
    client = _FakeAsyncPlannerClient({"would": "not-call"})

    report = await execute_symbolic_retrieval_planner_runtime_service_batch(
        {"topology_hash": "topo-wave"},
        client=client,
        preflight=True,
    )

    assert report["preflight"] is True
    assert report["blocked"] is False
    assert report["invocation_responses"][0]["response"]["preflight"] is True
    assert client.calls == []


async def test_planner_runtime_handles_prebuilt_planner_request() -> None:
    request = build_symbolic_retrieval_planner_request(
        {"topology_hash": "topo-wave"},
        dry_run=False,
        limit=7,
    )

    report = await execute_symbolic_retrieval_planner_runtime_service_batch(
        planner_requests=request,
        dry_run=True,
    )

    invocation_report = report["invocation_responses"][0]
    assert invocation_report["input_kind"] == "planner_request"
    assert invocation_report["request_valid"] is True
    assert invocation_report["planner_request"] == request
    assert invocation_report["replay_hashes"]["planner_request_hash"] == request[
        "request_hash"
    ]
    assert invocation_report["replay_hashes"]["fetch_plan_hash"] == request[
        "fetch_plan"
    ]["plan_hash"]


async def test_planner_runtime_fake_async_client_non_dry_run() -> None:
    planner_response = {
        "report_kind": SYMBOLIC_RETRIEVAL_PLANNER_RESPONSE_KIND,
        "blocked": False,
        "dry_run": False,
        "executable_candidates": [{"candidate_key": "expr-reviewed-wave"}],
        "external_knowledge_suggestions": [{"candidate_key": "expr-raw-wave"}],
        "blocked_candidates": [
            {
                "candidate_key": "expr-blocked-wave",
                "compiler_contract": {"blockers": ["missing_dimensional_metadata"]},
            }
        ],
        "diagnostics": [{"severity": "info", "code": "ok", "message": "ok"}],
    }
    client = _FakeAsyncPlannerClient(planner_response)

    report = await execute_symbolic_retrieval_planner_runtime_service_batch(
        {"topology_hash": "topo-wave"},
        planner_client=client,
        dry_run=False,
        service_name="runtime_planner",
    )

    assert len(client.calls) == 1
    assert client.calls[0]["service_name"] == "runtime_planner"
    assert report["blocked"] is False
    assert report["summary"]["executable_candidate_count"] == 1
    assert report["summary"]["external_knowledge_candidate_count"] == 1
    assert report["summary"]["blocked_candidate_count"] == 1
    assert report["summary"]["compiler_blocker_count"] == 1
    assert report["summary"]["diagnostic_count"] == 1


async def test_planner_runtime_missing_client_uses_retrieval_io_blocked_response() -> None:
    report = await execute_symbolic_retrieval_planner_runtime_service_batch(
        {"topology_hash": "topo-wave"},
        dry_run=False,
    )

    response = report["invocation_responses"][0]["response"]
    assert report["blocked"] is True
    assert report["summary"]["blocked_response_count"] == 1
    assert report["summary"]["blocked_candidate_count"] == 1
    assert report["summary"]["compiler_blocker_count"] == 1
    assert report["summary"]["blocking_diagnostic_count"] == 1
    assert response["diagnostics"][0]["code"] == "missing_planner_client"
    assert response["blocked_candidates"][0]["candidate_key"] == "<planner_service>"


async def test_planner_runtime_invalid_request_diagnostic_pass_through() -> None:
    client = _FakeAsyncPlannerClient({})

    report = await execute_symbolic_retrieval_planner_runtime_service_batch(
        planner_requests={"request_kind": "wrong"},
        planner_client=client,
    )

    invocation_report = report["invocation_responses"][0]
    response = invocation_report["response"]
    assert client.calls == []
    assert report["blocked"] is True
    assert invocation_report["request_valid"] is False
    assert invocation_report["planned_service_invocation"] == {}
    assert invocation_report["replay_hashes"]["planner_request_hash"] == ""
    assert response["diagnostics"] == [
        {
            "severity": "error",
            "code": "invalid_planner_request",
            "message": "planner_request must be a symbolic retrieval planner request",
        }
    ]
    assert response["blocked_candidates"][0]["compiler_contract"]["blockers"] == [
        "invalid_planner_request"
    ]


async def test_planner_runtime_report_is_json_serializable() -> None:
    report = await execute_symbolic_retrieval_planner_runtime_service_batch(
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

    assert json.loads(json.dumps(report, sort_keys=True)) == report
