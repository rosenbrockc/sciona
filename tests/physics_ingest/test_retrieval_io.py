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
    build_symbolic_retrieval_fetch_plan,
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
