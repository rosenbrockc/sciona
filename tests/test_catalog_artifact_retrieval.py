from __future__ import annotations

from types import SimpleNamespace

import pytest

from sciona.services.artifact_retrieval import MacroArtifactRetriever
from sciona.services.catalog_artifact_retrieval import (
    CatalogMacroArtifactRetriever,
)
from sciona.services.models import (
    MacroArtifactCandidate,
    MacroMatchRequest,
)


class _FakeRpcQuery:
    def __init__(self, client: "_FakeSupabase", fn_name: str, params: dict[str, object]) -> None:
        self._client = client
        self._fn_name = fn_name
        self._params = dict(params)

    async def execute(self):
        return SimpleNamespace(data=self._client.rpc_payload(self._fn_name, self._params))


class _FakeTableQuery:
    def __init__(self, client: "_FakeSupabase", table_name: str) -> None:
        self._client = client
        self._table_name = table_name
        self._filters: list[tuple[str, object]] = []
        self._or_clause = ""
        self._limit: int | None = None

    def select(self, _fields: str):
        return self

    def or_(self, clause: str):
        self._or_clause = clause
        return self

    def eq(self, field: str, value: object):
        self._filters.append((field, value))
        return self

    def limit(self, value: int):
        self._limit = value
        return self

    def maybe_single(self):
        return self

    async def execute(self):
        return SimpleNamespace(
            data=self._client.table_payload(
                self._table_name,
                self._filters,
                self._or_clause,
                self._limit,
            )
        )


class _FakeSupabase:
    def __init__(
        self,
        *,
        search_rows: list[dict[str, object]] | None = None,
        documents: dict[str, dict[str, object]] | None = None,
        versions: dict[str, dict[str, object]] | None = None,
        catalog_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self._search_rows = list(search_rows or [])
        self._documents = dict(documents or {})
        self._versions = dict(versions or {})
        self._catalog_rows = list(catalog_rows or [])

    def rpc(self, fn_name: str, params: dict[str, object]):
        return _FakeRpcQuery(self, fn_name, params)

    def rpc_payload(self, fn_name: str, params: dict[str, object]):
        if fn_name == "search_artifacts_hybrid":
            return list(self._search_rows)
        if fn_name == "get_artifact_document":
            return self._documents.get(str(params.get("request_fqdn", "")))
        raise AssertionError(f"unexpected rpc {fn_name}")

    def table(self, table_name: str):
        return _FakeTableQuery(self, table_name)

    def table_payload(
        self,
        table_name: str,
        filters: list[tuple[str, object]],
        or_clause: str,
        limit: int | None,
    ) -> list[dict[str, object]] | dict[str, object] | None:
        if table_name == "catalog_artifacts_served":
            rows = list(self._catalog_rows)
            filter_map = {field: value for field, value in filters}
            artifact_kind = str(filter_map.get("artifact_kind", "") or "").strip()
            if artifact_kind:
                rows = [
                    row
                    for row in rows
                    if str(row.get("artifact_kind", "") or "") == artifact_kind
                ]
            if or_clause:
                rows = []
            if limit is not None:
                rows = rows[:limit]
            return rows
        if table_name != "artifact_versions":
            raise AssertionError(f"unexpected table {table_name}")
        filter_map = {field: value for field, value in filters}
        return self._versions.get(str(filter_map.get("artifact_id", "")))


@pytest.mark.asyncio
async def test_catalog_macro_retriever_returns_catalog_cdg_candidate() -> None:
    document = {
        "artifact": {
            "artifact_id": "artifact-1",
            "fqdn": "cdg.skeleton.signal_detect_measure",
            "artifact_kind": "cdg",
            "source_symbol": "signal_detect_measure",
            "verified_leaf_coverage": 0.8,
            "visibility_tier": "general",
            "namespace_root": "sciona.architect.assets.skeletons",
            "namespace_path": "signal_detect_measure",
        },
        "descriptions": [
            {
                "kind": "dejargonized",
                "content": "Condition a signal, detect events, and estimate a rate.",
            }
        ],
        "cdg_nodes": [
            {
                "node_id": "root",
                "parent_node_id": "",
                "name": "Signal Detect Measure",
                "description": "Solve the measurement task.",
                "concept_type": "analysis",
                "status": "decomposed",
                "type_signature": "signal -> rate",
            },
            {
                "node_id": "leaf",
                "parent_node_id": "root",
                "name": "Detect Events",
                "description": "Find events in the signal.",
                "concept_type": "analysis",
                "status": "atomic",
                "type_signature": "signal -> events",
                "matched_primitive": "pkg.signal.detect_events",
            },
        ],
        "cdg_edges": [
            {
                "source_id": "leaf",
                "target_id": "root",
                "output_name": "events",
                "input_name": "events",
                "source_type": "events",
                "target_type": "events",
            }
        ],
    }
    retriever = CatalogMacroArtifactRetriever(
        _FakeSupabase(
            search_rows=[
                {
                    "artifact_id": "artifact-1",
                    "artifact_kind": "cdg",
                    "fqdn": "cdg.skeleton.signal_detect_measure",
                    "technical_description": "Detect heart rate from ECG with a published CDG.",
                    "domain_tags": ["ecg", "rate_estimation"],
                    "score": 0.9,
                }
            ],
            documents={"cdg.skeleton.signal_detect_measure": document},
            versions={
                "artifact-1": {
                    "version_id": "version-1",
                    "semver": "phase2.v1",
                    "content_hash": "hash-123",
                }
            },
        ),
        min_score=0.3,
    )

    result = await retriever.match_goal(
        MacroMatchRequest(goal="Detect heart rate from ECG")
    )

    assert result.success is True
    assert result.candidate is not None
    assert result.candidate.fqdn == "cdg.skeleton.signal_detect_measure"
    assert result.candidate.semver == "phase2.v1"
    assert result.candidate.content_hash == "hash-123"
    assert result.candidate.terminal_on_match is False
    assert result.candidate.cdg is not None
    assert len(result.candidate.cdg.nodes) == 2
    assert len(result.candidate.cdg.edges) == 1
    assert "sciona.architect.assets.skeletons" in result.candidate.domain_tags


@pytest.mark.asyncio
async def test_catalog_macro_retriever_falls_back_when_catalog_misses() -> None:
    fallback = MacroArtifactRetriever(
        [
            MacroArtifactCandidate(
                fqdn="cdg.skeleton.family.divide_and_conquer.v1",
                semver="phase2.v1",
                content_hash="fallback-hash",
                description="Break a problem into smaller subproblems.",
                conceptual_summary="divide and conquer family skeleton",
                verified_leaf_coverage=0.7,
                terminal_on_match=False,
            )
        ],
        min_score=0.3,
    )
    retriever = CatalogMacroArtifactRetriever(
        _FakeSupabase(search_rows=[]),
        fallback=fallback,
        min_score=0.3,
    )

    result = await retriever.match_goal(
        MacroMatchRequest(goal="Divide a large optimization into subproblems")
    )

    assert result.success is True
    assert result.candidate is not None
    assert result.candidate.fqdn == "cdg.skeleton.family.divide_and_conquer.v1"


@pytest.mark.asyncio
async def test_catalog_macro_retriever_ranks_over_catalog_rows_when_rpc_misses() -> None:
    document = {
        "artifact": {
            "artifact_id": "artifact-2",
            "fqdn": "cdg.skeleton.family.divide_and_conquer.v1",
            "artifact_kind": "cdg",
            "source_symbol": "divide_and_conquer",
            "verified_leaf_coverage": 0.9,
            "visibility_tier": "general",
        },
        "descriptions": [
            {
                "kind": "dejargonized",
                "content": "Break a large problem into smaller subproblems.",
            }
        ],
        "cdg_nodes": [
            {
                "node_id": "root",
                "parent_node_id": "",
                "name": "Divide And Conquer",
                "description": "Split and combine.",
                "concept_type": "analysis",
                "status": "decomposed",
                "type_signature": "input -> output",
            }
        ],
        "cdg_edges": [],
    }
    retriever = CatalogMacroArtifactRetriever(
        _FakeSupabase(
            search_rows=[],
            catalog_rows=[
                {
                    "artifact_id": "artifact-2",
                    "artifact_kind": "cdg",
                    "fqdn": "cdg.skeleton.family.divide_and_conquer.v1",
                    "technical_description": "Recursive decomposition family skeleton.",
                    "domain_tags": [],
                    "score": 0.0,
                }
            ],
            documents={"cdg.skeleton.family.divide_and_conquer.v1": document},
            versions={
                "artifact-2": {
                    "version_id": "version-2",
                    "semver": "v1",
                    "content_hash": "hash-456",
                }
            },
        ),
        min_score=0.3,
    )

    result = await retriever.match_goal(
        MacroMatchRequest(goal="Break a large problem into smaller subproblems")
    )

    assert result.success is True
    assert result.candidate is not None
    assert result.candidate.fqdn == "cdg.skeleton.family.divide_and_conquer.v1"
