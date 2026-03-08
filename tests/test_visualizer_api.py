"""Tests for the CDG Visualizer API."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def mock_driver():
    """Create a mock Neo4j async driver."""
    driver = AsyncMock()
    return driver


@pytest.fixture()
def client(mock_driver):
    """Create a TestClient with mocked Neo4j driver (no real lifespan)."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ageom.visualizer_api import app

    # Replace lifespan to avoid real Neo4j connection
    @asynccontextmanager
    async def _test_lifespan(a: FastAPI):
        a.state.driver = mock_driver
        yield

    app.router.lifespan_context = _test_lifespan

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class _FakeSession:
    """Mock async session with async context manager support."""

    def __init__(self, run_side_effects):
        self._effects = list(run_side_effects)
        self._call_idx = 0

    async def run(self, *args, **kwargs):
        result = self._effects[self._call_idx]
        self._call_idx += 1
        return result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _make_result(records):
    """Create an async-iterable mock result from a list of dicts."""
    result = AsyncMock()

    async def _aiter(self):
        for rec in records:
            yield rec

    result.__aiter__ = _aiter
    return result


class TestListCDGs:
    def test_returns_list(self, client, mock_driver):
        records = [
            {
                "repo": "biosppy",
                "node_count": 5,
                "concept_types": ["sorting", "filtering"],
                "statuses": ["atomic", "decomposed"],
            }
        ]
        session = _FakeSession([_make_result(records)])
        mock_driver.session = MagicMock(return_value=session)

        resp = client.get("/api/cdgs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["repo"] == "biosppy"
        assert data[0]["node_count"] == 5

    def test_empty_list(self, client, mock_driver):
        session = _FakeSession([_make_result([])])
        mock_driver.session = MagicMock(return_value=session)

        resp = client.get("/api/cdgs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_filter_by_concept_type(self, client, mock_driver):
        records = [
            {
                "repo": "repo_a",
                "node_count": 3,
                "concept_types": ["sorting"],
                "statuses": ["atomic"],
            },
            {
                "repo": "repo_b",
                "node_count": 7,
                "concept_types": ["graph_traversal"],
                "statuses": ["decomposed"],
            },
        ]
        session = _FakeSession([_make_result(records)])
        mock_driver.session = MagicMock(return_value=session)

        resp = client.get("/api/cdgs?concept_type=sorting")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["repo"] == "repo_a"

    def test_filter_by_status(self, client, mock_driver):
        records = [
            {
                "repo": "repo_a",
                "node_count": 3,
                "concept_types": ["sorting"],
                "statuses": ["atomic"],
            },
        ]
        session = _FakeSession([_make_result(records)])
        mock_driver.session = MagicMock(return_value=session)

        resp = client.get("/api/cdgs?status=decomposed")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_filter_by_search_query(self, client, mock_driver):
        records = [
            {
                "repo": "biosppy",
                "node_count": 5,
                "concept_types": ["sorting"],
                "statuses": ["atomic"],
            }
        ]
        session = _FakeSession([_make_result(records)])
        mock_driver.session = MagicMock(return_value=session)

        resp = client.get("/api/cdgs?q=bio")
        assert resp.status_code == 200
        # The `q` filter is applied at the Cypher level, so we just check it passes through
        data = resp.json()
        assert len(data) == 1


class TestGetCDG:
    def test_returns_cdg(self, client, mock_driver):
        node_records = [
            {
                "a": {
                    "node_id": "n1",
                    "name": "Node 1",
                    "description": "A node",
                    "concept_type": "sorting",
                    "status": "atomic",
                    "depth": 0,
                    "type_signature": "list -> list",
                    "is_optional": False,
                    "is_opaque": False,
                    "is_external": False,
                    "parallelizable": False,
                    "conceptual_summary": "",
                },
                "inputs": [{"name": "data", "type_desc": "list", "constraints": ""}],
                "outputs": [{"name": "sorted", "type_desc": "list", "constraints": ""}],
                "children": [],
                "parent_id": None,
            }
        ]
        edge_records = [
            {
                "source_id": "n1",
                "target_id": "n2",
                "output_name": "sorted",
                "input_name": "data",
                "source_type": "list",
                "target_type": "list",
                "requires_glue": False,
            }
        ]

        session = _FakeSession([
            _make_result(node_records),
            _make_result(edge_records),
        ])
        mock_driver.session = MagicMock(return_value=session)

        resp = client.get("/api/cdgs/test_repo")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["node_id"] == "n1"
        assert len(data["edges"]) == 1
        assert data["metadata"]["repo"] == "test_repo"

    def test_not_found(self, client, mock_driver):
        session = _FakeSession([
            _make_result([]),  # no nodes
        ])
        mock_driver.session = MagicMock(return_value=session)

        resp = client.get("/api/cdgs/nonexistent")
        assert resp.status_code == 404

    def test_edges_empty(self, client, mock_driver):
        node_records = [
            {
                "a": {
                    "node_id": "n1",
                    "name": "Leaf",
                    "description": "",
                    "concept_type": "",
                    "status": "atomic",
                    "depth": 0,
                    "type_signature": "",
                    "is_optional": False,
                    "is_opaque": False,
                    "is_external": False,
                    "parallelizable": False,
                    "conceptual_summary": "",
                },
                "inputs": [],
                "outputs": [],
                "children": [],
                "parent_id": None,
            }
        ]
        session = _FakeSession([
            _make_result(node_records),
            _make_result([]),
        ])
        mock_driver.session = MagicMock(return_value=session)

        resp = client.get("/api/cdgs/leaf_repo")
        assert resp.status_code == 200
        data = resp.json()
        assert data["edges"] == []


class TestStaticFiles:
    def test_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "CDG Visualizer" in resp.text


class TestDashboardAPI:
    def test_list_runs_and_latest(self, client, monkeypatch, tmp_path):
        from ageom.telemetry import reset_telemetry_runtime

        reset_telemetry_runtime()
        now = time.time()
        payload = {
            "run_id": "abc123",
            "pipeline": "algorithm_creation",
            "status": "running",
            "started_at": now - 3.0,
            "last_update_at": now - 1.0,
            "stages": {},
            "prompt_by_key": {},
            "inflight_prompts": {},
            "prompt_dispatches": 2,
            "prompt_successes": 1,
            "prompt_failures": 0,
            "prompt_inflight": 1,
            "events_count": 4,
            "metadata": {"goal": "test"},
        }
        (tmp_path / "run_abc123.json").write_text(json.dumps(payload))
        monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

        resp = client.get("/api/dashboard/runs")
        assert resp.status_code == 200
        data = resp.json()
        run_ids = [r.get("run_id") for r in data["runs"]]
        assert "abc123" in run_ids

        latest = client.get("/api/dashboard/latest")
        assert latest.status_code == 200
        assert latest.json()["run_id"] == "abc123"

    def test_dashboard_run_includes_routing_and_retrieval_summaries(
        self, client, monkeypatch, tmp_path
    ):
        from ageom.telemetry import reset_telemetry_runtime

        reset_telemetry_runtime()
        now = time.time()
        payload = {
            "run_id": "route123",
            "pipeline": "algorithm_creation",
            "status": "running",
            "started_at": now - 2.0,
            "last_update_at": now - 1.0,
            "stages": {},
            "prompt_by_key": {},
            "inflight_prompts": {},
            "prompt_dispatches": 1,
            "prompt_successes": 1,
            "prompt_failures": 0,
            "prompt_inflight": 0,
            "events_count": 2,
            "metadata": {
                "retrieval_policy": {
                    "confidence_band": "high",
                    "skill_index": True,
                    "graph_retrieval": False,
                    "semantic_backend": "lexical",
                    "hunter_mode": "standard",
                },
                "llm_routing": {
                    "architect": {
                        "default_provider": "anthropic",
                        "default_model": "claude-sonnet",
                        "active_overrides": [
                            {
                                "prompt_key": "architect_strategy",
                                "provider": "codex_shim",
                                "model": "gpt-5.3-codex",
                            }
                        ],
                        "suppressed_default_overrides": [],
                        "custom_nonbenchmark_overrides": [],
                    },
                    "hunter": {
                        "default_provider": "llama_cpp",
                        "default_model": "qwen2.5-coder:7b",
                        "active_overrides": [],
                        "suppressed_default_overrides": ["hunter_unused"],
                        "custom_nonbenchmark_overrides": [],
                    },
                },
            },
        }
        (tmp_path / "run_route123.json").write_text(json.dumps(payload))
        monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

        resp = client.get("/api/dashboard/runs/route123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_summary"]["confidence_band"] == "high"
        assert data["retrieval_summary"]["semantic_backend"] == "lexical"
        assert data["routing_summary"]["architect"]["default"] == "anthropic:claude-sonnet"
        assert data["routing_summary"]["architect"]["active_count"] == 1
        assert data["routing_summary"]["hunter"]["suppressed_count"] == 1

    def test_dashboard_run_includes_benchmark_and_release_validation_summaries(
        self, client, monkeypatch, tmp_path
    ):
        from ageom.telemetry import reset_telemetry_runtime

        reset_telemetry_runtime()
        now = time.time()
        payload = {
            "run_id": "bench123",
            "pipeline": "release_validation",
            "status": "completed",
            "started_at": now - 5.0,
            "last_update_at": now - 1.0,
            "ended_at": now - 1.0,
            "stages": {},
            "prompt_by_key": {},
            "inflight_prompts": {},
            "prompt_dispatches": 0,
            "prompt_successes": 0,
            "prompt_failures": 0,
            "prompt_inflight": 0,
            "events_count": 1,
            "metadata": {
                "benchmark_validation": {
                    "summary_report": "build/release_validation/benchmarks/summary.json",
                    "prompt_report": "build/release_validation/benchmarks/prompt_benchmark.json",
                    "flow_report": "build/release_validation/benchmarks/flow_benchmark.json",
                    "prompt_cases": 12,
                    "prompt_results": 24,
                    "prompt_summary": "prompt summary",
                    "flow_cases": 4,
                    "flow_results": 16,
                    "flow_summary": "flow summary",
                },
                "release_validation": {
                    "manifest": "build/release_validation/release_validation.json",
                    "benchmarks_dir": "build/release_validation/benchmarks",
                    "status": "passed",
                },
            },
        }
        (tmp_path / "run_bench123.json").write_text(json.dumps(payload))
        monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

        resp = client.get("/api/dashboard/runs/bench123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["benchmark_summary"]["prompt_cases"] == 12
        assert data["benchmark_summary"]["prompt_results"] == 24
        assert data["benchmark_summary"]["flow_cases"] == 4
        assert data["benchmark_summary"]["flow_results"] == 16
        assert data["benchmark_summary"]["release_status"] == "passed"
        assert data["benchmark_summary"]["manifest"].endswith("release_validation.json")

    def test_run_hang_annotation(self, client, monkeypatch, tmp_path):
        from ageom.telemetry import reset_telemetry_runtime

        reset_telemetry_runtime()
        now = time.time()
        payload = {
            "run_id": "hung-run",
            "pipeline": "algorithm_creation",
            "status": "running",
            "started_at": now - 200.0,
            "last_update_at": now - 50.0,
            "stages": {
                "hunter_round_1": {
                    "name": "hunter_round_1",
                    "status": "running",
                    "started_at": now - 180.0,
                    "last_heartbeat_at": now - 25.0,
                    "ended_at": None,
                    "message": "pending=4",
                    "completed": 1,
                    "total": 4,
                }
            },
            "prompt_by_key": {},
            "inflight_prompts": {},
            "prompt_dispatches": 3,
            "prompt_successes": 1,
            "prompt_failures": 0,
            "prompt_inflight": 2,
            "events_count": 8,
            "metadata": {},
        }
        (tmp_path / "run_hung-run.json").write_text(json.dumps(payload))
        monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("AGEOM_TELEMETRY_STALE_SECONDS", "5")

        resp = client.get("/api/dashboard/runs/hung-run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_hung"] is True
        assert len(data["stale_stages"]) == 1
