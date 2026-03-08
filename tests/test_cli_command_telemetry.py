from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeMetrics:
    def __init__(self, backend: str) -> None:
        self._backend = backend

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "backend": self._backend,
            "searches_total": 1,
            "search_hit_rate": 1.0,
            "search_latency_ms_avg": 2.0,
            "puts_total": 1,
            "duplicate_suppression_rate": 0.0,
            "match_success_delta": 0.0,
            "promotions_total": 0,
            "injected_blocks": 0,
            "injected_chars": 0,
            "template_search_hits": 0,
            "template_searches_total": 0,
            "template_puts_total": 0,
            "template_injected_blocks": 0,
        }


class _AsyncNullContext:
    async def __aenter__(self) -> "_AsyncNullContext":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeEnv:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _latest_persisted_run(root: Path) -> dict[str, object]:
    persisted = sorted(root.glob("run_*.json"))
    assert persisted
    return json.loads(persisted[-1].read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_decompose_writes_persisted_telemetry_run(monkeypatch, tmp_path: Path):
    from ageom.cli import _cmd_decompose
    from ageom.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

    output_path = tmp_path / "cdg.json"
    metrics = _FakeMetrics("memory")
    fake_cdg = SimpleNamespace(
        metadata={"thread_id": "thread-1"},
        nodes=[],
        edges=[],
        is_complete=lambda: True,
    )

    monkeypatch.setattr(
        "ageom.cli._load_architect_catalog",
        lambda args, config: (SimpleNamespace(size=3), {"source_candidates": 3}),
    )
    monkeypatch.setattr(
        "ageom.cli._resolve_retrieval_policy",
        lambda **kwargs: SimpleNamespace(
            catalog_confidence=0.9,
            confidence_band="high",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=None,
            hunter_mode="standard",
        ),
    )
    monkeypatch.setattr("ageom.cli._load_skill_index_or_empty", lambda config, enabled=True: object())
    monkeypatch.setattr("ageom.cli._create_llm_router", lambda *args, **kwargs: object())

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_create_shared_context(*args, **kwargs):
        return None, metrics

    async def _fake_run_decompose(agent, args, max_depth, catalog):
        return fake_cdg

    monkeypatch.setattr("ageom.cli._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr("ageom.cli._create_shared_context", _fake_create_shared_context)
    monkeypatch.setattr("ageom.cli._run_decompose", _fake_run_decompose)
    monkeypatch.setattr(
        "ageom.architect.checkpointer.create_checkpointer",
        lambda uri: _AsyncNullContext(),
    )
    monkeypatch.setattr(
        "ageom.architect.graph.DecompositionAgent",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "ageom.architect.handoff.save_json",
        lambda cdg, path: Path(path).write_text('{"ok": true}', encoding="utf-8"),
    )

    await _cmd_decompose(
        argparse.Namespace(
            goal="Test decomposition goal",
            max_depth=2,
            mode="verified",
            no_persist=True,
            output=str(output_path),
            thread_id="thread-1",
        )
    )

    payload = _latest_persisted_run(tmp_path)
    assert payload["pipeline"] == "decompose"
    assert payload["status"] == "completed"
    assert payload["metadata"]["command"] == "decompose"
    assert payload["metadata"]["goal"] == "Test decomposition goal"
    assert payload["metadata"]["llm_routing"]["architect"]["round"] == "architect"
    assert payload["metadata"]["catalog_alignment"]["source_candidates"] == 3
    assert payload["metadata"]["shared_context"]["contexts"]["architect"]["backend"] == "memory"
    assert Path(payload["metadata"]["shared_context"]["metrics_path"]).exists()
    assert payload["stages"]["setup"]["status"] == "completed"
    assert payload["stages"]["architect_decompose"]["status"] == "completed"


@pytest.mark.asyncio
async def test_match_writes_persisted_telemetry_run(monkeypatch, tmp_path: Path):
    from ageom.cli import _cmd_match
    from ageom.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("AGEOM_TELEMETRY_RUNS_DIR", str(tmp_path))

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    metrics = _FakeMetrics("postgres")
    env = _FakeEnv()

    monkeypatch.setattr(
        "ageom.cli._load_architect_catalog",
        lambda args, config: (SimpleNamespace(size=5), {"source_candidates": 5}),
    )
    monkeypatch.setattr(
        "ageom.cli._resolve_retrieval_policy",
        lambda **kwargs: SimpleNamespace(
            catalog_confidence=0.6,
            confidence_band="medium",
            skill_index_enabled=True,
            graph_retrieval_enabled=False,
            semantic_index_backend_override="lexical",
            hunter_mode="standard",
        ),
    )
    monkeypatch.setattr("ageom.cli._load_semantic_index", lambda *args, **kwargs: (object(), "lexical"))
    monkeypatch.setattr("ageom.cli._create_proof_env", lambda prover, config: env)
    monkeypatch.setattr(
        "ageom.judge.checker.VerificationOracleImpl",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr("ageom.cli._create_llm_router", lambda *args, **kwargs: object())

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_create_shared_context(*args, **kwargs):
        return None, metrics

    class _FakeHunterAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def find_match(self, node):
            declaration = SimpleNamespace(
                name="identity",
                type_signature="forall x, x = x",
            )
            candidate = SimpleNamespace(declaration=declaration)
            verified_match = SimpleNamespace(candidate=candidate)
            return SimpleNamespace(
                success=True,
                verified_match=verified_match,
                all_candidates=[candidate],
                all_verifications=[],
            )

    monkeypatch.setattr("ageom.cli._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr("ageom.cli._create_shared_context", _fake_create_shared_context)
    monkeypatch.setattr("ageom.hunter.graph.HunterAgent", _FakeHunterAgent)

    await _cmd_match(
        argparse.Namespace(
            statement="forall x, x = x",
            pdg_file=None,
            prover="python",
            index_dir=str(index_dir),
            mode="verified",
        )
    )

    payload = _latest_persisted_run(tmp_path)
    assert payload["pipeline"] == "match"
    assert payload["status"] == "completed"
    assert payload["metadata"]["command"] == "match"
    assert payload["metadata"]["statement_count"] == 1
    assert payload["metadata"]["llm_routing"]["hunter"]["round"] == "hunter"
    assert payload["metadata"]["retrieval_policy"]["semantic_backend"] == "lexical"
    assert payload["metadata"]["shared_context"]["contexts"]["hunter"]["backend"] == "postgres"
    assert payload["stages"]["setup"]["status"] == "completed"
    assert payload["stages"]["matching"]["status"] == "completed"
    assert payload["stages"]["matching"]["completed"] == 1
    assert payload["stages"]["matching"]["total"] == 1
    assert env.closed is True
