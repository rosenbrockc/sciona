from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from sciona.services.artifact_retrieval import MacroArtifactRetriever
from sciona.services.models import MacroArtifactCandidate
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)


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


def _successful_match_result(node: PDGNode) -> MatchResult:
    declaration = Declaration(
        name="algorithms.detect_heart_rate",
        type_signature="np.ndarray -> float",
        conceptual_summary="detect heart rate from ecg",
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=declaration,
        score=0.95,
        retrieval_method="lexical",
    )
    verification = VerificationResult(
        candidate=candidate,
        verified=True,
        compiler_output="ok",
        proof_term="algorithms.detect_heart_rate",
        verification_level=VerificationLevel.TYPE_CHECKED,
    )
    return MatchResult(
        pdg_node=node,
        verified_match=verification,
        all_candidates=[candidate],
        all_verifications=[verification],
    )


def _latest_persisted_run(root: Path) -> dict[str, object]:
    persisted = sorted(root.glob("run_*.json"))
    assert persisted
    return json.loads(persisted[-1].read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_decompose_writes_persisted_telemetry_run(monkeypatch, tmp_path: Path):
    from sciona.cli import _cmd_decompose
    from sciona.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("SCIONA_TELEMETRY_RUNS_DIR", str(tmp_path))

    output_path = tmp_path / "cdg.json"
    metrics = _FakeMetrics("memory")
    fake_cdg = SimpleNamespace(
        metadata={"thread_id": "thread-1"},
        nodes=[],
        edges=[],
        is_complete=lambda: True,
    )

    _D = "sciona.commands.decompose_cmds"
    monkeypatch.setattr(
        f"{_D}._load_architect_catalog",
        lambda args, config: (SimpleNamespace(size=3), {"source_candidates": 3}),
    )
    monkeypatch.setattr(
        f"{_D}._resolve_retrieval_policy",
        lambda **kwargs: SimpleNamespace(
            catalog_confidence=0.9,
            confidence_band="high",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override=None,
            hunter_mode="standard",
        ),
    )
    monkeypatch.setattr(f"{_D}._load_skill_index_or_empty", lambda config, enabled=True: object())
    monkeypatch.setattr(f"{_D}._create_llm_router", lambda *args, **kwargs: object())

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_create_shared_context(*args, **kwargs):
        return None, metrics

    async def _fake_run_decompose(agent, args, max_depth, catalog):
        return fake_cdg

    monkeypatch.setattr(f"{_D}._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr(f"{_D}._create_shared_context", _fake_create_shared_context)
    monkeypatch.setattr(f"{_D}._run_decompose", _fake_run_decompose)
    monkeypatch.setattr(
        "sciona.architect.checkpointer.create_checkpointer",
        lambda uri: _AsyncNullContext(),
    )
    monkeypatch.setattr(
        "sciona.architect.graph.DecompositionAgent",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "sciona.architect.handoff.save_json",
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
    from sciona.cli import _cmd_match
    from sciona.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("SCIONA_TELEMETRY_RUNS_DIR", str(tmp_path))

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    metrics = _FakeMetrics("postgres")
    env = _FakeEnv()

    _M = "sciona.commands.match_cmds"
    monkeypatch.setattr(
        f"{_M}._load_architect_catalog",
        lambda args, config: (SimpleNamespace(size=5), {"source_candidates": 5}),
    )
    monkeypatch.setattr(
        f"{_M}._resolve_retrieval_policy",
        lambda **kwargs: SimpleNamespace(
            catalog_confidence=0.6,
            confidence_band="medium",
            skill_index_enabled=True,
            graph_retrieval_enabled=False,
            semantic_index_backend_override="lexical",
            hunter_mode="standard",
        ),
    )
    monkeypatch.setattr(f"{_M}._load_semantic_index", lambda *args, **kwargs: (object(), "lexical"))
    monkeypatch.setattr(f"{_M}._create_proof_env", lambda prover, config: env)
    monkeypatch.setattr(
        "sciona.judge.checker.VerificationOracleImpl",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(f"{_M}._create_llm_router", lambda *args, **kwargs: object())

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

    monkeypatch.setattr(f"{_M}._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr(f"{_M}._create_shared_context", _fake_create_shared_context)
    monkeypatch.setattr("sciona.hunter.graph.HunterAgent", _FakeHunterAgent)

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


@pytest.mark.asyncio
async def test_run_rapid_mode_uses_direct_match_path(monkeypatch, tmp_path: Path):
    from sciona.cli import _cmd_run
    from sciona.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("SCIONA_TELEMETRY_RUNS_DIR", str(tmp_path))

    output_dir = tmp_path / "run_output"
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    metrics = _FakeMetrics("memory")
    env = _FakeEnv()

    _R = "sciona.commands.run_cmds"
    monkeypatch.setattr(
        f"{_R}._load_architect_catalog",
        lambda args, config: (SimpleNamespace(size=0), {"source_candidates": 0}),
    )
    monkeypatch.setattr(
        f"{_R}._resolve_retrieval_policy",
        lambda **kwargs: SimpleNamespace(
            catalog_confidence=0.0,
            confidence_band="none",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override="lexical",
            hunter_mode="standard",
        ),
    )
    monkeypatch.setattr(f"{_R}._load_semantic_index", lambda *args, **kwargs: (object(), "lexical"))
    monkeypatch.setattr(f"{_R}._create_proof_env", lambda prover, config: env)
    monkeypatch.setattr(
        "sciona.judge.checker.VerificationOracleImpl",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(f"{_R}._create_llm_router", lambda *args, **kwargs: object())

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_create_shared_context(*args, **kwargs):
        return None, metrics

    class _FakeHunterAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def find_match(self, node):
            return _successful_match_result(node)

    def _save_cdg(cdg, path):
        Path(path).write_text(cdg.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(f"{_R}._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr(f"{_R}._create_shared_context", _fake_create_shared_context)
    monkeypatch.setattr("sciona.hunter.graph.HunterAgent", _FakeHunterAgent)
    monkeypatch.setattr("sciona.architect.handoff.save_json", _save_cdg)
    monkeypatch.setattr(
        "sciona.architect.graph.DecompositionAgent",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("architect should be skipped")),
    )
    monkeypatch.setattr(
        "sciona.orchestrator.run_orchestration",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("orchestration should be skipped")
        ),
    )

    await _cmd_run(
        argparse.Namespace(
            goal="Detect heart rate from ECG",
            prover="python",
            output=str(output_dir),
            trace=False,
            max_rounds=2,
            mode="rapid",
        )
    )

    payload = _latest_persisted_run(tmp_path)
    assert payload["pipeline"] == "algorithm_creation"
    assert payload["status"] == "completed"
    assert payload["metadata"]["command"] == "run"
    assert payload["metadata"]["rapid_direct_path"] is True
    assert "architect" not in payload["metadata"]["llm_routing"]
    assert "hunter" in payload["metadata"]["llm_routing"]
    assert "architect_decompose" not in payload["stages"]
    assert payload["stages"]["rapid_direct_match"]["status"] == "completed"
    assert (output_dir / "cdg.json").exists()
    assert (output_dir / "matches.json").exists()
    assert env.closed is True
    assert payload["metadata"]["llm_routing"]["hunter"]["round"] == "hunter"
    assert payload["metadata"]["retrieval_policy"]["semantic_backend"] == "lexical"
    assert payload["metadata"]["shared_context"]["contexts"]["hunter"]["backend"] == "memory"
    assert payload["stages"]["setup"]["status"] == "completed"
    assert env.closed is True


@pytest.mark.asyncio
async def test_run_structured_mode_uses_single_pass_matching(monkeypatch, tmp_path: Path):
    from sciona.architect.handoff import CDGExport
    from sciona.architect.models import AlgorithmicNode, ConceptType, NodeStatus
    from sciona.cli import _cmd_run
    from sciona.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("SCIONA_TELEMETRY_RUNS_DIR", str(tmp_path))

    output_dir = tmp_path / "structured_output"
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    metrics = _FakeMetrics("memory")
    env = _FakeEnv()

    fake_cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Design Filter",
                description="Design stable ECG bandpass coefficients.",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                type_signature="FilterSpec -> Coefficients",
            )
        ],
        edges=[],
        metadata={"goal": "Detect heart rate from ECG"},
    )

    _R = "sciona.commands.run_cmds"
    monkeypatch.setattr(
        f"{_R}._load_architect_catalog",
        lambda args, config: (SimpleNamespace(size=2), {"source_candidates": 2}),
    )
    monkeypatch.setattr(
        f"{_R}._resolve_retrieval_policy",
        lambda **kwargs: SimpleNamespace(
            catalog_confidence=0.5,
            confidence_band="medium",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override="lexical",
            hunter_mode="standard",
        ),
    )
    monkeypatch.setattr(f"{_R}._load_skill_index_or_empty", lambda config, enabled=True: object())
    monkeypatch.setattr(f"{_R}._load_semantic_index", lambda *args, **kwargs: (object(), "lexical"))
    monkeypatch.setattr(f"{_R}._create_proof_env", lambda prover, config: env)
    monkeypatch.setattr(
        "sciona.judge.checker.VerificationOracleImpl",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(f"{_R}._create_llm_router", lambda *args, **kwargs: object())

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_create_shared_context(*args, **kwargs):
        return None, metrics

    class _FakeArchitectAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def decompose(self, goal):
            return fake_cdg

    class _FakeHunterAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def find_match(self, node):
            return _successful_match_result(node)

    def _save_cdg(cdg, path):
        Path(path).write_text(cdg.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(f"{_R}._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr(f"{_R}._create_shared_context", _fake_create_shared_context)
    monkeypatch.setattr("sciona.architect.graph.DecompositionAgent", _FakeArchitectAgent)
    monkeypatch.setattr("sciona.hunter.graph.HunterAgent", _FakeHunterAgent)
    monkeypatch.setattr("sciona.architect.handoff.save_json", _save_cdg)
    monkeypatch.setattr(
        "sciona.architect.checkpointer.create_checkpointer",
        lambda uri: _AsyncNullContext(),
    )
    monkeypatch.setattr(
        "sciona.orchestrator.run_orchestration",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("orchestration should be skipped")
        ),
    )

    await _cmd_run(
        argparse.Namespace(
            goal="Detect heart rate from ECG",
            prover="python",
            output=str(output_dir),
            trace=False,
            max_rounds=2,
            mode="structured",
        )
    )

    payload = _latest_persisted_run(tmp_path)
    assert payload["pipeline"] == "algorithm_creation"
    assert payload["status"] == "completed"
    assert payload["metadata"]["command"] == "run"
    assert payload["metadata"]["execution_path"] == "structured_single_pass"
    assert payload["metadata"]["rapid_direct_path"] is False
    assert "architect" in payload["metadata"]["llm_routing"]
    assert "hunter" in payload["metadata"]["llm_routing"]
    assert payload["stages"]["architect_decompose"]["status"] == "completed"
    assert payload["stages"]["structured_match"]["status"] == "completed"
    assert "orchestration" not in payload["stages"]
    assert (output_dir / "cdg.json").exists()
    assert (output_dir / "matches.json").exists()
    assert env.closed is True


@pytest.mark.asyncio
async def test_run_single_agent_mode_uses_direct_first_planner(monkeypatch, tmp_path: Path):
    from sciona.cli import _cmd_run
    from sciona.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("SCIONA_TELEMETRY_RUNS_DIR", str(tmp_path))

    output_dir = tmp_path / "single_agent_output"
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    metrics = _FakeMetrics("memory")
    env = _FakeEnv()

    _R = "sciona.commands.run_cmds"
    monkeypatch.setattr(
        f"{_R}._load_architect_catalog",
        lambda args, config: (SimpleNamespace(size=1), {"source_candidates": 1}),
    )
    monkeypatch.setattr(
        f"{_R}._resolve_retrieval_policy",
        lambda **kwargs: SimpleNamespace(
            catalog_confidence=0.5,
            confidence_band="medium",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override="lexical",
            hunter_mode="standard",
        ),
    )
    monkeypatch.setattr(f"{_R}._load_skill_index_or_empty", lambda config, enabled=True: object())
    monkeypatch.setattr(f"{_R}._load_semantic_index", lambda *args, **kwargs: (object(), "lexical"))
    monkeypatch.setattr(f"{_R}._create_proof_env", lambda prover, config: env)
    monkeypatch.setattr(
        "sciona.judge.checker.VerificationOracleImpl",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(f"{_R}._create_llm_router", lambda *args, **kwargs: object())

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_create_shared_context(*args, **kwargs):
        return None, metrics

    class _FakeHunterAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def find_match(self, node):
            return _successful_match_result(node)

    def _save_cdg(cdg, path):
        Path(path).write_text(cdg.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(f"{_R}._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr(f"{_R}._create_shared_context", _fake_create_shared_context)
    monkeypatch.setattr("sciona.hunter.graph.HunterAgent", _FakeHunterAgent)
    monkeypatch.setattr("sciona.architect.handoff.save_json", _save_cdg)
    monkeypatch.setattr(
        "sciona.services.skeleton_artifacts.build_local_skeleton_macro_retriever",
        lambda min_score=0.55: MacroArtifactRetriever([], min_score=0.99),
    )
    monkeypatch.setattr(
        "sciona.architect.graph.DecompositionAgent",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("architect should be skipped on direct single-agent success")
        ),
    )
    monkeypatch.setattr(
        "sciona.orchestrator.run_orchestration",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("orchestration should be skipped on direct single-agent success")
        ),
    )

    await _cmd_run(
        argparse.Namespace(
            goal="Detect heart rate from ECG",
            prover="python",
            output=str(output_dir),
            trace=False,
            max_rounds=2,
            mode="single_agent",
        )
    )

    payload = _latest_persisted_run(tmp_path)
    assert payload["pipeline"] == "algorithm_creation"
    assert payload["status"] == "completed"
    assert payload["metadata"]["execution_mode"] == "single_agent"
    assert payload["metadata"]["execution_path"] == "single_agent_direct"
    assert payload["metadata"]["single_agent_mode"] is True
    assert payload["stages"]["single_agent_planner"]["status"] == "completed"
    assert "architect_decompose" not in payload["stages"]
    assert payload["metadata"]["single_agent"]["policy"]["direct_grounding_enabled"] is True
    assert payload["metadata"]["single_agent"]["policy"]["decomposition_mode"] == "single_pass"
    assert payload["metadata"]["single_agent"]["policy"]["retrieval_intensity"] == "light"
    assert payload["metadata"]["single_agent"]["termination_reason"] == "direct_verified"
    assert payload["metadata"]["single_agent"]["verification_status"] == "verified"
    assert payload["metadata"]["single_agent"]["step_budget"] == 6
    assert payload["metadata"]["single_agent"]["steps_used"] == 2
    assert payload["metadata"]["single_agent"]["tool_dispatch_count_total"] == 2
    assert payload["metadata"]["single_agent"]["tool_latency_ms_total"] >= 0.0
    assert payload["metadata"]["single_agent"]["tool_metrics"]["artifact.match_goal"]["dispatches"] == 1
    assert payload["metadata"]["single_agent"]["tool_metrics"]["hunter.match_goal"]["dispatches"] == 1
    assert payload["metadata"]["single_agent"]["escalation_events"] == []
    assert payload["metadata"]["single_agent"]["open_failures"] == []
    assert payload["metadata"]["single_agent"]["artifacts"] == {
        "cdg": "direct_goal_cdg",
        "match_results": "direct_match_result",
    }
    assert payload["metadata"]["single_agent"]["artifact_mutations"] == {
        "cdg": 1,
        "match_results": 1,
    }
    manifest_path = Path(payload["metadata"]["single_agent"]["artifact_manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["execution_path"] == "single_agent_direct"
    assert manifest["escalation_events"] == []
    assert manifest["tool_metrics"]["artifact.match_goal"]["dispatches"] == 1
    assert manifest["tool_metrics"]["hunter.match_goal"]["dispatches"] == 1
    assert manifest["artifacts"]["cdg"]["path"] == str(output_dir / "cdg.json")
    assert manifest["artifacts"]["cdg"]["exists"] is True
    assert manifest["artifacts"]["match_results"]["path"] == str(output_dir / "matches.json")
    assert payload["metadata"]["single_agent"]["concrete_artifacts"]["cdg"]["path"] == str(
        output_dir / "cdg.json"
    )
    assert payload["metadata"]["single_agent"]["attempt_history"] == [
        "direct_macro_match",
        "direct_match",
    ]
    assert payload["metadata"]["single_agent"]["steps"][0]["action"] == "direct_macro_match"
    assert payload["metadata"]["single_agent"]["steps"][0]["status"] == "failed"
    assert payload["metadata"]["single_agent"]["steps"][1]["action"] == "direct_match"
    assert payload["metadata"]["single_agent"]["steps"][1]["status"] == "completed"
    assert (output_dir / "cdg.json").exists()
    assert (output_dir / "matches.json").exists()
    assert (output_dir / "planner_artifacts.json").exists()
    assert env.closed is True


@pytest.mark.asyncio
async def test_run_single_agent_mode_uses_macro_skeleton_before_architect(
    monkeypatch,
    tmp_path: Path,
):
    from sciona.cli import _cmd_run
    from sciona.telemetry import configure_dashboard_output, reset_telemetry_runtime

    reset_telemetry_runtime()
    configure_dashboard_output(tmp_path)
    monkeypatch.setenv("SCIONA_TELEMETRY_RUNS_DIR", str(tmp_path))

    output_dir = tmp_path / "single_agent_macro_output"
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    metrics = _FakeMetrics("memory")
    env = _FakeEnv()
    macro_cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="macro_filter",
                name="Filter Signal For Detection",
                description="Condition the raw signal for robust event detection.",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                type_signature="signal -> conditioned_signal",
            ),
            AlgorithmicNode(
                node_id="macro_rate",
                name="Compute Event Rate",
                description="Estimate a rate from detected events.",
                concept_type=ConceptType.ANALYSIS,
                status=NodeStatus.ATOMIC,
                type_signature="events -> rate",
            ),
        ],
        edges=[],
        metadata={"artifact_kind": "cdg"},
    )

    _R = "sciona.commands.run_cmds"
    monkeypatch.setattr(
        f"{_R}._load_architect_catalog",
        lambda args, config: (SimpleNamespace(size=1), {"source_candidates": 1}),
    )
    monkeypatch.setattr(
        f"{_R}._resolve_retrieval_policy",
        lambda **kwargs: SimpleNamespace(
            catalog_confidence=0.5,
            confidence_band="medium",
            skill_index_enabled=False,
            graph_retrieval_enabled=False,
            semantic_index_backend_override="lexical",
            hunter_mode="standard",
        ),
    )
    monkeypatch.setattr(f"{_R}._load_skill_index_or_empty", lambda config, enabled=True: object())
    monkeypatch.setattr(f"{_R}._load_semantic_index", lambda *args, **kwargs: (object(), "lexical"))
    monkeypatch.setattr(f"{_R}._create_proof_env", lambda prover, config: env)
    monkeypatch.setattr(
        "sciona.judge.checker.VerificationOracleImpl",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(f"{_R}._create_llm_router", lambda *args, **kwargs: object())

    async def _noop_warm(*args, **kwargs):
        return None

    async def _fake_create_shared_context(*args, **kwargs):
        return None, metrics

    class _FakeHunterAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def find_match(self, node):
            return _successful_match_result(node)

    def _save_cdg(cdg, path):
        Path(path).write_text(cdg.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(f"{_R}._warm_llm_if_supported", _noop_warm)
    monkeypatch.setattr(f"{_R}._create_shared_context", _fake_create_shared_context)
    monkeypatch.setattr("sciona.hunter.graph.HunterAgent", _FakeHunterAgent)
    monkeypatch.setattr("sciona.architect.handoff.save_json", _save_cdg)
    monkeypatch.setattr(
        "sciona.architect.graph.DecompositionAgent",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("architect should be skipped when macro skeleton is selected")
        ),
    )
    monkeypatch.setattr(
        "sciona.orchestrator.run_orchestration",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("orchestration should be skipped on macro structured success")
        ),
    )
    monkeypatch.setattr(
        "sciona.services.skeleton_artifacts.build_local_skeleton_macro_retriever",
        lambda min_score=0.55: MacroArtifactRetriever(
            [
                MacroArtifactCandidate(
                    fqdn="cdg.skeleton.signal_detect_measure",
                    semver="v1",
                    content_hash="macro123",
                    description="Condition a raw signal, detect events, and compute rate.",
                    conceptual_summary="signal detect and measure family skeleton",
                    domain_tags=["signal_detect_measure", "event_rate_estimation", "ecg"],
                    cdg=macro_cdg,
                    terminal_on_match=False,
                )
            ],
            min_score=0.3,
        ),
    )

    await _cmd_run(
        argparse.Namespace(
            goal="Detect heart rate from ECG",
            prover="python",
            output=str(output_dir),
            trace=False,
            max_rounds=2,
            mode="single_agent",
        )
    )

    payload = _latest_persisted_run(tmp_path)
    assert payload["status"] == "completed"
    assert payload["metadata"]["execution_path"] == "single_agent_macro_structured"
    assert payload["metadata"]["single_agent"]["termination_reason"] == "macro_structured_verified"
    assert payload["metadata"]["single_agent"]["verification_status"] == "verified"
    assert payload["metadata"]["single_agent"]["artifacts"] == {
        "cdg": "macro_artifact_cdg",
        "match_results": "hunter_batch_match",
    }
    assert payload["metadata"]["single_agent"]["attempt_history"] == [
        "direct_macro_match",
        "macro_decompose",
        "match_decomposed",
    ]
    assert payload["metadata"]["single_agent"]["tool_metrics"]["artifact.match_goal"]["dispatches"] == 1
    assert payload["metadata"]["single_agent"]["tool_metrics"]["hunter.match_batch"]["dispatches"] == 1
    assert (output_dir / "cdg.json").exists()
    assert (output_dir / "matches.json").exists()
    assert env.closed is True
