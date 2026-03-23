"""End-to-end Principal acceptance tests for cross-family expansion behavior."""

from __future__ import annotations

import asyncio
import importlib.util
import tempfile
import uuid
from pathlib import Path

import pytest

if importlib.util.find_spec("langgraph") is None:
    pytest.skip("requires langgraph", allow_module_level=True)

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionDiagnostic, ExpansionResult
from sciona.principal.graph import PrincipalDeps, PrincipalState, build_principal_graph
from sciona.principal.models import BenchmarkResult, NodeTelemetry, OptimizationMetric
from sciona.synthesizer.models import ExportBundle


def _build_baseline_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="cross_family_root",
        name="Cross-family signal analysis",
        description="Baseline single-family signal pipeline.",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.DECOMPOSED,
        depth=0,
        children=["filter_signal"],
    )
    filter_signal = AlgorithmicNode(
        node_id="filter_signal",
        parent_id="cross_family_root",
        name="Filter Signal",
        description="Filter the raw signal before downstream processing.",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        depth=1,
        matched_primitive="ageoa.signal.filter_signal_basic",
        inputs=[IOSpec(name="signal", type_desc="ndarray")],
        outputs=[IOSpec(name="filtered_signal", type_desc="ndarray")],
        type_signature="ndarray -> ndarray",
    )
    return CDGExport(nodes=[root, filter_signal], edges=[])


def _build_expanded_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="cross_family_root",
        name="Cross-family signal analysis",
        description="Expanded signal pipeline with a statistics-family quality node.",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.DECOMPOSED,
        depth=0,
        children=["filter_signal", "score_signal_quality"],
    )
    filter_signal = AlgorithmicNode(
        node_id="filter_signal",
        parent_id="cross_family_root",
        name="Filter Signal",
        description="Filter the raw signal before downstream processing.",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        depth=1,
        matched_primitive="ageoa.signal.filter_signal_basic",
        inputs=[IOSpec(name="signal", type_desc="ndarray")],
        outputs=[IOSpec(name="filtered_signal", type_desc="ndarray")],
        type_signature="ndarray -> ndarray",
    )
    score_signal_quality = AlgorithmicNode(
        node_id="score_signal_quality",
        parent_id="cross_family_root",
        name="Score Signal Quality",
        description="Estimate signal quality using a statistics-family primitive.",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.ATOMIC,
        depth=1,
        matched_primitive="ageoa.statistics.score_signal_quality",
        inputs=[IOSpec(name="filtered_signal", type_desc="ndarray")],
        outputs=[IOSpec(name="quality_score", type_desc="float")],
        type_signature="ndarray -> float",
    )
    edge = DependencyEdge(
        source_id="filter_signal",
        target_id="score_signal_quality",
        output_name="filtered_signal",
        input_name="filtered_signal",
        source_type="ndarray",
        target_type="ndarray",
    )
    return CDGExport(nodes=[root, filter_signal, score_signal_quality], edges=[edge])


def _build_catalog() -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    catalog.add(
        AlgorithmicPrimitive(
            name="ageoa.signal.filter_signal_basic",
            source="ageoa.signal",
            category=ConceptType.SIGNAL_FILTER,
            description="Signal-family filtering primitive.",
            inputs=[IOSpec(name="signal", type_desc="ndarray")],
            outputs=[IOSpec(name="filtered_signal", type_desc="ndarray")],
        )
    )
    catalog.add(
        AlgorithmicPrimitive(
            name="ageoa.statistics.score_signal_quality",
            source="ageoa.statistics",
            category=ConceptType.ANALYSIS,
            description="Statistics-family signal quality primitive.",
            inputs=[IOSpec(name="filtered_signal", type_desc="ndarray")],
            outputs=[IOSpec(name="quality_score", type_desc="float")],
        )
    )
    return catalog


class _SingleShotArchitect:
    def __init__(self) -> None:
        self.decompose_calls: list[dict[str, str]] = []

    async def decompose(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
    ) -> CDGExport:
        thread_id = thread_id or uuid.uuid4().hex
        self.decompose_calls.append({"goal": goal, "thread_id": thread_id})
        return _build_baseline_cdg()

    async def get_state_history(self, thread_id: str) -> list[dict]:
        return []

    async def fork(
        self,
        source_thread_id: str,
        checkpoint_id: str,
        new_thread_id: str | None = None,
    ) -> str:
        raise AssertionError("time-travel should not be reached in the expansion test")


class _ExpansionAwareSandbox:
    def __init__(self) -> None:
        self.call_count = 0

    async def evaluate(
        self,
        bundle: ExportBundle,
        dataset_path: str,
        metric: OptimizationMetric,
        evaluation_spec: object | None = None,
    ) -> BenchmarkResult:
        self.call_count += 1
        if self.call_count == 1:
            return BenchmarkResult(
                global_loss=100.0,
                node_telemetry={
                    "filter_signal": NodeTelemetry(
                        node_id="filter_signal",
                        execution_time_ms=80.0,
                        peak_memory_bytes=2048,
                        error_expansion=0.0,
                    )
                },
                runtime_artifacts={
                    "stdout_payload": {"metric": "latency", "global_loss": 100.0}
                },
            )
        return BenchmarkResult(
            global_loss=60.0,
            node_telemetry={
                "filter_signal": NodeTelemetry(
                    node_id="filter_signal",
                    execution_time_ms=40.0,
                    peak_memory_bytes=2048,
                    error_expansion=0.0,
                ),
                "score_signal_quality": NodeTelemetry(
                    node_id="score_signal_quality",
                    execution_time_ms=8.0,
                    peak_memory_bytes=512,
                    error_expansion=0.0,
                ),
            },
            runtime_artifacts={
                "stdout_payload": {"metric": "latency", "global_loss": 60.0}
            },
        )


class _SingleShotExpansionEngine:
    def __init__(self) -> None:
        self.call_count = 0

    def expand(self, cdg: CDGExport, context) -> ExpansionResult:
        self.call_count += 1
        if self.call_count == 1:
            return ExpansionResult(
                cdg=_build_expanded_cdg(),
                applied_rules=("inject_signal_quality_gate",),
                diagnostics=(
                    ExpansionDiagnostic(
                        rule_name="inject_signal_quality_gate",
                        severity=0.85,
                        evidence="quality scoring would stabilize downstream behavior",
                        metric_name="quality_instability",
                        metric_value=0.85,
                        threshold=0.30,
                        source_domain="statistics",
                    ),
                ),
                expanded=True,
            )
        return ExpansionResult(
            cdg=cdg,
            applied_rules=(),
            diagnostics=(),
            expanded=False,
        )


def _mock_match_results_fn(cdg: CDGExport) -> list:
    return []


async def _mock_synthesize_fn(cdg: CDGExport, match_results: list) -> ExportBundle:
    tmp = Path(tempfile.mkdtemp())
    source = tmp / "cross_family.py"
    source.write_text("# mock artifact\n")
    return ExportBundle(target="python", output_dir=tmp, source_path=source)


@pytest.fixture(scope="module")
def principal_cross_family_result():
    async def _run():
        architect = _SingleShotArchitect()
        sandbox = _ExpansionAwareSandbox()
        expansion_engine = _SingleShotExpansionEngine()
        deps = PrincipalDeps(
            architect=architect,
            sandbox=sandbox,
            match_results_fn=_mock_match_results_fn,
            synthesize_fn=_mock_synthesize_fn,
            catalog=_build_catalog(),
            expansion_engine=expansion_engine,
        )
        graph = build_principal_graph()
        compiled = graph.compile()
        initial_state = PrincipalState(
            goal="Improve a signal-analysis pipeline with quality control",
            metric=OptimizationMetric.LATENCY,
            max_trials=2,
        )
        config = {"configurable": {"deps": deps}}
        result = await compiled.ainvoke(initial_state, config=config)
        state = PrincipalState(**result) if isinstance(result, dict) else result
        return state, sandbox, architect, expansion_engine

    return asyncio.get_event_loop().run_until_complete(_run())


class TestPrincipalCrossFamilyExpansionE2E:
    def test_expansion_creates_second_trial(self, principal_cross_family_result):
        state, sandbox, architect, expansion_engine = principal_cross_family_result
        assert len(state.trial_history) == 2
        assert state.current_trial == 2
        assert sandbox.call_count == 2
        assert len(architect.decompose_calls) == 1
        assert expansion_engine.call_count == 2

    def test_trial_history_records_expansion_metadata(self, principal_cross_family_result):
        state, _, _, _ = principal_cross_family_result
        first_trial = state.trial_history[0]
        assert first_trial["expansion"]["applied"] is True
        assert first_trial["expansion"]["rules_applied"] == [
            "inject_signal_quality_gate"
        ]
        assert first_trial["expansion"]["diagnostic_count"] == 1
        assert first_trial["expansion"]["context_summary"]["has_eval_result"] is True

    def test_second_trial_records_cross_family_structure(self, principal_cross_family_result):
        state, _, _, _ = principal_cross_family_result
        second_trial = state.trial_history[1]
        structure = second_trial["structure"]
        assert structure["distinct_primitive_family_count"] == 2
        assert set(structure["distinct_primitive_families"]) == {
            "ageoa.signal",
            "ageoa.statistics",
        }
        assert structure["cross_family_edge_count"] == 1
        assert structure["cross_family_node_count"] == 2
        assert structure["family_entropy"] > 0.0
        assert state.best_loss == 60.0
