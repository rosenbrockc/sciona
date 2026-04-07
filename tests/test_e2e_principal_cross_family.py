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
from sciona.principal.atom_ledger import AtomLedger, compute_slot_signature
from sciona.principal.admissibility import (
    AdmissibilityContext,
    AdmissibilityDecision,
    AdmissibilityDisposition,
    AdmissibilityEvaluator,
    AdmissibilityReport,
)
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


def _build_catalog_with_local_alternative() -> PrimitiveCatalog:
    catalog = _build_catalog()
    catalog.add(
        AlgorithmicPrimitive(
            name="ageoa.signal.filter_signal_fast",
            source="ageoa.signal",
            category=ConceptType.SIGNAL_FILTER,
            description="Same-family faster filtering primitive.",
            inputs=[IOSpec(name="signal", type_desc="ndarray")],
            outputs=[IOSpec(name="filtered_signal", type_desc="ndarray")],
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


class _ProposalPreferenceSandbox:
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
        source_text = bundle.source_path.read_text()
        if "ageoa.signal.filter_signal_fast" in source_text:
            loss = 40.0
            telemetry = {
                "filter_signal": NodeTelemetry(
                    node_id="filter_signal",
                    execution_time_ms=40.0,
                    peak_memory_bytes=1024,
                    error_expansion=0.0,
                )
            }
        elif "ageoa.statistics.score_signal_quality" in source_text:
            loss = 60.0
            telemetry = {
                "filter_signal": NodeTelemetry(
                    node_id="filter_signal",
                    execution_time_ms=52.0,
                    peak_memory_bytes=2048,
                    error_expansion=0.0,
                ),
                "score_signal_quality": NodeTelemetry(
                    node_id="score_signal_quality",
                    execution_time_ms=8.0,
                    peak_memory_bytes=512,
                    error_expansion=0.0,
                ),
            }
        else:
            loss = 100.0
            telemetry = {
                "filter_signal": NodeTelemetry(
                    node_id="filter_signal",
                    execution_time_ms=80.0,
                    peak_memory_bytes=2048,
                    error_expansion=0.0,
                )
            }
        return BenchmarkResult(
            global_loss=loss,
            node_telemetry=telemetry,
            runtime_artifacts={
                "stdout_payload": {"metric": "latency", "global_loss": loss}
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
                        asset_id="expansion.statistics.quality_gate.v1",
                        asset_version="phase3.v1",
                        asset_family="statistics",
                        asset_source_kind="local_asset",
                        asset_review_status="transitional",
                        asset_operation="Inject Signal Quality Gate",
                    ),
                ),
                applied_assets=(
                    {
                        "asset_id": "expansion.statistics.quality_gate.v1",
                        "asset_version": "phase3.v1",
                        "asset_family": "statistics",
                        "asset_source_kind": "local_asset",
                        "asset_review_status": "transitional",
                        "asset_operation": "Inject Signal Quality Gate",
                        "rule_name": "inject_signal_quality_gate",
                    },
                ),
                expanded=True,
            )
        return ExpansionResult(
            cdg=cdg,
            applied_rules=(),
            diagnostics=(),
            expanded=False,
        )


class _RouteToRefinementEvaluator(AdmissibilityEvaluator):
    def __init__(self) -> None:
        self.call_count = 0
        super().__init__([])

    def evaluate(self, context: AdmissibilityContext) -> AdmissibilityReport:
        self.call_count += 1
        return AdmissibilityReport(
            decisions=(
                AdmissibilityDecision(
                    rule_id="unstable_event_intervals",
                    disposition=AdmissibilityDisposition.ROUTE_TO_REFINEMENT,
                    summary="Detected intervals are unstable enough to require refinement.",
                    severity=0.8,
                    evidence="Synthetic refinement-routing test evidence.",
                    metric_name="events.outlier_fraction",
                    observed_value=0.42,
                    threshold=0.15,
                    family="signal_event_rate",
                    suggested_refinement="inject_signal_quality_gate",
                ),
            )
        )


def _mock_match_results_fn(cdg: CDGExport) -> list:
    return []


async def _mock_synthesize_fn(cdg: CDGExport, match_results: list) -> ExportBundle:
    tmp = Path(tempfile.mkdtemp())
    source = tmp / "cross_family.py"
    primitive_names = [
        str(node.matched_primitive or "")
        for node in cdg.nodes
        if node.status == NodeStatus.ATOMIC
    ]
    source.write_text("\n".join(primitive_names) + "\n")
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
        assert expansion_engine.call_count == 1

    def test_trial_history_records_expansion_metadata(self, principal_cross_family_result):
        state, _, _, _ = principal_cross_family_result
        first_trial = state.trial_history[0]
        assert first_trial["expansion"]["applied"] is True
        assert first_trial["expansion"]["rules_applied"] == [
            "inject_signal_quality_gate"
        ]
        assert first_trial["expansion"]["diagnostic_count"] == 1
        assert first_trial["expansion"]["applied_assets"][0]["asset_id"] == (
            "expansion.statistics.quality_gate.v1"
        )
        assert first_trial["expansion"]["diagnostic_assets"][0]["asset_operation"] == (
            "Inject Signal Quality Gate"
        )
        assert first_trial["expansion"]["context_summary"]["has_eval_result"] is True

    def test_second_trial_records_cross_family_structure(self, principal_cross_family_result):
        state, _, _, _ = principal_cross_family_result
        second_trial = state.trial_history[1]
        structure = second_trial["structure"]
        assert second_trial["reused_cached_evaluation"] is True
        assert structure["distinct_primitive_family_count"] == 2
        assert set(structure["distinct_primitive_families"]) == {
            "ageoa.signal",
            "ageoa.statistics",
        }
        assert structure["cross_family_edge_count"] == 1
        assert structure["cross_family_node_count"] == 2
        assert structure["family_entropy"] > 0.0
        assert state.best_loss == 60.0


@pytest.fixture(scope="module")
def principal_admissibility_refinement_result():
    async def _run():
        architect = _SingleShotArchitect()
        sandbox = _ExpansionAwareSandbox()
        expansion_engine = _SingleShotExpansionEngine()
        admissibility = _RouteToRefinementEvaluator()
        deps = PrincipalDeps(
            architect=architect,
            sandbox=sandbox,
            match_results_fn=_mock_match_results_fn,
            synthesize_fn=_mock_synthesize_fn,
            catalog=_build_catalog(),
            expansion_engine=expansion_engine,
            admissibility_evaluator=admissibility,
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
        return state, sandbox, architect, expansion_engine, admissibility

    return asyncio.get_event_loop().run_until_complete(_run())


class TestPrincipalAdmissibilityRefinementE2E:
    def test_admissibility_routes_directly_to_refinement(
        self, principal_admissibility_refinement_result
    ):
        state, sandbox, architect, expansion_engine, admissibility = (
            principal_admissibility_refinement_result
        )
        assert len(state.trial_history) == 2
        assert sandbox.call_count == 2
        assert len(architect.decompose_calls) == 1
        assert expansion_engine.call_count >= 1
        assert admissibility.call_count >= 1
        assert state.bottleneck_node_id == ""
        assert state.trial_history[0]["admissibility"]["routed_to_refinement"] is True
        assert (
            state.trial_history[0]["proposal_selection"]["selected"] == "expansion"
        )


@pytest.fixture(scope="module")
def principal_cross_family_preference_result():
    async def _run():
        architect = _SingleShotArchitect()
        sandbox = _ProposalPreferenceSandbox()
        expansion_engine = _SingleShotExpansionEngine()
        catalog = _build_catalog_with_local_alternative()
        ledger = AtomLedger()
        baseline = _build_baseline_cdg()
        node = baseline.nodes[1]
        root = baseline.nodes[0]
        slot = compute_slot_signature(node, root)
        for trial in range(5):
            ledger.record(slot, "ageoa.signal.filter_signal_basic", 80.0, trial=trial)
            ledger.record(slot, "ageoa.signal.filter_signal_fast", 5.0, trial=trial)

        deps = PrincipalDeps(
            architect=architect,
            sandbox=sandbox,
            match_results_fn=_mock_match_results_fn,
            synthesize_fn=_mock_synthesize_fn,
            catalog=catalog,
            atom_ledger=ledger,
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


class TestPrincipalExpansionPreferredOverFallbackE2E:
    def test_local_mutation_can_beat_expansion_when_it_scores_better(
        self, principal_cross_family_preference_result
    ):
        state, sandbox, architect, expansion_engine = principal_cross_family_preference_result
        assert len(state.trial_history) == 2
        assert sandbox.call_count == 3
        assert len(architect.decompose_calls) == 1
        assert expansion_engine.call_count == 1
        assert state.cdg is not None
        by_name = {
            node.name: node
            for node in state.cdg.nodes
            if node.status == NodeStatus.ATOMIC
        }
        assert "Score Signal Quality" not in by_name
        assert by_name["Filter Signal"].matched_primitive == "ageoa.signal.filter_signal_fast"
        proposal = state.trial_history[0]["proposal_selection"]
        assert proposal["selected"] == "local_mutation"


def _build_graph_opt_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="graph_opt_root",
        name="Graph Optimization Pipeline",
        description="Optimize a graph update step.",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.DECOMPOSED,
        depth=0,
        children=["relax_edges"],
    )
    relax_edges = AlgorithmicNode(
        node_id="relax_edges",
        parent_id="graph_opt_root",
        name="Relax Edges",
        description="Apply a repeated relaxation update over a dense state vector.",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        depth=1,
        matched_primitive="ageoa.graph.relax_edges_iterative",
        inputs=[IOSpec(name="state", type_desc="np.ndarray")],
        outputs=[IOSpec(name="updated_state", type_desc="np.ndarray")],
        type_signature="np.ndarray -> np.ndarray",
    )
    return CDGExport(nodes=[root, relax_edges], edges=[])


def _build_graph_opt_catalog() -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    catalog.add(
        AlgorithmicPrimitive(
            name="ageoa.graph.relax_edges_iterative",
            source="ageoa.graph",
            category=ConceptType.GRAPH_OPTIMIZATION,
            description="Graph-local iterative edge relaxation.",
            inputs=[IOSpec(name="state", type_desc="np.ndarray")],
            outputs=[IOSpec(name="updated_state", type_desc="np.ndarray")],
        )
    )
    catalog.add(
        AlgorithmicPrimitive(
            name="ageoa.linalg.solve_relaxation_system",
            source="ageoa.linalg",
            category=ConceptType.ALGEBRA,
            description="Linear-algebra solver for a structurally equivalent state update.",
            inputs=[IOSpec(name="state", type_desc="np.ndarray")],
            outputs=[IOSpec(name="updated_state", type_desc="np.ndarray")],
        )
    )
    return catalog


class _GraphOptArchitect:
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
        return _build_graph_opt_cdg()

    async def get_state_history(self, thread_id: str) -> list[dict]:
        return []

    async def fork(
        self,
        source_thread_id: str,
        checkpoint_id: str,
        new_thread_id: str | None = None,
    ) -> str:
        raise AssertionError("time-travel should not be reached in the ledger mutation test")


class _NoopExpansionEngine:
    def __init__(self) -> None:
        self.call_count = 0

    def expand(self, cdg: CDGExport, context) -> ExpansionResult:
        self.call_count += 1
        return ExpansionResult(cdg=cdg, applied_rules=(), diagnostics=(), expanded=False)


class _GraphOptMutationSandbox:
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
        source_text = bundle.source_path.read_text()
        if "ageoa.linalg.solve_relaxation_system" in source_text:
            return BenchmarkResult(
                global_loss=30.0,
                node_telemetry={
                    "relax_edges": NodeTelemetry(
                        node_id="relax_edges",
                        execution_time_ms=30.0,
                        peak_memory_bytes=1024,
                        error_expansion=0.0,
                    )
                },
                runtime_artifacts={
                    "stdout_payload": {"metric": "latency", "global_loss": 30.0}
                },
            )
        return BenchmarkResult(
            global_loss=90.0,
            node_telemetry={
                "relax_edges": NodeTelemetry(
                    node_id="relax_edges",
                    execution_time_ms=90.0,
                    peak_memory_bytes=1024,
                    error_expansion=0.0,
                )
            },
            runtime_artifacts={
                "stdout_payload": {"metric": "latency", "global_loss": 90.0}
            },
        )


@pytest.fixture(scope="module")
def principal_cross_family_mutation_result():
    async def _run():
        architect = _GraphOptArchitect()
        sandbox = _GraphOptMutationSandbox()
        expansion_engine = _NoopExpansionEngine()
        catalog = _build_graph_opt_catalog()
        ledger = AtomLedger()
        cdg = _build_graph_opt_cdg()
        node = cdg.nodes[1]
        root = cdg.nodes[0]
        slot = compute_slot_signature(node, root)
        for trial in range(5):
            ledger.record(slot, "ageoa.graph.relax_edges_iterative", 85.0, trial=trial)
            ledger.record(slot, "ageoa.linalg.solve_relaxation_system", 10.0, trial=trial)

        deps = PrincipalDeps(
            architect=architect,
            sandbox=sandbox,
            match_results_fn=_mock_match_results_fn,
            synthesize_fn=_mock_synthesize_fn,
            catalog=catalog,
            atom_ledger=ledger,
            expansion_engine=expansion_engine,
        )
        graph = build_principal_graph()
        compiled = graph.compile()
        initial_state = PrincipalState(
            goal="Optimize a graph relaxation update",
            metric=OptimizationMetric.LATENCY,
            max_trials=2,
        )
        config = {"configurable": {"deps": deps}}
        result = await compiled.ainvoke(initial_state, config=config)
        state = PrincipalState(**result) if isinstance(result, dict) else result
        return state, sandbox, architect, expansion_engine

    return asyncio.get_event_loop().run_until_complete(_run())


class TestPrincipalCrossFamilyMutationE2E:
    def test_ledger_mutation_adopts_foreign_family_primitive(
        self, principal_cross_family_mutation_result
    ):
        state, sandbox, architect, expansion_engine = principal_cross_family_mutation_result
        assert len(state.trial_history) == 2
        assert sandbox.call_count == 2
        assert len(architect.decompose_calls) == 1
        assert expansion_engine.call_count == 1
        assert state.cdg is not None
        atomic = [node for node in state.cdg.nodes if node.status == NodeStatus.ATOMIC]
        assert len(atomic) == 1
        assert atomic[0].matched_primitive == "ageoa.linalg.solve_relaxation_system"
        assert state.best_loss == 30.0

    def test_mutation_trial_records_foreign_family_binding(
        self, principal_cross_family_mutation_result
    ):
        state, _, _, _ = principal_cross_family_mutation_result
        second_trial = state.trial_history[1]
        assert second_trial["reused_cached_evaluation"] is True
        structure = second_trial["structure"]
        assert structure["topology_changed"] is False
        assert structure["primitive_assignment_changed"] is True
        assert structure["foreign_family_binding_count"] == 1
        assert structure["foreign_family_bindings"] == ["relax_edges"]
        assert structure["distinct_primitive_families"] == ["ageoa.linalg"]


class _HarmfulExpansionSandbox:
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
                global_loss=50.0,
                node_telemetry={
                    "filter_signal": NodeTelemetry(
                        node_id="filter_signal",
                        execution_time_ms=50.0,
                        peak_memory_bytes=2048,
                        error_expansion=0.0,
                    )
                },
                runtime_artifacts={
                    "stdout_payload": {"metric": "latency", "global_loss": 50.0}
                },
            )
        return BenchmarkResult(
            global_loss=75.0,
            node_telemetry={
                "filter_signal": NodeTelemetry(
                    node_id="filter_signal",
                    execution_time_ms=50.0,
                    peak_memory_bytes=2048,
                    error_expansion=0.0,
                ),
                "score_signal_quality": NodeTelemetry(
                    node_id="score_signal_quality",
                    execution_time_ms=25.0,
                    peak_memory_bytes=1024,
                    error_expansion=0.0,
                ),
            },
            runtime_artifacts={
                "stdout_payload": {"metric": "latency", "global_loss": 75.0}
            },
        )


@pytest.fixture(scope="module")
def principal_harmful_expansion_result():
    async def _run():
        architect = _SingleShotArchitect()
        sandbox = _HarmfulExpansionSandbox()
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


class TestPrincipalCrossFamilyRollbackE2E:
    def test_harmful_expansion_is_rejected_at_proposal_selection(
        self, principal_harmful_expansion_result
    ):
        state, sandbox, architect, expansion_engine = principal_harmful_expansion_result
        assert len(state.trial_history) == 1
        assert sandbox.call_count == 2
        assert len(architect.decompose_calls) == 1
        assert expansion_engine.call_count == 1
        assert state.done is True
        assert state.best_loss == 50.0
        assert state.cdg is not None
        atomic_names = {
            node.name for node in state.cdg.nodes if node.status == NodeStatus.ATOMIC
        }
        assert atomic_names == {"Filter Signal"}

    def test_harmful_expansion_records_no_selected_proposal(
        self, principal_harmful_expansion_result
    ):
        state, _, _, _ = principal_harmful_expansion_result
        first_trial = state.trial_history[0]
        assert first_trial["proposal_selection"]["selected"] == ""
        assert first_trial["expansion"]["applied"] is False
