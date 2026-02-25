"""Principal E2E test: multi-trial optimization over Hodges EMG decomposition.

Exercises the full Principal optimization loop:
  seed → forward → evaluate → gradients → time_travel → forward (loop)

The Architect is a scripted agent that produces two structurally different
decompositions driven by the Principal's constraint injection:
  Trial 1: canonical 6-stage Hodges (baseline)
  Trial 2: alternative 5-stage Hodges (merged baseline removal + test statistic)

No live LLM, no Memgraph, no external services required.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

if importlib.util.find_spec("langgraph") is None:
    pytest.skip("requires langgraph", allow_module_level=True)

from ageom.architect.handoff import CDGExport
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.principal.graph import (
    PrincipalDeps,
    PrincipalState,
    build_principal_graph,
)
from ageom.principal.models import (
    BenchmarkResult,
    NodeTelemetry,
    OptimizationMetric,
)
from ageom.synthesizer.models import ExportBundle


# ---------------------------------------------------------------------------
# Pre-built CDGExports for the two decomposition trials
# ---------------------------------------------------------------------------


def _build_6_stage_cdg() -> CDGExport:
    """Canonical 6-stage Hodges decomposition."""
    root = AlgorithmicNode(
        node_id="hodges_root",
        name="Hodges time-domain EMG onset detection",
        description="Hodges & Bui (2003) EMG onset detection algorithm",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.DECOMPOSED,
        depth=0,
        children=[
            "estimate_rest_baseline_statistics",
            "remove_baseline_offset",
            "compute_normalised_test_statistic",
            "smooth_test_statistic",
            "threshold_crossing_state_machine",
            "merge_adjacent_events",
        ],
    )
    nodes = [
        root,
        AlgorithmicNode(
            node_id="estimate_rest_baseline_statistics",
            parent_id="hodges_root",
            name="Estimate Rest Baseline Statistics",
            description="Compute mean and standard deviation of the rest-segment EMG signal.",
            concept_type=ConceptType.ANALYSIS,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="rest_signal", type_desc="ndarray"),
                IOSpec(name="sampling_rate", type_desc="float"),
            ],
            outputs=[
                IOSpec(name="rest_mean", type_desc="float"),
                IOSpec(name="rest_std", type_desc="float"),
            ],
            type_signature="ndarray -> float -> tuple[float, float]",
        ),
        AlgorithmicNode(
            node_id="remove_baseline_offset",
            parent_id="hodges_root",
            name="Remove Baseline Offset",
            description="Subtract the rest-segment mean from the test signal.",
            concept_type=ConceptType.SIGNAL_FILTER,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="signal", type_desc="ndarray"),
                IOSpec(name="rest_mean", type_desc="float"),
            ],
            outputs=[IOSpec(name="centered_signal", type_desc="ndarray")],
            type_signature="ndarray -> float -> ndarray",
        ),
        AlgorithmicNode(
            node_id="compute_normalised_test_statistic",
            parent_id="hodges_root",
            name="Compute Normalised Test Statistic",
            description="Form h(n) = |centered(n)| / sigma_rest.",
            concept_type=ConceptType.ARITHMETIC,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="centered_signal", type_desc="ndarray"),
                IOSpec(name="rest_std", type_desc="float"),
            ],
            outputs=[IOSpec(name="test_statistic", type_desc="ndarray")],
            type_signature="ndarray -> float -> ndarray",
        ),
        AlgorithmicNode(
            node_id="smooth_test_statistic",
            parent_id="hodges_root",
            name="Smooth Test Statistic",
            description="Apply a moving-average window to suppress transient spikes.",
            concept_type=ConceptType.SIGNAL_FILTER,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="test_statistic", type_desc="ndarray"),
                IOSpec(name="sampling_rate", type_desc="float"),
            ],
            outputs=[IOSpec(name="smoothed_statistic", type_desc="ndarray")],
            type_signature="ndarray -> float -> ndarray",
        ),
        AlgorithmicNode(
            node_id="threshold_crossing_state_machine",
            parent_id="hodges_root",
            name="Threshold Crossing State Machine",
            description="Detect onset and offset transitions.",
            concept_type=ConceptType.SEQUENTIAL_FILTER,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="smoothed_statistic", type_desc="ndarray"),
                IOSpec(name="threshold", type_desc="float"),
            ],
            outputs=[
                IOSpec(name="raw_onsets", type_desc="ndarray"),
                IOSpec(name="raw_offsets", type_desc="ndarray"),
            ],
            type_signature="ndarray -> float -> tuple[ndarray, ndarray]",
        ),
        AlgorithmicNode(
            node_id="merge_adjacent_events",
            parent_id="hodges_root",
            name="Merge Adjacent Events",
            description="Merge onset/offset pairs within refractory window.",
            concept_type=ConceptType.SET_THEORY,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="raw_onsets", type_desc="ndarray"),
                IOSpec(name="raw_offsets", type_desc="ndarray"),
            ],
            outputs=[
                IOSpec(name="onsets", type_desc="ndarray"),
                IOSpec(name="offsets", type_desc="ndarray"),
            ],
            type_signature="ndarray -> ndarray -> tuple[ndarray, ndarray]",
        ),
    ]
    edges = [
        DependencyEdge(
            source_id="estimate_rest_baseline_statistics",
            target_id="remove_baseline_offset",
            output_name="rest_mean", input_name="rest_mean",
            source_type="float", target_type="float",
        ),
        DependencyEdge(
            source_id="estimate_rest_baseline_statistics",
            target_id="compute_normalised_test_statistic",
            output_name="rest_std", input_name="rest_std",
            source_type="float", target_type="float",
        ),
        DependencyEdge(
            source_id="remove_baseline_offset",
            target_id="compute_normalised_test_statistic",
            output_name="centered_signal", input_name="centered_signal",
            source_type="ndarray", target_type="ndarray",
        ),
        DependencyEdge(
            source_id="compute_normalised_test_statistic",
            target_id="smooth_test_statistic",
            output_name="test_statistic", input_name="test_statistic",
            source_type="ndarray", target_type="ndarray",
        ),
        DependencyEdge(
            source_id="smooth_test_statistic",
            target_id="threshold_crossing_state_machine",
            output_name="smoothed_statistic", input_name="smoothed_statistic",
            source_type="ndarray", target_type="ndarray",
        ),
        DependencyEdge(
            source_id="threshold_crossing_state_machine",
            target_id="merge_adjacent_events",
            output_name="raw_onsets", input_name="raw_onsets",
            source_type="ndarray", target_type="ndarray",
        ),
        DependencyEdge(
            source_id="threshold_crossing_state_machine",
            target_id="merge_adjacent_events",
            output_name="raw_offsets", input_name="raw_offsets",
            source_type="ndarray", target_type="ndarray",
        ),
    ]
    return CDGExport(
        nodes=nodes,
        edges=edges,
        metadata={"goal": "Hodges EMG onset detection", "paradigm": "signal_filter"},
    )


def _build_5_stage_cdg() -> CDGExport:
    """Alternative 5-stage Hodges decomposition (merged baseline + test stat)."""
    root = AlgorithmicNode(
        node_id="hodges_root_v2",
        name="Hodges time-domain EMG onset detection",
        description="Hodges & Bui (2003) EMG onset detection — optimized variant",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.DECOMPOSED,
        depth=0,
        children=[
            "estimate_rest_baseline_statistics_v2",
            "compute_centred_test_statistic",
            "smooth_test_statistic_v2",
            "threshold_crossing_state_machine_v2",
            "merge_adjacent_events_v2",
        ],
    )
    nodes = [
        root,
        AlgorithmicNode(
            node_id="estimate_rest_baseline_statistics_v2",
            parent_id="hodges_root_v2",
            name="Estimate Rest Baseline Statistics",
            description="Compute mean and std of rest-segment EMG.",
            concept_type=ConceptType.ANALYSIS,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="rest_signal", type_desc="ndarray"),
                IOSpec(name="sampling_rate", type_desc="float"),
            ],
            outputs=[
                IOSpec(name="rest_mean", type_desc="float"),
                IOSpec(name="rest_std", type_desc="float"),
            ],
            type_signature="ndarray -> float -> tuple[float, float]",
        ),
        AlgorithmicNode(
            node_id="compute_centred_test_statistic",
            parent_id="hodges_root_v2",
            name="Compute Centred Test Statistic",
            description="Subtract rest mean and normalise: h(n) = |x(n)-mu|/sigma.",
            concept_type=ConceptType.ARITHMETIC,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="signal", type_desc="ndarray"),
                IOSpec(name="rest_mean", type_desc="float"),
                IOSpec(name="rest_std", type_desc="float"),
            ],
            outputs=[IOSpec(name="test_statistic", type_desc="ndarray")],
            type_signature="ndarray -> float -> float -> ndarray",
        ),
        AlgorithmicNode(
            node_id="smooth_test_statistic_v2",
            parent_id="hodges_root_v2",
            name="Smooth Test Statistic",
            description="Moving-average smoothing of test statistic.",
            concept_type=ConceptType.SIGNAL_FILTER,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="test_statistic", type_desc="ndarray"),
                IOSpec(name="sampling_rate", type_desc="float"),
            ],
            outputs=[IOSpec(name="smoothed_statistic", type_desc="ndarray")],
            type_signature="ndarray -> float -> ndarray",
        ),
        AlgorithmicNode(
            node_id="threshold_crossing_state_machine_v2",
            parent_id="hodges_root_v2",
            name="Threshold Crossing State Machine",
            description="Detect onset and offset transitions.",
            concept_type=ConceptType.SEQUENTIAL_FILTER,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="smoothed_statistic", type_desc="ndarray"),
                IOSpec(name="threshold", type_desc="float"),
            ],
            outputs=[
                IOSpec(name="raw_onsets", type_desc="ndarray"),
                IOSpec(name="raw_offsets", type_desc="ndarray"),
            ],
            type_signature="ndarray -> float -> tuple[ndarray, ndarray]",
        ),
        AlgorithmicNode(
            node_id="merge_adjacent_events_v2",
            parent_id="hodges_root_v2",
            name="Merge Adjacent Events",
            description="Merge onset/offset pairs within refractory window.",
            concept_type=ConceptType.SET_THEORY,
            status=NodeStatus.ATOMIC,
            depth=1,
            inputs=[
                IOSpec(name="raw_onsets", type_desc="ndarray"),
                IOSpec(name="raw_offsets", type_desc="ndarray"),
            ],
            outputs=[
                IOSpec(name="onsets", type_desc="ndarray"),
                IOSpec(name="offsets", type_desc="ndarray"),
            ],
            type_signature="ndarray -> ndarray -> tuple[ndarray, ndarray]",
        ),
    ]
    edges = [
        DependencyEdge(
            source_id="estimate_rest_baseline_statistics_v2",
            target_id="compute_centred_test_statistic",
            output_name="rest_mean", input_name="rest_mean",
            source_type="float", target_type="float",
        ),
        DependencyEdge(
            source_id="estimate_rest_baseline_statistics_v2",
            target_id="compute_centred_test_statistic",
            output_name="rest_std", input_name="rest_std",
            source_type="float", target_type="float",
        ),
        DependencyEdge(
            source_id="compute_centred_test_statistic",
            target_id="smooth_test_statistic_v2",
            output_name="test_statistic", input_name="test_statistic",
            source_type="ndarray", target_type="ndarray",
        ),
        DependencyEdge(
            source_id="smooth_test_statistic_v2",
            target_id="threshold_crossing_state_machine_v2",
            output_name="smoothed_statistic", input_name="smoothed_statistic",
            source_type="ndarray", target_type="ndarray",
        ),
        DependencyEdge(
            source_id="threshold_crossing_state_machine_v2",
            target_id="merge_adjacent_events_v2",
            output_name="raw_onsets", input_name="raw_onsets",
            source_type="ndarray", target_type="ndarray",
        ),
        DependencyEdge(
            source_id="threshold_crossing_state_machine_v2",
            target_id="merge_adjacent_events_v2",
            output_name="raw_offsets", input_name="raw_offsets",
            source_type="ndarray", target_type="ndarray",
        ),
    ]
    return CDGExport(
        nodes=nodes,
        edges=edges,
        metadata={"goal": "Hodges EMG onset detection (optimized)", "paradigm": "signal_filter"},
    )


# ---------------------------------------------------------------------------
# Scripted DecompositionAgent (duck-types the real one)
# ---------------------------------------------------------------------------


class _ScriptedDecompositionAgent:
    """Scripted architect that returns pre-built CDGs.

    - First call to ``decompose()``: returns 6-stage canonical CDG.
    - Subsequent calls (after constraint injection via time_travel): returns
      5-stage alternative CDG.
    - Supports ``get_state_history()`` and ``fork()`` for time-travel.
    """

    def __init__(self) -> None:
        self.decompose_calls: list[dict[str, str]] = []
        # Simulate checkpoint history: thread_id -> list of snapshots
        self._history: dict[str, list[dict]] = {}
        # Internal compiled graph reference (needed by time_travel_update)
        self._graph = _FakeCompiledGraph(self)

    async def decompose(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
    ) -> CDGExport:
        if thread_id is None:
            thread_id = uuid.uuid4().hex

        self.decompose_calls.append({"goal": goal, "thread_id": thread_id})

        if "CONSTRAINT:" in goal:
            cdg = _build_5_stage_cdg()
        else:
            cdg = _build_6_stage_cdg()

        # Store a checkpoint snapshot for this thread
        checkpoint_id = uuid.uuid4().hex
        if thread_id not in self._history:
            self._history[thread_id] = []
        self._history[thread_id].append(
            {
                "values": {
                    "goal": goal,
                    "nodes": list(cdg.nodes),
                    "edges": list(cdg.edges),
                    "done": True,
                },
                "checkpoint_id": checkpoint_id,
            }
        )

        return cdg

    async def get_state_history(self, thread_id: str) -> list[dict]:
        return list(reversed(self._history.get(thread_id, [])))

    async def fork(
        self,
        source_thread_id: str,
        checkpoint_id: str,
        new_thread_id: str | None = None,
    ) -> str:
        if new_thread_id is None:
            new_thread_id = uuid.uuid4().hex
        # Copy history from source to new thread
        source = self._history.get(source_thread_id, [])
        self._history[new_thread_id] = [
            s for s in source if s["checkpoint_id"] == checkpoint_id
        ] or list(source)
        return new_thread_id


class _FakeStateSnapshot:
    """Mimics LangGraph state snapshot for time_travel_update."""

    def __init__(self, values: dict) -> None:
        self.values = values


class _FakeCompiledGraph:
    """Mimics the compiled graph interface needed by time_travel_update."""

    def __init__(self, agent: _ScriptedDecompositionAgent) -> None:
        self._agent = agent

    async def aget_state(self, config: dict) -> _FakeStateSnapshot:
        thread_id = config["configurable"]["thread_id"]
        history = self._agent._history.get(thread_id, [])
        if history:
            return _FakeStateSnapshot(dict(history[-1]["values"]))
        return _FakeStateSnapshot({"goal": "", "done": False, "error": ""})

    async def aupdate_state(self, config: dict, values: dict) -> None:
        thread_id = config["configurable"]["thread_id"]
        if thread_id not in self._agent._history:
            self._agent._history[thread_id] = []
        checkpoint_id = uuid.uuid4().hex
        self._agent._history[thread_id].append(
            {"values": dict(values), "checkpoint_id": checkpoint_id}
        )


# ---------------------------------------------------------------------------
# Mock dependencies
# ---------------------------------------------------------------------------


class _MockSandbox:
    """Mock ExecutionSandbox returning decreasing loss across trials."""

    def __init__(self) -> None:
        self.call_count = 0
        self.evaluations: list[BenchmarkResult] = []

    async def evaluate(
        self,
        bundle: ExportBundle,
        dataset_path: str,
        metric: OptimizationMetric,
    ) -> BenchmarkResult:
        self.call_count += 1
        if self.call_count == 1:
            # Trial 1: high loss, bottleneck on smooth_test_statistic
            result = BenchmarkResult(
                global_loss=100.0,
                node_telemetry={
                    "estimate_rest_baseline_statistics": NodeTelemetry(
                        node_id="estimate_rest_baseline_statistics",
                        execution_time_ms=5.0,
                        peak_memory_bytes=1024,
                        error_expansion=0.0,
                    ),
                    "remove_baseline_offset": NodeTelemetry(
                        node_id="remove_baseline_offset",
                        execution_time_ms=3.0,
                        peak_memory_bytes=1024,
                        error_expansion=0.0,
                    ),
                    "compute_normalised_test_statistic": NodeTelemetry(
                        node_id="compute_normalised_test_statistic",
                        execution_time_ms=4.0,
                        peak_memory_bytes=1024,
                        error_expansion=0.0,
                    ),
                    "smooth_test_statistic": NodeTelemetry(
                        node_id="smooth_test_statistic",
                        execution_time_ms=80.0,
                        peak_memory_bytes=8192,
                        error_expansion=0.0,
                    ),
                    "threshold_crossing_state_machine": NodeTelemetry(
                        node_id="threshold_crossing_state_machine",
                        execution_time_ms=5.0,
                        peak_memory_bytes=1024,
                        error_expansion=0.0,
                    ),
                    "merge_adjacent_events": NodeTelemetry(
                        node_id="merge_adjacent_events",
                        execution_time_ms=3.0,
                        peak_memory_bytes=512,
                        error_expansion=0.0,
                    ),
                },
            )
        else:
            # Trial 2: lower loss (5-stage decomposition is "faster")
            result = BenchmarkResult(
                global_loss=30.0,
                node_telemetry={
                    "estimate_rest_baseline_statistics_v2": NodeTelemetry(
                        node_id="estimate_rest_baseline_statistics_v2",
                        execution_time_ms=5.0,
                        peak_memory_bytes=1024,
                        error_expansion=0.0,
                    ),
                    "compute_centred_test_statistic": NodeTelemetry(
                        node_id="compute_centred_test_statistic",
                        execution_time_ms=6.0,
                        peak_memory_bytes=1024,
                        error_expansion=0.0,
                    ),
                    "smooth_test_statistic_v2": NodeTelemetry(
                        node_id="smooth_test_statistic_v2",
                        execution_time_ms=10.0,
                        peak_memory_bytes=2048,
                        error_expansion=0.0,
                    ),
                    "threshold_crossing_state_machine_v2": NodeTelemetry(
                        node_id="threshold_crossing_state_machine_v2",
                        execution_time_ms=5.0,
                        peak_memory_bytes=1024,
                        error_expansion=0.0,
                    ),
                    "merge_adjacent_events_v2": NodeTelemetry(
                        node_id="merge_adjacent_events_v2",
                        execution_time_ms=4.0,
                        peak_memory_bytes=512,
                        error_expansion=0.0,
                    ),
                },
            )
        self.evaluations.append(result)
        return result


def _mock_match_results_fn(cdg: CDGExport) -> list:
    """Return empty match results — the sandbox is fully mocked."""
    return []


async def _mock_synthesize_fn(cdg: CDGExport, match_results: list) -> ExportBundle:
    """Return a minimal ExportBundle for the mock sandbox."""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    source = tmp / "hodges.py"
    source.write_text("# mock artifact\n")
    return ExportBundle(
        target="python",
        output_dir=tmp,
        source_path=source,
    )


# ---------------------------------------------------------------------------
# Shared fixture: run the principal graph once for all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def principal_result():
    """Run the Principal graph with 2 trials and return the final state + deps."""
    import asyncio

    async def _run():
        architect = _ScriptedDecompositionAgent()
        sandbox = _MockSandbox()

        deps = PrincipalDeps(
            architect=architect,
            sandbox=sandbox,
            match_results_fn=_mock_match_results_fn,
            synthesize_fn=_mock_synthesize_fn,
        )

        graph = build_principal_graph()
        compiled = graph.compile()

        initial_state = PrincipalState(
            goal="Hodges time-domain EMG onset detection",
            metric=OptimizationMetric.LATENCY,
            max_trials=2,
        )

        config = {"configurable": {"deps": deps}}
        result = await compiled.ainvoke(initial_state, config=config)

        # LangGraph may return a dict; wrap it back into PrincipalState
        if isinstance(result, dict):
            state = PrincipalState(**result)
        else:
            state = result

        return state, sandbox, architect

    return asyncio.get_event_loop().run_until_complete(_run())


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPrincipalHodgesE2E:
    """End-to-end tests for the Principal optimization loop on Hodges EMG."""

    def test_principal_runs_two_trials(self, principal_result):
        """Principal completes 2 trials, trial_history has 2 entries."""
        state, _, _ = principal_result
        assert len(state.trial_history) == 2, (
            f"Expected 2 trials, got {len(state.trial_history)}: {state.trial_history}"
        )
        assert state.current_trial == 2

    def test_second_trial_has_lower_loss(self, principal_result):
        """Trial 2 loss < Trial 1 loss; best_loss tracks the minimum."""
        state, _, _ = principal_result
        trial_1_loss = state.trial_history[0]["loss"]
        trial_2_loss = state.trial_history[1]["loss"]
        assert trial_2_loss < trial_1_loss, (
            f"Expected trial 2 loss ({trial_2_loss}) < trial 1 loss ({trial_1_loss})"
        )
        assert state.best_loss == trial_2_loss

    def test_sandbox_evaluated_twice(self, principal_result):
        """The sandbox evaluate() was called exactly twice."""
        _, sandbox, _ = principal_result
        assert sandbox.call_count == 2

    def test_trials_produce_different_cdgs(self, principal_result):
        """The two trials' CDGs have different node counts (6 vs 5 atomic leaves)."""
        state, _, _ = principal_result
        # The final CDG is from trial 2; we check it has a different
        # structure from the trial 1 baseline.
        cdg = state.cdg
        assert cdg is not None
        atomic_leaves = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
        # Trial 2 (5-stage) should have 5 atomic leaves
        assert len(atomic_leaves) == 5, (
            f"Expected 5 atomic leaves (5-stage alt), got {len(atomic_leaves)}: "
            f"{[n.name for n in atomic_leaves]}"
        )

    def test_bottleneck_identified_after_trial_1(self, principal_result):
        """CreditAssigner identifies a bottleneck node after trial 1."""
        state, _, _ = principal_result
        # The smooth_test_statistic node consumed 80% of latency in trial 1
        assert state.bottleneck_node_id, "No bottleneck identified"
        assert "smooth" in state.bottleneck_node_id.lower(), (
            f"Expected 'smooth' bottleneck, got '{state.bottleneck_node_id}'"
        )

    def test_time_travel_forks_new_thread(self, principal_result):
        """thread_id changes between trials (fork happened)."""
        state, _, _ = principal_result
        # Trial history records thread_ids; they should differ
        t1_thread = state.trial_history[0]["thread_id"]
        t2_thread = state.trial_history[1]["thread_id"]
        assert t1_thread != t2_thread, (
            f"Expected different thread_ids, both are '{t1_thread}'"
        )

    def test_constraint_injected_into_goal(self, principal_result):
        """The second decomposition call received the CONSTRAINT injection."""
        _, _, architect = principal_result
        assert len(architect.decompose_calls) >= 2, (
            f"Expected >=2 decompose calls, got {len(architect.decompose_calls)}"
        )
        # The second call should contain CONSTRAINT
        second_goal = architect.decompose_calls[1]["goal"]
        assert "CONSTRAINT:" in second_goal, (
            f"Expected CONSTRAINT in second goal, got: {second_goal[:200]}"
        )

    def test_full_loop_seed_to_convergence(self, principal_result):
        """Combined: full loop completes, best trial uses the 5-stage CDG."""
        state, sandbox, architect = principal_result

        # Loop completed
        assert state.current_trial == 2
        assert len(state.trial_history) == 2

        # Best loss is from trial 2
        assert state.best_loss == 30.0

        # Final CDG is the 5-stage alternative
        cdg = state.cdg
        assert cdg is not None
        node_names = {n.name for n in cdg.nodes if n.status == NodeStatus.ATOMIC}
        assert "Compute Centred Test Statistic" in node_names, (
            f"Expected 5-stage alt node, got: {node_names}"
        )
        # Should NOT have the 6-stage split nodes
        assert "Remove Baseline Offset" not in node_names
        assert "Compute Normalised Test Statistic" not in node_names
