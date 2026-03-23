"""Tests for the sciona.principal module."""

from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.models import (
    AlgorithmicPrimitive,
    AlgorithmicNode,
    ConceptType,
    IOSpec,
    NodeStatus,
    ParamStatus,
    PrimitiveParamSpec,
)
from sciona.principal.models import (
    BenchmarkResult,
    NodeGradient,
    NodeTelemetry,
    OptimizationMetric,
)
from sciona.synthesizer.ghost_sim import GhostSimReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_telemetry(
    node_id: str, time_ms: float, mem: int, err: float
) -> NodeTelemetry:
    return NodeTelemetry(
        node_id=node_id,
        execution_time_ms=time_ms,
        peak_memory_bytes=mem,
        error_expansion=err,
    )


def _make_cdg(*atomic_specs: tuple[str, str]) -> CDGExport:
    """Build a minimal CDGExport with ATOMIC leaf nodes.

    Each *atomic_spec* is (node_id, name).
    """
    nodes = [
        AlgorithmicNode(
            node_id=nid,
            name=name,
            description=f"Test node {name}",
            concept_type=ConceptType.ARITHMETIC,
            status=NodeStatus.ATOMIC,
        )
        for nid, name in atomic_specs
    ]
    return CDGExport(nodes=nodes, edges=[])


def _make_benchmark(*node_specs: tuple[str, float, int, float]) -> BenchmarkResult:
    """Build a BenchmarkResult from (node_id, time_ms, peak_mem, error_expansion) tuples."""
    telemetry = {}
    total_loss = 0.0
    for nid, t, m, e in node_specs:
        telemetry[nid] = _make_telemetry(nid, t, m, e)
        total_loss += t
    return BenchmarkResult(global_loss=total_loss, node_telemetry=telemetry)


# ===================================================================
# Tests for sciona.principal.models
# ===================================================================


class TestOptimizationMetric:
    def test_values(self):
        assert OptimizationMetric.LATENCY.value == "latency"
        assert OptimizationMetric.MEMORY.value == "memory"
        assert OptimizationMetric.PRECISION.value == "precision"
        assert OptimizationMetric.FLOP_COUNT.value == "flop_count"
        assert OptimizationMetric.STRUCTURE.value == "structure"

    def test_str_enum(self):
        assert isinstance(OptimizationMetric.LATENCY, str)
        assert OptimizationMetric("latency") == OptimizationMetric.LATENCY


class TestNodeTelemetry:
    def test_construction(self):
        t = _make_telemetry("n1", 12.5, 1024, 0.01)
        assert t.node_id == "n1"
        assert t.execution_time_ms == 12.5
        assert t.peak_memory_bytes == 1024
        assert t.error_expansion == 0.01

    def test_frozen(self):
        t = _make_telemetry("n1", 1.0, 100, 0.0)
        with pytest.raises(Exception):
            t.node_id = "n2"


class TestBenchmarkResult:
    def test_empty(self):
        br = BenchmarkResult(global_loss=0.5)
        assert br.node_telemetry == {}

    def test_with_telemetry(self):
        tel = {"a": _make_telemetry("a", 1.0, 100, 0.0)}
        br = BenchmarkResult(global_loss=1.0, node_telemetry=tel)
        assert "a" in br.node_telemetry


class TestOptunaHyperparams:
    def test_suggest_node_params_returns_assignments(self):
        from sciona.principal.hpo import OptunaManager

        catalog = PrimitiveCatalog()
        catalog.add(
            AlgorithmicPrimitive(
                name="compute_event_rate_smoothed",
                source="test",
                category=ConceptType.ANALYSIS,
                description="Smoothed event-rate estimator",
                inputs=[
                    IOSpec(name="events", type_desc="np.ndarray"),
                    IOSpec(name="sampling_rate", type_desc="float"),
                ],
                outputs=[IOSpec(name="rate", type_desc="tuple[np.ndarray, np.ndarray]")],
                tunable_params=[
                    PrimitiveParamSpec(
                        name="smoothing_window",
                        kind="int",
                        default=5,
                        min_value=1,
                        max_value=15,
                        step=2,
                    )
                ],
                param_status=ParamStatus.APPROVED,
            )
        )
        cdg = CDGExport(
            nodes=[
                AlgorithmicNode(
                    node_id="n1",
                    name="Compute Event Rate",
                    description="Rate from event indices",
                    concept_type=ConceptType.ANALYSIS,
                    status=NodeStatus.ATOMIC,
                    matched_primitive="compute_event_rate_smoothed",
                )
            ],
            edges=[],
        )

        manager = OptunaManager(study_name="test-principal")
        suggested = manager.suggest_node_params(
            signature="sigA",
            cdg=cdg,
            catalog=catalog,
        )
        assert set(suggested.assignments) == {"n1"}
        assert "smoothing_window" in suggested.assignments["n1"]
        assert suggested.assignments["n1"]["smoothing_window"] == 5
        manager.complete_trial(
            signature=suggested.signature,
            trial_number=suggested.trial_number,
            loss=1.23,
        )

        second = manager.suggest_node_params(
            signature="sigA",
            cdg=cdg,
            catalog=catalog,
        )
        assert second.assignments["n1"]["smoothing_window"] in {1, 3, 5, 7, 9, 11, 13, 15}


class TestNodeGradient:
    def test_construction(self):
        g = NodeGradient(
            node_id="n1",
            gradient_score=85.0,
            metric_type=OptimizationMetric.LATENCY,
            bottleneck_reason="slow",
        )
        assert g.node_id == "n1"
        assert g.gradient_score == 85.0


# ===================================================================
# Tests for sciona.principal.evaluator
# ===================================================================


class TestParseTrace:
    def test_valid_trace(self, tmp_path: Path):
        from sciona.principal.evaluator import _parse_trace

        trace = tmp_path / "trace.jsonl"
        records = [
            {
                "node_id": "a",
                "execution_time_ms": 10.0,
                "peak_memory_bytes": 100,
                "error_expansion": 0.01,
            },
            {
                "node_id": "b",
                "execution_time_ms": 20.0,
                "peak_memory_bytes": 200,
                "error_expansion": 0.02,
            },
        ]
        trace.write_text("\n".join(json.dumps(r) for r in records))

        result = _parse_trace(trace)
        assert len(result) == 2
        assert result["a"].execution_time_ms == 10.0
        assert result["b"].peak_memory_bytes == 200

    def test_missing_file(self, tmp_path: Path):
        from sciona.principal.evaluator import _parse_trace

        result = _parse_trace(tmp_path / "nope.jsonl")
        assert result == {}

    def test_malformed_line(self, tmp_path: Path):
        from sciona.principal.evaluator import _parse_trace

        trace = tmp_path / "trace.jsonl"
        trace.write_text('not json\n{"node_id": "a", "execution_time_ms": 1}\n')

        result = _parse_trace(trace)
        assert len(result) == 1
        assert "a" in result

    def test_missing_node_id(self, tmp_path: Path):
        from sciona.principal.evaluator import _parse_trace

        trace = tmp_path / "trace.jsonl"
        trace.write_text('{"execution_time_ms": 1}\n')

        result = _parse_trace(trace)
        assert result == {}


class TestComputeLoss:
    def test_latency(self):
        from sciona.principal.evaluator import _compute_loss

        tel = {
            "a": _make_telemetry("a", 10.0, 100, 0.0),
            "b": _make_telemetry("b", 20.0, 200, 0.0),
        }
        loss = _compute_loss(tel, OptimizationMetric.LATENCY, None)
        assert loss == 30.0

    def test_memory(self):
        from sciona.principal.evaluator import _compute_loss

        tel = {
            "a": _make_telemetry("a", 10.0, 100, 0.0),
            "b": _make_telemetry("b", 20.0, 500, 0.0),
        }
        loss = _compute_loss(tel, OptimizationMetric.MEMORY, None)
        assert loss == 500.0

    def test_precision_from_stdout(self):
        from sciona.principal.evaluator import _compute_loss

        tel = {"a": _make_telemetry("a", 1.0, 100, 0.0)}
        stdout = b'some output\n{"mse": 0.042}\n'
        loss = _compute_loss(tel, OptimizationMetric.PRECISION, stdout)
        assert loss == pytest.approx(0.042)

    def test_precision_prefers_explicit_loss(self):
        from sciona.principal.evaluator import _compute_loss

        tel = {"a": _make_telemetry("a", 1.0, 100, 0.0)}
        stdout = b'{"mse": 0.04, "rmse": 0.2, "loss": 0.3}\n'
        loss = _compute_loss(tel, OptimizationMetric.PRECISION, stdout)
        assert loss == pytest.approx(0.3)

    def test_flop_count_proxied_by_latency(self):
        from sciona.principal.evaluator import _compute_loss

        tel = {
            "a": _make_telemetry("a", 5.0, 0, 0.0),
            "b": _make_telemetry("b", 7.0, 0, 0.0),
        }
        loss = _compute_loss(tel, OptimizationMetric.FLOP_COUNT, None)
        assert loss == 12.0

    def test_empty_telemetry_penalty(self):
        from sciona.principal.evaluator import _compute_loss, _FAILURE_PENALTY

        loss = _compute_loss({}, OptimizationMetric.LATENCY, None)
        assert loss == _FAILURE_PENALTY


class TestParsePrecisionLossFromStdout:
    def test_valid(self):
        from sciona.principal.evaluator import _parse_precision_loss_from_stdout

        assert _parse_precision_loss_from_stdout(b'{"mse": 0.5}') == pytest.approx(0.5)

    def test_none_stdout(self):
        from sciona.principal.evaluator import (
            _FAILURE_PENALTY,
            _parse_precision_loss_from_stdout,
        )

        assert _parse_precision_loss_from_stdout(None) == _FAILURE_PENALTY

    def test_no_mse_key(self):
        from sciona.principal.evaluator import (
            _FAILURE_PENALTY,
            _parse_precision_loss_from_stdout,
        )

        assert _parse_precision_loss_from_stdout(b'{"foo": 1}') == _FAILURE_PENALTY

    def test_last_valid_line(self):
        from sciona.principal.evaluator import _parse_precision_loss_from_stdout

        stdout = b'log line 1\nlog line 2\n{"mse": 0.123}\n'
        assert _parse_precision_loss_from_stdout(stdout) == pytest.approx(0.123)

    def test_prefers_loss_then_rmse(self):
        from sciona.principal.evaluator import _parse_precision_loss_from_stdout

        assert _parse_precision_loss_from_stdout(b'{"rmse": 1.2}') == pytest.approx(1.2)
        assert _parse_precision_loss_from_stdout(b'{"mse": 0.5, "loss": 2.5}') == pytest.approx(2.5)


class TestMetricSelection:
    def test_uncertainty_alias_maps_to_precision(self):
        from sciona.principal.metric_selection import resolve_optimization_objective

        metric, eval_spec, label = resolve_optimization_objective("uncertainty")
        assert metric == OptimizationMetric.PRECISION
        assert eval_spec is None
        assert label == "uncertainty"

    def test_rmse_alias_overrides_eval_spec_loss(self):
        from sciona.principal.metric_selection import resolve_optimization_objective

        metric, eval_spec, label = resolve_optimization_objective(
            "rmse",
            {"loss": "mse", "prediction": {"value_output": 0}, "reference": {"value_source": "x"}},
        )
        assert metric == OptimizationMetric.PRECISION
        assert eval_spec["loss"] == "rmse"
        assert label == "rmse"

    def test_rmse_requires_eval_spec(self):
        from sciona.principal.metric_selection import resolve_optimization_objective

        with pytest.raises(ValueError, match="requires an evaluation spec"):
            resolve_optimization_objective("rmse")

    def test_structure_maps_to_structure_metric(self):
        from sciona.principal.metric_selection import resolve_optimization_objective

        metric, eval_spec, label = resolve_optimization_objective("structure")
        assert metric == OptimizationMetric.STRUCTURE
        assert eval_spec is None
        assert label == "structure"


class TestExecutionSandbox:
    def test_evaluate_supports_relative_artifact_paths(self, tmp_path: Path, monkeypatch):
        from sciona.principal.evaluator import ExecutionSandbox
        from sciona.synthesizer.models import ExportBundle

        artifact = tmp_path / "artifact.py"
        artifact.write_text(
            "\n".join(
                [
                    "from pathlib import Path",
                    "import json",
                    "Path('trace.jsonl').write_text("
                    "json.dumps({'node_id': 'n1', 'execution_time_ms': 1.5, 'peak_memory_bytes': 10}) + '\\n'"
                    ")",
                    "print(json.dumps({'mse': 0.25}))",
                ]
            )
        )
        dataset = tmp_path / "dataset.json"
        dataset.write_text("{}")

        monkeypatch.chdir(tmp_path)
        bundle = ExportBundle(
            target="python-pkg",
            output_dir=Path("."),
            source_path=Path("artifact.py"),
            compiled_artifact=Path("artifact.py"),
        )

        result = asyncio.run(
            ExecutionSandbox(timeout_s=5.0).evaluate(
                bundle, str(dataset), OptimizationMetric.PRECISION
            )
        )

        assert result.global_loss == pytest.approx(0.25)
        assert "n1" in result.node_telemetry

    def test_evaluate_adapter_prefers_python_runner(self, tmp_path: Path, monkeypatch):
        from sciona.principal.evaluator import ExecutionSandbox
        from sciona.synthesizer.models import ExportBundle

        runner = tmp_path / "runner.py"
        runner.write_text("print('ok')\n")
        (tmp_path / "sciona.yml").write_text("name: test\n")
        trace = tmp_path / "trace.jsonl"
        bundle = ExportBundle(
            target="python-pkg",
            output_dir=tmp_path,
            source_path=runner,
            compiled_artifact=runner,
            executable_artifact=runner,
        )

        class DummyProc:
            returncode = 0

            async def communicate(self):
                trace.write_text(
                    '{"node_id":"leaf","execution_time_ms":1.0,"peak_memory_bytes":2}\n'
                )
                return (b'{"mse": 0.5}\n', b"")

        calls: list[list[str]] = []

        async def fake_exec(*cmd, **kwargs):
            calls.append(list(cmd))
            return DummyProc()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

        result = asyncio.run(
            ExecutionSandbox(timeout_s=5.0).evaluate_adapter(
                bundle,
                str(tmp_path / "sciona.yml"),
                OptimizationMetric.PRECISION,
                varset={"tracker": "full"},
            )
        )

        assert result.global_loss == pytest.approx(0.5)
        assert calls
        assert "--dataset-root" in calls[0]
        assert str(tmp_path) in calls[0]
        assert "--dataset-var" in calls[0]

    def test_evaluate_adapter_forwards_eval_spec(self, tmp_path: Path, monkeypatch):
        from sciona.principal.evaluator import ExecutionSandbox
        from sciona.synthesizer.models import ExportBundle

        runner = tmp_path / "runner.py"
        runner.write_text("print('ok')\n")
        (tmp_path / "sciona.yml").write_text("name: test\n")
        trace = tmp_path / "trace.jsonl"
        bundle = ExportBundle(
            target="python-pkg",
            output_dir=tmp_path,
            source_path=runner,
            compiled_artifact=runner,
            executable_artifact=runner,
        )

        class DummyProc:
            returncode = 0

            async def communicate(self):
                trace.write_text(
                    '{"node_id":"leaf","execution_time_ms":1.0,"peak_memory_bytes":2}\n'
                )
                return (b'{"loss": 1.5}\n', b"")

        calls: list[list[str]] = []

        async def fake_exec(*cmd, **kwargs):
            calls.append(list(cmd))
            return DummyProc()

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

        result = asyncio.run(
            ExecutionSandbox(timeout_s=5.0).evaluate_adapter(
                bundle,
                str(tmp_path / "sciona.yml"),
                OptimizationMetric.PRECISION,
                evaluation_spec='{"loss":"rmse"}',
            )
        )

        assert result.global_loss == pytest.approx(1.5)
        assert "--eval-spec" in calls[0]


# ===================================================================
# Tests for sciona.principal.backprop
# ===================================================================


class TestCreditAssigner:
    def setup_method(self):
        from sciona.principal.backprop import CreditAssigner

        self.assigner = CreditAssigner()
        self.cdg = _make_cdg(("a", "NodeA"), ("b", "NodeB"))

    def test_latency_gradients(self):
        bench = _make_benchmark(("a", 80.0, 100, 0.0), ("b", 20.0, 100, 0.0))
        sim = GhostSimReport()

        grads = self.assigner.compute_gradients(
            self.cdg,
            bench,
            sim,
            OptimizationMetric.LATENCY,
        )
        assert len(grads) == 2
        # Sorted descending
        assert grads[0].node_id == "a"
        assert grads[0].gradient_score == pytest.approx(80.0)
        assert grads[1].gradient_score == pytest.approx(20.0)
        assert grads[0].metric_type == OptimizationMetric.LATENCY

    def test_flop_count_uses_latency(self):
        bench = _make_benchmark(("a", 60.0, 0, 0.0), ("b", 40.0, 0, 0.0))
        sim = GhostSimReport()

        grads = self.assigner.compute_gradients(
            self.cdg,
            bench,
            sim,
            OptimizationMetric.FLOP_COUNT,
        )
        assert len(grads) == 2
        assert grads[0].metric_type == OptimizationMetric.FLOP_COUNT

    def test_memory_gradients(self):
        bench = _make_benchmark(("a", 1.0, 300, 0.0), ("b", 1.0, 700, 0.0))
        sim = GhostSimReport()

        grads = self.assigner.compute_gradients(
            self.cdg,
            bench,
            sim,
            OptimizationMetric.MEMORY,
        )
        assert grads[0].node_id == "b"
        assert grads[0].gradient_score == pytest.approx(70.0)
        assert "70.0%" in grads[0].bottleneck_reason

    def test_precision_from_ghost_sim(self):
        bench = _make_benchmark(("a", 1.0, 100, 0.0), ("b", 1.0, 100, 0.0))
        sim = GhostSimReport(precision_gradients={"a": 3.0, "b": 1.0})

        grads = self.assigner.compute_gradients(
            self.cdg,
            bench,
            sim,
            OptimizationMetric.PRECISION,
        )
        assert grads[0].node_id == "a"
        assert grads[0].gradient_score == pytest.approx(75.0)

    def test_precision_fallback_to_error_expansion(self):
        bench = _make_benchmark(("a", 1.0, 100, 5.0), ("b", 1.0, 100, 15.0))
        sim = GhostSimReport()  # no precision_gradients

        grads = self.assigner.compute_gradients(
            self.cdg,
            bench,
            sim,
            OptimizationMetric.PRECISION,
        )
        assert grads[0].node_id == "b"
        assert grads[0].gradient_score == pytest.approx(75.0)

    def test_structure_gradients_follow_ghost_risk(self):
        bench = BenchmarkResult(global_loss=0.0)
        sim = GhostSimReport(
            ran=True,
            passed=False,
            skipped_nodes=["NodeB"],
            error_node="NodeA",
        )

        grads = self.assigner.compute_gradients(
            self.cdg,
            bench,
            sim,
            OptimizationMetric.STRUCTURE,
        )
        assert grads[0].node_id == "a"
        assert grads[0].metric_type == OptimizationMetric.STRUCTURE
        assert "structural risk" in grads[0].bottleneck_reason

    def test_empty_telemetry_returns_empty(self):
        bench = BenchmarkResult(global_loss=0.0)
        sim = GhostSimReport()

        grads = self.assigner.compute_gradients(
            self.cdg,
            bench,
            sim,
            OptimizationMetric.LATENCY,
        )
        assert grads == []

    def test_bottleneck_reason_includes_name(self):
        bench = _make_benchmark(("a", 100.0, 100, 0.0))
        sim = GhostSimReport()

        grads = self.assigner.compute_gradients(
            self.cdg,
            bench,
            sim,
            OptimizationMetric.LATENCY,
        )
        assert "NodeA" in grads[0].bottleneck_reason

    def test_non_atomic_nodes_excluded(self):
        """Non-ATOMIC nodes should not get gradients."""
        cdg = CDGExport(
            nodes=[
                AlgorithmicNode(
                    node_id="root",
                    name="Root",
                    description="parent",
                    concept_type=ConceptType.ARITHMETIC,
                    status=NodeStatus.DECOMPOSED,
                ),
                AlgorithmicNode(
                    node_id="leaf",
                    name="Leaf",
                    description="child",
                    concept_type=ConceptType.ARITHMETIC,
                    status=NodeStatus.ATOMIC,
                ),
            ],
            edges=[],
        )
        bench = _make_benchmark(("root", 50.0, 100, 0.0), ("leaf", 50.0, 100, 0.0))
        sim = GhostSimReport()

        grads = self.assigner.compute_gradients(
            cdg,
            bench,
            sim,
            OptimizationMetric.LATENCY,
        )
        assert len(grads) == 1
        assert grads[0].node_id == "leaf"


# ===================================================================
# Tests for sciona.principal.hpo
# ===================================================================


class TestOptunaManager:
    def test_creation(self):
        from sciona.principal.hpo import OptunaManager

        mgr = OptunaManager(study_name="test")
        assert mgr.study is not None
        assert mgr.study.study_name == "test"

    def test_check_early_prune_clean(self):
        from sciona.principal.hpo import OptunaManager

        report = GhostSimReport(ran=True, passed=True, precision_gradients={"a": 1.0})
        # Should not raise
        OptunaManager.check_early_prune(report)

    def test_check_early_prune_not_ran(self):
        from sciona.principal.hpo import OptunaManager

        report = GhostSimReport(ran=False, passed=False)
        # ran=False means we don't prune based on failure
        OptunaManager.check_early_prune(report)

    def test_check_early_prune_failed_sim(self):
        from sciona.principal.hpo import OptunaManager, TrialPrunedEarly

        report = GhostSimReport(ran=True, passed=False, error="mismatch")
        with pytest.raises(TrialPrunedEarly, match="mismatch"):
            OptunaManager.check_early_prune(report)

    def test_check_early_prune_inf_gradient(self):
        from sciona.principal.hpo import OptunaManager, TrialPrunedEarly

        report = GhostSimReport(
            ran=True,
            passed=True,
            precision_gradients={"a": float("inf")},
        )
        with pytest.raises(TrialPrunedEarly, match="Infinite/NaN"):
            OptunaManager.check_early_prune(report)

    def test_check_early_prune_nan_gradient(self):
        from sciona.principal.hpo import OptunaManager, TrialPrunedEarly

        report = GhostSimReport(
            ran=True,
            passed=True,
            precision_gradients={"a": float("nan")},
        )
        with pytest.raises(TrialPrunedEarly):
            OptunaManager.check_early_prune(report)

    def test_param_importances_too_few_trials(self):
        from sciona.principal.hpo import OptunaManager

        mgr = OptunaManager(study_name="test_importance")
        result = mgr.param_importances()
        assert result == {}


# ===================================================================
# Tests for sciona.principal.graph (routing + state + build)
# ===================================================================


class TestPrincipalState:
    def test_defaults(self):
        from sciona.principal.graph import PrincipalState

        state = PrincipalState()
        assert state.goal == ""
        assert state.metric == OptimizationMetric.LATENCY
        assert state.max_trials == 50
        assert state.current_trial == 0
        assert state.best_loss == float("inf")
        assert state.done is False
        assert state.trial_history == []
        assert state.cdg is None
        assert state.export_bundle is None
        assert state.benchmark is None


class TestRouteAfterGradients:
    def test_done(self):
        from sciona.principal.graph import PrincipalState, route_after_gradients

        state = PrincipalState(done=True)
        assert route_after_gradients(state) == "end"

    def test_max_trials_reached(self):
        from sciona.principal.graph import PrincipalState, route_after_gradients

        state = PrincipalState(
            current_trial=50,
            max_trials=50,
            trial_history=[{"trial": i} for i in range(50)],
        )
        assert route_after_gradients(state) == "end"

    def test_non_pruned_error(self):
        from sciona.principal.graph import PrincipalState, route_after_gradients

        state = PrincipalState(error="fatal crash")
        assert route_after_gradients(state) == "end"

    def test_pruned_error_continues(self):
        from sciona.principal.graph import PrincipalState, route_after_gradients

        state = PrincipalState(error="Trial pruned early")
        assert route_after_gradients(state) == "select_proposal"

    def test_normal_continues(self):
        from sciona.principal.graph import PrincipalState, route_after_gradients

        state = PrincipalState()
        assert route_after_gradients(state) == "select_proposal"


class TestRouteAfterExpansion:
    def test_done(self):
        from sciona.principal.graph import PrincipalState, route_after_expansion

        state = PrincipalState(done=True)
        assert route_after_expansion(state) == "end"

    def test_max_trials_reached(self):
        from sciona.principal.graph import PrincipalState, route_after_expansion

        state = PrincipalState(max_trials=1, trial_history=[{"trial": 1}])
        assert route_after_expansion(state) == "end"

    def test_expansion_loops_to_param_search(self):
        from sciona.principal.graph import PrincipalState, route_after_expansion

        state = PrincipalState(expansion_applied=True)
        assert route_after_expansion(state) == "suggest_params"

    def test_no_expansion_continues_to_gradients(self):
        from sciona.principal.graph import PrincipalState, route_after_expansion

        state = PrincipalState(expansion_applied=False)
        assert route_after_expansion(state) == "gradients"


class TestRouteAfterUpdate:
    def test_done(self):
        from sciona.principal.graph import PrincipalState, route_after_update

        state = PrincipalState(done=True)
        assert route_after_update(state) == "end"

    def test_max_trials(self):
        from sciona.principal.graph import PrincipalState, route_after_update

        state = PrincipalState(
            current_trial=10,
            max_trials=10,
            trial_history=[{"trial": i} for i in range(10)],
        )
        assert route_after_update(state) == "end"

    def test_continues(self):
        from sciona.principal.graph import PrincipalState, route_after_update

        state = PrincipalState(current_trial=5, max_trials=10)
        assert route_after_update(state) == "suggest_params"


class TestRouteAfterForward:
    def test_done(self):
        from sciona.principal.graph import PrincipalState, route_after_forward

        state = PrincipalState(done=True)
        assert route_after_forward(state) == "end"

    def test_error_skips_to_time_travel(self):
        from sciona.principal.graph import PrincipalState, route_after_forward

        state = PrincipalState(error="pruned early")
        assert route_after_forward(state) == "time_travel"

    def test_normal_evaluates(self):
        from sciona.principal.graph import PrincipalState, route_after_forward

        state = PrincipalState()
        assert route_after_forward(state) == "evaluate"


class TestBuildPrincipalGraph:
    def test_graph_compiles(self):
        from sciona.principal.graph import build_principal_graph

        graph = build_principal_graph()
        assert graph is not None
        # Verify all expected nodes are present
        assert "seed" in graph.nodes
        assert "forward" in graph.nodes
        assert "evaluate" in graph.nodes
        assert "expand" in graph.nodes
        assert "gradients" in graph.nodes
        assert "time_travel" in graph.nodes
