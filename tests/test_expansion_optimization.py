"""Tests for the Continuous Optimization expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.optimization import OptimizationExpansionRuleSet
from sciona.expansion_atoms.runtime_optimization import (
    detect_vanishing_gradient, analyze_loss_landscape,
    check_constraint_violation, monitor_convergence_rate,
)


def _node(nid, name, concept=ConceptType.CUSTOM, primitive=None):
    return AlgorithmicNode(
        node_id=nid, name=name, description=name, concept_type=concept,
        status=NodeStatus.ATOMIC, matched_primitive=primitive,
        inputs=[IOSpec(name="in", type_desc="ndarray")],
        outputs=[IOSpec(name="out", type_desc="ndarray")],
        type_signature=f"{name} -> r",
    )

def _edge(src, tgt):
    return DependencyEdge(source_id=src, target_id=tgt, output_name="out", input_name="in", source_type="ndarray", target_type="ndarray")

def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})

def _optimization_cdg():
    return _cdg(
        [_node("src", "Source"),
         _node("init", "Initialize", ConceptType.OPTIMIZATION),
         _node("grad", "Compute Gradient", ConceptType.OPTIMIZATION),
         _node("upd", "Update Parameters", ConceptType.OPTIMIZATION),
         _node("conv", "Check Convergence", ConceptType.OPTIMIZATION),
         _node("out", "Output")],
        [_edge("src", "init"), _edge("init", "grad"), _edge("grad", "upd"),
         _edge("upd", "conv"), _edge("conv", "out")],
    )


class TestDetectVanishingGradient:
    def test_healthy(self):
        grads = np.ones((10, 5))
        mn, vanishing = detect_vanishing_gradient(grads)
        assert not vanishing
        assert mn > 0

    def test_vanishing(self):
        grads = np.zeros((10, 5))
        mn, vanishing = detect_vanishing_gradient(grads)
        assert vanishing

    def test_empty(self):
        mn, vanishing = detect_vanishing_gradient(np.array([]))
        assert not vanishing


class TestAnalyzeLossLandscape:
    def test_well_conditioned(self):
        eigs = np.array([10.0, 5.0, 1.0])
        cond, ill = analyze_loss_landscape(eigs)
        assert not ill
        assert cond == pytest.approx(10.0)

    def test_ill_conditioned(self):
        eigs = np.array([1e12, 0.01])
        cond, ill = analyze_loss_landscape(eigs)
        assert ill

    def test_empty(self):
        cond, ill = analyze_loss_landscape(np.array([]))
        assert not ill


class TestCheckConstraintViolation:
    def test_feasible(self):
        values = np.array([0.5, 1.5])
        bounds = np.array([[0.0, 1.0], [1.0, 2.0]])
        viol, feasible = check_constraint_violation(values, bounds)
        assert feasible
        assert viol == 0.0

    def test_infeasible(self):
        values = np.array([1.5, 2.5])
        bounds = np.array([[0.0, 1.0], [0.0, 2.0]])
        viol, feasible = check_constraint_violation(values, bounds)
        assert not feasible
        assert viol > 0

    def test_empty(self):
        viol, feasible = check_constraint_violation(np.array([]), np.array([]))
        assert feasible


class TestMonitorConvergenceRate:
    def test_fast_convergence(self):
        # Geometrically decreasing: 100, 10, 1, 0.1
        history = np.array([100.0, 10.0, 1.0, 0.1])
        order, converging = monitor_convergence_rate(history)
        assert converging

    def test_slow_convergence(self):
        # Nearly flat: differences barely decrease
        history = np.array([10.0, 9.5, 9.1, 8.8, 8.6, 8.5])
        order, converging = monitor_convergence_rate(history)
        assert not converging

    def test_short_history(self):
        order, converging = monitor_convergence_rate(np.array([1.0, 2.0]))
        assert converging


class TestOptimizationRules:
    def _get_rules(self):
        return {r.name: r for r in OptimizationExpansionRuleSet().rules()}

    def test_vanishing_gradient_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_vanishing_gradient_detection_after_compute_gradient"], _optimization_cdg())
        assert not result.is_failure
        assert "detect_vanishing_gradient" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_loss_landscape_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_loss_landscape_analysis_before_update"], _optimization_cdg())
        assert not result.is_failure

    def test_constraint_violation_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_constraint_violation_check_after_update"], _optimization_cdg())
        assert not result.is_failure

    def test_convergence_rate_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_convergence_rate_monitoring_after_check"], _optimization_cdg())
        assert not result.is_failure


class TestOptimizationDiagnostics:
    def test_diagnose_vanishing_gradient(self):
        diags = OptimizationExpansionRuleSet().diagnose(_optimization_cdg(), ExpansionContext(intermediates={"gradient_min_norm": 1e-20}))
        assert "insert_vanishing_gradient_detection_after_compute_gradient" in {d.rule_name for d in diags}

    def test_healthy_gradient_no_trigger(self):
        diags = OptimizationExpansionRuleSet().diagnose(_optimization_cdg(), ExpansionContext(intermediates={"gradient_min_norm": 1.0}))
        assert not [d for d in diags if d.rule_name == "insert_vanishing_gradient_detection_after_compute_gradient"]

    def test_diagnose_loss_landscape(self):
        diags = OptimizationExpansionRuleSet().diagnose(_optimization_cdg(), ExpansionContext(intermediates={"hessian_condition_number": 1e12}))
        assert "insert_loss_landscape_analysis_before_update" in {d.rule_name for d in diags}

    def test_diagnose_constraint_violation(self):
        diags = OptimizationExpansionRuleSet().diagnose(_optimization_cdg(), ExpansionContext(intermediates={"max_constraint_violation": 0.5}))
        assert "insert_constraint_violation_check_after_update" in {d.rule_name for d in diags}

    def test_diagnose_convergence_rate(self):
        diags = OptimizationExpansionRuleSet().diagnose(_optimization_cdg(), ExpansionContext(intermediates={"convergence_order": 0.1}))
        assert "insert_convergence_rate_monitoring_after_check" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert OptimizationExpansionRuleSet().diagnose(_optimization_cdg(), ExpansionContext()) == []


class TestOptimizationIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([OptimizationExpansionRuleSet()]).expand(
            _optimization_cdg(), ExpansionContext(intermediates={"gradient_min_norm": 1e-20, "max_constraint_violation": 0.5}))
        assert result.expanded
