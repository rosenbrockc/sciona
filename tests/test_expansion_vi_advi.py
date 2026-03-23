"""Tests for the VI/ADVI expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.vi_advi import VIADVIExpansionRuleSet
from sciona.expansion_atoms.runtime_vi_advi import (
    monitor_elbo_convergence, analyze_gradient_variance,
    detect_posterior_collapse, check_step_size_stability,
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

def _vi_advi_cdg():
    return _cdg(
        [_node("src", "Source"), _node("sa", "Shape Alloc", ConceptType.VI_ELBO),
         _node("rp", "Reparameterization", ConceptType.VI_ELBO),
         _node("ee", "ELBO Eval", ConceptType.PROBABILISTIC_ORACLE),
         _node("opt", "L-BFGS Optimizer", ConceptType.VI_ELBO),
         _node("out", "Output")],
        [_edge("src", "sa"), _edge("sa", "rp"), _edge("rp", "ee"), _edge("ee", "opt"), _edge("opt", "out")],
    )


class TestMonitorElboConvergence:
    def test_converged(self):
        # Constant plateau — both windows see the same values
        history = np.full(30, 195.1)
        rel, converged = monitor_elbo_convergence(history)
        assert converged

    def test_not_converged(self):
        history = np.array([100, 150, 200, 250, 300])
        rel, converged = monitor_elbo_convergence(history)
        assert not converged

    def test_short_history(self):
        rel, converged = monitor_elbo_convergence(np.array([100.0]))
        assert not converged


class TestAnalyzeGradientVariance:
    def test_low_variance(self):
        rng = np.random.RandomState(42)
        grads = rng.randn(100, 5) * 0.01 + 1.0
        cv, low = analyze_gradient_variance(grads)
        assert low

    def test_high_variance(self):
        rng = np.random.RandomState(42)
        grads = rng.randn(100, 5) * 10.0
        cv, low = analyze_gradient_variance(grads)
        assert not low

    def test_single_sample(self):
        cv, low = analyze_gradient_variance(np.array([[1.0, 2.0]]))
        assert low


class TestDetectPosteriorCollapse:
    def test_no_collapse(self):
        kl = np.array([1.0, 0.5, 2.0, 1.5])
        n, frac = detect_posterior_collapse(kl)
        assert n == 0
        assert frac == 0.0

    def test_partial_collapse(self):
        kl = np.array([1.0, 0.001, 0.005, 2.0])
        n, frac = detect_posterior_collapse(kl)
        assert n == 2
        assert frac == 0.5

    def test_empty(self):
        n, frac = detect_posterior_collapse(np.array([]))
        assert n == 0


class TestCheckStepSizeStability:
    def test_stable(self):
        steps = np.ones(50) * 0.01
        cv, stable = check_step_size_stability(steps)
        assert stable

    def test_unstable(self):
        rng = np.random.RandomState(42)
        steps = rng.exponential(1.0, 50)
        cv, stable = check_step_size_stability(steps)
        assert not stable

    def test_single(self):
        cv, stable = check_step_size_stability(np.array([0.01]))
        assert stable


class TestVIADVIRules:
    def _get_rules(self):
        return {r.name: r for r in VIADVIExpansionRuleSet().rules()}

    def test_elbo_convergence_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_elbo_convergence_monitoring_after_elbo_eval"], _vi_advi_cdg())
        assert not result.is_failure
        assert "monitor_elbo_convergence" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_gradient_variance_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_gradient_variance_analysis_after_reparameterization"], _vi_advi_cdg())
        assert not result.is_failure

    def test_posterior_collapse_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_posterior_collapse_detection_after_shape_alloc"], _vi_advi_cdg())
        assert not result.is_failure

    def test_step_size_stability_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_step_size_stability_check_after_optimizer"], _vi_advi_cdg())
        assert not result.is_failure


class TestVIADVIDiagnostics:
    def test_diagnose_not_converged(self):
        diags = VIADVIExpansionRuleSet().diagnose(_vi_advi_cdg(), ExpansionContext(intermediates={"elbo_relative_improvement": 0.05}))
        assert "insert_elbo_convergence_monitoring_after_elbo_eval" in {d.rule_name for d in diags}

    def test_converged_no_trigger(self):
        diags = VIADVIExpansionRuleSet().diagnose(_vi_advi_cdg(), ExpansionContext(intermediates={"elbo_relative_improvement": 0.001}))
        assert not [d for d in diags if d.rule_name == "insert_elbo_convergence_monitoring_after_elbo_eval"]

    def test_diagnose_gradient_variance(self):
        diags = VIADVIExpansionRuleSet().diagnose(_vi_advi_cdg(), ExpansionContext(intermediates={"gradient_mean_cv": 3.0}))
        assert "insert_gradient_variance_analysis_after_reparameterization" in {d.rule_name for d in diags}

    def test_diagnose_posterior_collapse(self):
        diags = VIADVIExpansionRuleSet().diagnose(_vi_advi_cdg(), ExpansionContext(intermediates={"posterior_collapse_fraction": 0.5}))
        assert "insert_posterior_collapse_detection_after_shape_alloc" in {d.rule_name for d in diags}

    def test_diagnose_step_size(self):
        diags = VIADVIExpansionRuleSet().diagnose(_vi_advi_cdg(), ExpansionContext(intermediates={"step_size_cv": 0.8}))
        assert "insert_step_size_stability_check_after_optimizer" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert VIADVIExpansionRuleSet().diagnose(_vi_advi_cdg(), ExpansionContext()) == []


class TestVIADVIIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([VIADVIExpansionRuleSet()]).expand(
            _vi_advi_cdg(), ExpansionContext(intermediates={"elbo_relative_improvement": 0.1, "posterior_collapse_fraction": 0.3}))
        assert result.expanded
