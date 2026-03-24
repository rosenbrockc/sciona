"""Tests for the Neural Network expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.neural_network import NeuralNetworkExpansionRuleSet
from sciona.expansion_atoms.runtime_neural_network import (
    detect_gradient_explosion, analyze_activation_statistics,
    monitor_loss_convergence, check_weight_distribution,
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

def _neural_network_cdg():
    return _cdg(
        [_node("src", "Source"),
         _node("fwd", "Forward Pass", ConceptType.NEURAL_NETWORK),
         _node("loss", "Loss Computation", ConceptType.NEURAL_NETWORK),
         _node("bwd", "Backward Pass", ConceptType.NEURAL_NETWORK),
         _node("upd", "Parameter Update", ConceptType.NEURAL_NETWORK),
         _node("out", "Output")],
        [_edge("src", "fwd"), _edge("fwd", "loss"), _edge("loss", "bwd"),
         _edge("bwd", "upd"), _edge("upd", "out")],
    )


class TestDetectGradientExplosion:
    def test_healthy(self):
        grads = np.ones((10, 5))
        norm, exploding = detect_gradient_explosion(grads)
        assert not exploding
        assert norm > 0

    def test_exploding(self):
        grads = np.ones((10, 5)) * 1000
        norm, exploding = detect_gradient_explosion(grads)
        assert exploding
        assert norm > 100.0

    def test_empty(self):
        norm, exploding = detect_gradient_explosion(np.array([]))
        assert not exploding


class TestAnalyzeActivationStatistics:
    def test_healthy(self):
        activations = np.ones((100, 50))
        frac, dead = analyze_activation_statistics(activations)
        assert not dead
        assert frac == 0.0

    def test_dead_neurons(self):
        activations = np.zeros((100, 50))
        frac, dead = analyze_activation_statistics(activations)
        assert dead
        assert frac == 1.0

    def test_empty(self):
        frac, dead = analyze_activation_statistics(np.array([]))
        assert not dead


class TestMonitorLossConvergence:
    def test_converging(self):
        history = np.array([10.0, 5.0, 2.5, 1.25])
        ratio, plateaued = monitor_loss_convergence(history)
        assert not plateaued

    def test_plateaued(self):
        history = np.array([1.0, 1.0, 1.0])
        ratio, plateaued = monitor_loss_convergence(history)
        assert plateaued

    def test_short(self):
        ratio, plateaued = monitor_loss_convergence(np.array([1.0]))
        assert not plateaued


class TestCheckWeightDistribution:
    def test_balanced(self):
        weights = np.ones((5, 10))
        ratio, balanced = check_weight_distribution(weights)
        assert balanced
        assert ratio == pytest.approx(1.0)

    def test_imbalanced(self):
        weights = np.array([[1.0] * 10, [1000.0] * 10])
        ratio, balanced = check_weight_distribution(weights)
        assert not balanced
        assert ratio > 100.0

    def test_empty(self):
        ratio, balanced = check_weight_distribution(np.array([]))
        assert balanced


class TestNeuralNetworkRules:
    def _get_rules(self):
        return {r.name: r for r in NeuralNetworkExpansionRuleSet().rules()}

    def test_gradient_explosion_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_gradient_explosion_detection_after_backward"], _neural_network_cdg())
        assert not result.is_failure
        assert "detect_gradient_explosion" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_activation_statistics_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_activation_statistics_after_forward"], _neural_network_cdg())
        assert not result.is_failure

    def test_loss_convergence_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_loss_convergence_monitoring_after_loss"], _neural_network_cdg())
        assert not result.is_failure

    def test_weight_distribution_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_weight_distribution_check_after_update"], _neural_network_cdg())
        assert not result.is_failure


class TestNeuralNetworkDiagnostics:
    def test_diagnose_gradient_explosion(self):
        diags = NeuralNetworkExpansionRuleSet().diagnose(_neural_network_cdg(), ExpansionContext(intermediates={"gradient_max_norm": 500.0}))
        assert "insert_gradient_explosion_detection_after_backward" in {d.rule_name for d in diags}

    def test_healthy_gradient_no_trigger(self):
        diags = NeuralNetworkExpansionRuleSet().diagnose(_neural_network_cdg(), ExpansionContext(intermediates={"gradient_max_norm": 1.0}))
        assert not [d for d in diags if d.rule_name == "insert_gradient_explosion_detection_after_backward"]

    def test_diagnose_activation_statistics(self):
        diags = NeuralNetworkExpansionRuleSet().diagnose(_neural_network_cdg(), ExpansionContext(intermediates={"dead_neuron_fraction": 0.8}))
        assert "insert_activation_statistics_after_forward" in {d.rule_name for d in diags}

    def test_diagnose_loss_convergence(self):
        diags = NeuralNetworkExpansionRuleSet().diagnose(_neural_network_cdg(), ExpansionContext(intermediates={"loss_plateau_ratio": 1e-10}))
        assert "insert_loss_convergence_monitoring_after_loss" in {d.rule_name for d in diags}

    def test_diagnose_weight_distribution(self):
        diags = NeuralNetworkExpansionRuleSet().diagnose(_neural_network_cdg(), ExpansionContext(intermediates={"weight_norm_ratio": 500.0}))
        assert "insert_weight_distribution_check_after_update" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert NeuralNetworkExpansionRuleSet().diagnose(_neural_network_cdg(), ExpansionContext()) == []


class TestNeuralNetworkIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([NeuralNetworkExpansionRuleSet()]).expand(
            _neural_network_cdg(), ExpansionContext(intermediates={"gradient_max_norm": 500.0, "dead_neuron_fraction": 0.8}))
        assert result.expanded
