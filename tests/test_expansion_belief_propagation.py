"""Tests for the Belief Propagation expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.belief_propagation import BeliefPropagationExpansionRuleSet
from sciona.expansion_atoms.runtime_belief_propagation import (
    monitor_message_convergence, validate_belief_normalization,
    analyze_message_damping, detect_graph_cycles,
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

def _bp_cdg():
    return _cdg(
        [_node("src", "Source"),
         _node("v2f", "Variable to Factor", ConceptType.MESSAGE_PASSING),
         _node("f2v", "Factor to Variable", ConceptType.MESSAGE_PASSING),
         _node("mg", "Marginal Computation", ConceptType.MESSAGE_PASSING),
         _node("memo", "Memoization State", ConceptType.MESSAGE_PASSING),
         _node("out", "Output")],
        [_edge("src", "v2f"), _edge("v2f", "f2v"), _edge("f2v", "mg"),
         _edge("v2f", "memo"), _edge("f2v", "memo"),
         _edge("memo", "v2f"), _edge("mg", "out")],
    )


class TestMonitorMessageConvergence:
    def test_converged(self):
        deltas = np.array([1.0, 0.1, 0.01, 0.001, 1e-7])
        d, converged = monitor_message_convergence(deltas)
        assert converged

    def test_not_converged(self):
        deltas = np.array([1.0, 0.5, 0.3, 0.1])
        d, converged = monitor_message_convergence(deltas)
        assert not converged

    def test_empty(self):
        d, converged = monitor_message_convergence(np.array([]))
        assert converged


class TestValidateBeliefNormalization:
    def test_normalized(self):
        beliefs = np.array([[0.3, 0.7], [0.5, 0.5], [0.1, 0.9]])
        dev, normalized = validate_belief_normalization(beliefs)
        assert normalized

    def test_not_normalized(self):
        beliefs = np.array([[0.3, 0.8], [0.5, 0.5]])
        dev, normalized = validate_belief_normalization(beliefs)
        assert not normalized

    def test_empty(self):
        dev, normalized = validate_belief_normalization(np.array([]).reshape(0, 2))
        assert normalized


class TestAnalyzeMessageDamping:
    def test_no_oscillation(self):
        # Monotonically decreasing
        history = np.array([[10, 8], [8, 6], [6, 4], [4, 2], [2, 1]], dtype=float)
        score, needs = analyze_message_damping(history)
        assert not needs

    def test_oscillation(self):
        # Alternating
        history = np.array([[1, 1], [2, 2], [1, 1], [2, 2], [1, 1]], dtype=float)
        score, needs = analyze_message_damping(history)
        assert needs

    def test_short(self):
        score, needs = analyze_message_damping(np.array([[1.0], [2.0]]))
        assert not needs


class TestDetectGraphCycles:
    def test_tree(self):
        # Path graph: 0-1-2
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        extra, is_tree = detect_graph_cycles(A)
        assert is_tree
        assert extra == 0

    def test_cycle(self):
        # Triangle: 0-1-2-0
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        extra, is_tree = detect_graph_cycles(A)
        assert not is_tree
        assert extra == 1

    def test_empty(self):
        extra, is_tree = detect_graph_cycles(np.array([]).reshape(0, 0))
        assert is_tree


class TestBeliefPropagationRules:
    def _get_rules(self):
        return {r.name: r for r in BeliefPropagationExpansionRuleSet().rules()}

    def test_convergence_monitoring_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_message_convergence_monitoring_after_factor_to_var"], _bp_cdg())
        assert not result.is_failure
        assert "monitor_message_convergence" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_normalization_validation_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_belief_normalization_validation_after_marginal"], _bp_cdg())
        assert not result.is_failure

    def test_damping_analysis_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_message_damping_analysis_after_var_to_factor"], _bp_cdg())
        assert not result.is_failure

    def test_cycle_detection_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_cycle_detection_before_var_to_factor"], _bp_cdg())
        assert not result.is_failure


class TestBeliefPropagationDiagnostics:
    def test_diagnose_not_converged(self):
        diags = BeliefPropagationExpansionRuleSet().diagnose(_bp_cdg(), ExpansionContext(intermediates={"message_final_delta": 0.01}))
        assert "insert_message_convergence_monitoring_after_factor_to_var" in {d.rule_name for d in diags}

    def test_converged_no_trigger(self):
        diags = BeliefPropagationExpansionRuleSet().diagnose(_bp_cdg(), ExpansionContext(intermediates={"message_final_delta": 1e-8}))
        assert not [d for d in diags if d.rule_name == "insert_message_convergence_monitoring_after_factor_to_var"]

    def test_diagnose_normalization(self):
        diags = BeliefPropagationExpansionRuleSet().diagnose(_bp_cdg(), ExpansionContext(intermediates={"belief_max_deviation": 0.01}))
        assert "insert_belief_normalization_validation_after_marginal" in {d.rule_name for d in diags}

    def test_diagnose_oscillation(self):
        diags = BeliefPropagationExpansionRuleSet().diagnose(_bp_cdg(), ExpansionContext(intermediates={"oscillation_score": 0.3}))
        assert "insert_message_damping_analysis_after_var_to_factor" in {d.rule_name for d in diags}

    def test_diagnose_cycles(self):
        diags = BeliefPropagationExpansionRuleSet().diagnose(_bp_cdg(), ExpansionContext(intermediates={"n_extra_edges": 3}))
        assert "insert_cycle_detection_before_var_to_factor" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert BeliefPropagationExpansionRuleSet().diagnose(_bp_cdg(), ExpansionContext()) == []


class TestBeliefPropagationIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([BeliefPropagationExpansionRuleSet()]).expand(
            _bp_cdg(), ExpansionContext(intermediates={"message_final_delta": 0.1, "n_extra_edges": 2}))
        assert result.expanded
