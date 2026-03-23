"""Tests for the Kalman Filter expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.kalman_filter import KalmanFilterExpansionRuleSet
from sciona.expansion_atoms.runtime_kalman_filter import (
    check_innovation_consistency, validate_covariance_pd,
    analyze_kalman_gain_magnitude, check_state_smoothness,
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

def _kalman_cdg():
    return _cdg(
        [_node("src", "Source"),
         _node("ps", "Predict State", ConceptType.SEQUENTIAL_FILTER),
         _node("pc", "Predict Covariance", ConceptType.SEQUENTIAL_FILTER),
         _node("inn", "Innovation", ConceptType.SEQUENTIAL_FILTER),
         _node("kg", "Kalman Gain", ConceptType.CONJUGATE_UPDATE),
         _node("us", "Update State", ConceptType.CONJUGATE_UPDATE),
         _node("uc", "Update Covariance", ConceptType.CONJUGATE_UPDATE),
         _node("out", "Output")],
        [_edge("src", "ps"), _edge("src", "pc"), _edge("ps", "inn"),
         _edge("pc", "kg"), _edge("inn", "us"), _edge("kg", "us"),
         _edge("kg", "uc"), _edge("us", "out"), _edge("uc", "out")],
    )


class TestCheckInnovationConsistency:
    def test_consistent(self):
        rng = np.random.RandomState(42)
        S = np.eye(2)
        innovations = rng.multivariate_normal(np.zeros(2), S, 100)
        nis, consistent = check_innovation_consistency(innovations, S)
        assert consistent

    def test_inconsistent(self):
        # Innovations too large for covariance
        innovations = np.ones((50, 2)) * 100
        S = np.eye(2) * 0.01
        nis, consistent = check_innovation_consistency(innovations, S)
        assert not consistent

    def test_empty(self):
        nis, consistent = check_innovation_consistency(np.array([]).reshape(0, 2), np.eye(2))
        assert consistent


class TestValidateCovariancePD:
    def test_pd(self):
        P = np.eye(3) * 2.0
        min_eig, pd = validate_covariance_pd(P)
        assert pd
        assert min_eig > 0

    def test_not_pd(self):
        P = np.array([[1, 0], [0, -0.1]])
        min_eig, pd = validate_covariance_pd(P)
        assert not pd

    def test_empty(self):
        min_eig, pd = validate_covariance_pd(np.array([]).reshape(0, 0))
        assert pd


class TestAnalyzeKalmanGainMagnitude:
    def test_bounded(self):
        K = np.random.RandomState(42).randn(20, 3, 2) * 0.5
        norm, bounded = analyze_kalman_gain_magnitude(K)
        assert bounded

    def test_unbounded(self):
        K = np.ones((5, 3, 2)) * 200
        norm, bounded = analyze_kalman_gain_magnitude(K)
        assert not bounded

    def test_empty(self):
        norm, bounded = analyze_kalman_gain_magnitude(np.array([]))
        assert bounded


class TestCheckStateSmoothness:
    def test_smooth(self):
        x = np.linspace(0, 10, 100).reshape(-1, 1)
        n_jumps, frac = check_state_smoothness(x)
        assert n_jumps == 0

    def test_jumpy(self):
        # Smooth ramp with a big jump in the middle
        x = np.linspace(0, 10, 100).reshape(-1, 1)
        x[50] += 1000  # big jump
        n_jumps, frac = check_state_smoothness(x)
        assert n_jumps > 0

    def test_short(self):
        n_jumps, frac = check_state_smoothness(np.array([[1.0], [2.0]]))
        assert n_jumps == 0


class TestKalmanFilterRules:
    def _get_rules(self):
        return {r.name: r for r in KalmanFilterExpansionRuleSet().rules()}

    def test_innovation_consistency_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_innovation_consistency_check_after_innovation"], _kalman_cdg())
        assert not result.is_failure
        assert "check_innovation_consistency" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_covariance_pd_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_covariance_pd_validation_after_predict_covariance"], _kalman_cdg())
        assert not result.is_failure

    def test_gain_magnitude_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_gain_magnitude_analysis_after_kalman_gain"], _kalman_cdg())
        assert not result.is_failure

    def test_state_smoothness_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_state_smoothness_check_after_update_state"], _kalman_cdg())
        assert not result.is_failure


class TestKalmanFilterDiagnostics:
    def test_diagnose_inconsistent_innovation(self):
        diags = KalmanFilterExpansionRuleSet().diagnose(_kalman_cdg(), ExpansionContext(intermediates={"mean_nis": 10.0, "innovation_dim": 2}))
        assert "insert_innovation_consistency_check_after_innovation" in {d.rule_name for d in diags}

    def test_consistent_no_trigger(self):
        diags = KalmanFilterExpansionRuleSet().diagnose(_kalman_cdg(), ExpansionContext(intermediates={"mean_nis": 2.0, "innovation_dim": 2}))
        assert not [d for d in diags if d.rule_name == "insert_innovation_consistency_check_after_innovation"]

    def test_diagnose_not_pd(self):
        diags = KalmanFilterExpansionRuleSet().diagnose(_kalman_cdg(), ExpansionContext(intermediates={"min_covariance_eigenvalue": -0.01}))
        assert "insert_covariance_pd_validation_after_predict_covariance" in {d.rule_name for d in diags}

    def test_diagnose_large_gain(self):
        diags = KalmanFilterExpansionRuleSet().diagnose(_kalman_cdg(), ExpansionContext(intermediates={"max_kalman_gain_norm": 500.0}))
        assert "insert_gain_magnitude_analysis_after_kalman_gain" in {d.rule_name for d in diags}

    def test_diagnose_jumps(self):
        diags = KalmanFilterExpansionRuleSet().diagnose(_kalman_cdg(), ExpansionContext(intermediates={"state_jump_fraction": 0.1}))
        assert "insert_state_smoothness_check_after_update_state" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert KalmanFilterExpansionRuleSet().diagnose(_kalman_cdg(), ExpansionContext()) == []


class TestKalmanFilterIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([KalmanFilterExpansionRuleSet()]).expand(
            _kalman_cdg(), ExpansionContext(intermediates={"min_covariance_eigenvalue": -0.05, "max_kalman_gain_norm": 200.0}))
        assert result.expanded
