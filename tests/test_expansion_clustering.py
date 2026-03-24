"""Tests for the Clustering expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.clustering import ClusteringExpansionRuleSet
from sciona.expansion_atoms.runtime_clustering import (
    analyze_cluster_balance, monitor_assignment_stability,
    detect_empty_clusters, validate_separation,
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

def _clustering_cdg():
    return _cdg(
        [_node("src", "Source"),
         _node("init", "Initialize Centers", ConceptType.CLUSTERING),
         _node("assign", "Assign Points", ConceptType.CLUSTERING),
         _node("update", "Update Centers", ConceptType.CLUSTERING),
         _node("out", "Output")],
        [_edge("src", "init"), _edge("init", "assign"), _edge("assign", "update"),
         _edge("update", "out")],
    )


class TestAnalyzeClusterBalance:
    def test_balanced(self):
        sizes = np.array([100, 100, 100])
        ratio, balanced = analyze_cluster_balance(sizes)
        assert balanced
        assert ratio == pytest.approx(1.0)

    def test_imbalanced(self):
        sizes = np.array([1000, 10, 5])
        ratio, balanced = analyze_cluster_balance(sizes)
        assert not balanced
        assert ratio > 10.0

    def test_empty(self):
        ratio, balanced = analyze_cluster_balance(np.array([]))
        assert balanced


class TestMonitorAssignmentStability:
    def test_stable(self):
        prev = np.array([0, 1, 2, 0, 1])
        curr = np.array([0, 1, 2, 0, 1])
        frac, stable = monitor_assignment_stability(prev, curr)
        assert stable
        assert frac == 0.0

    def test_unstable(self):
        prev = np.array([0, 0, 0, 0, 0])
        curr = np.array([1, 1, 1, 1, 1])
        frac, stable = monitor_assignment_stability(prev, curr)
        assert not stable
        assert frac == 1.0

    def test_empty(self):
        frac, stable = monitor_assignment_stability(np.array([]), np.array([]))
        assert stable


class TestDetectEmptyClusters:
    def test_no_empty(self):
        sizes = np.array([10, 20, 30])
        n, has = detect_empty_clusters(sizes)
        assert not has
        assert n == 0

    def test_has_empty(self):
        sizes = np.array([10, 0, 30, 0])
        n, has = detect_empty_clusters(sizes)
        assert has
        assert n == 2

    def test_empty(self):
        n, has = detect_empty_clusters(np.array([]))
        assert not has


class TestValidateSeparation:
    def test_well_separated(self):
        inter = np.array([10.0, 12.0, 11.0])
        intra = np.array([2.0, 3.0, 2.5])
        ratio, sep = validate_separation(inter, intra)
        assert sep
        assert ratio > 1.0

    def test_poorly_separated(self):
        inter = np.array([1.0, 1.5])
        intra = np.array([5.0, 6.0])
        ratio, sep = validate_separation(inter, intra)
        assert not sep
        assert ratio < 1.0

    def test_empty(self):
        ratio, sep = validate_separation(np.array([]), np.array([]))
        assert sep


class TestClusteringRules:
    def _get_rules(self):
        return {r.name: r for r in ClusteringExpansionRuleSet().rules()}

    def test_cluster_balance_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_cluster_balance_analysis_after_assign"], _clustering_cdg())
        assert not result.is_failure
        assert "analyze_cluster_balance" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_assignment_stability_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_assignment_stability_after_update"], _clustering_cdg())
        assert not result.is_failure

    def test_empty_cluster_detection_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_empty_cluster_detection_after_assign"], _clustering_cdg())
        assert not result.is_failure

    def test_separation_validation_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_separation_validation_after_update"], _clustering_cdg())
        assert not result.is_failure


class TestClusteringDiagnostics:
    def test_diagnose_cluster_balance(self):
        diags = ClusteringExpansionRuleSet().diagnose(_clustering_cdg(), ExpansionContext(intermediates={"cluster_imbalance_ratio": 50.0}))
        assert "insert_cluster_balance_analysis_after_assign" in {d.rule_name for d in diags}

    def test_balanced_no_trigger(self):
        diags = ClusteringExpansionRuleSet().diagnose(_clustering_cdg(), ExpansionContext(intermediates={"cluster_imbalance_ratio": 2.0}))
        assert not [d for d in diags if d.rule_name == "insert_cluster_balance_analysis_after_assign"]

    def test_diagnose_assignment_stability(self):
        diags = ClusteringExpansionRuleSet().diagnose(_clustering_cdg(), ExpansionContext(intermediates={"assignment_change_fraction": 0.1}))
        assert "insert_assignment_stability_after_update" in {d.rule_name for d in diags}

    def test_diagnose_empty_clusters(self):
        diags = ClusteringExpansionRuleSet().diagnose(_clustering_cdg(), ExpansionContext(intermediates={"n_empty_clusters": 2}))
        assert "insert_empty_cluster_detection_after_assign" in {d.rule_name for d in diags}

    def test_diagnose_separation(self):
        diags = ClusteringExpansionRuleSet().diagnose(_clustering_cdg(), ExpansionContext(intermediates={"separation_ratio": 0.5}))
        assert "insert_separation_validation_after_update" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert ClusteringExpansionRuleSet().diagnose(_clustering_cdg(), ExpansionContext()) == []


class TestClusteringIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([ClusteringExpansionRuleSet()]).expand(
            _clustering_cdg(), ExpansionContext(intermediates={"cluster_imbalance_ratio": 50.0, "n_empty_clusters": 2}))
        assert result.expanded
