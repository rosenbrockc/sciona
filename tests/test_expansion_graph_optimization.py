"""Tests for the Graph Optimization expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.graph_optimization import (
    GraphOptimizationExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_graph_optimization import (
    analyze_graph_density,
    detect_distance_overflow,
    detect_negative_weights,
    monitor_relaxation_convergence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(nid, name, concept=ConceptType.CUSTOM, primitive=None):
    return AlgorithmicNode(
        node_id=nid,
        name=name,
        description=name,
        concept_type=concept,
        status=NodeStatus.ATOMIC,
        matched_primitive=primitive,
        inputs=[IOSpec(name="in", type_desc="ndarray")],
        outputs=[IOSpec(name="out", type_desc="ndarray")],
        type_signature=f"{name} -> r",
    )


def _edge(src, tgt):
    return DependencyEdge(
        source_id=src,
        target_id=tgt,
        output_name="out",
        input_name="in",
        source_type="ndarray",
        target_type="ndarray",
    )


def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})


def _graph_opt_cdg():
    """Build a minimal graph optimization CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("src", "Source", ConceptType.CUSTOM),
            _node("iw", "Init Weights", ConceptType.GRAPH_OPTIMIZATION),
            _node("re", "Relax Edges", ConceptType.GRAPH_OPTIMIZATION),
            _node("cn", "Check Negative Cycle", ConceptType.GRAPH_OPTIMIZATION),
            _node("ep", "Extract Path", ConceptType.GRAPH_OPTIMIZATION),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("src", "iw"),
            _edge("iw", "re"),
            _edge("re", "cn"),
            _edge("cn", "ep"),
            _edge("ep", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestDetectNegativeWeights:
    def test_no_negatives(self):
        weights = np.array([1.0, 2.5, 0.0, 3.0])
        n_neg, min_w = detect_negative_weights(weights)
        assert n_neg == 0
        assert min_w == 0.0

    def test_has_negatives(self):
        weights = np.array([1.0, -2.5, 3.0, -0.1])
        n_neg, min_w = detect_negative_weights(weights)
        assert n_neg == 2
        assert min_w == pytest.approx(-2.5)

    def test_all_negative(self):
        weights = np.array([-1.0, -2.0, -3.0])
        n_neg, min_w = detect_negative_weights(weights)
        assert n_neg == 3
        assert min_w == pytest.approx(-3.0)

    def test_empty(self):
        n_neg, min_w = detect_negative_weights(np.array([]))
        assert n_neg == 0
        assert min_w == 0.0


class TestMonitorRelaxationConvergence:
    def test_converges_early(self):
        snaps = np.array([
            [0.0, 5.0, float("inf")],
            [0.0, 3.0, 7.0],
            [0.0, 3.0, 7.0],  # converged
            [0.0, 3.0, 7.0],
        ])
        converged_at, has_conv = monitor_relaxation_convergence(snaps)
        assert has_conv
        assert converged_at == 2

    def test_no_convergence(self):
        snaps = np.array([
            [0.0, 5.0, 10.0],
            [0.0, 3.0, 8.0],
            [0.0, 2.0, 6.0],
        ])
        converged_at, has_conv = monitor_relaxation_convergence(snaps)
        assert not has_conv
        assert converged_at == -1

    def test_single_snapshot(self):
        snaps = np.array([[0.0, 1.0, 2.0]])
        converged_at, has_conv = monitor_relaxation_convergence(snaps)
        assert not has_conv

    def test_immediate_convergence(self):
        snaps = np.array([
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ])
        converged_at, has_conv = monitor_relaxation_convergence(snaps)
        assert has_conv
        assert converged_at == 1


class TestDetectDistanceOverflow:
    def test_no_overflow(self):
        dists = np.array([0.0, 10.0, 100.0, 1000.0])
        n_over, max_d = detect_distance_overflow(dists)
        assert n_over == 0
        assert max_d == 1000.0

    def test_overflow(self):
        dists = np.array([0.0, 1e16, 1e17])
        n_over, max_d = detect_distance_overflow(dists)
        assert n_over == 2
        assert max_d == pytest.approx(1e17)

    def test_custom_threshold(self):
        dists = np.array([100.0, 200.0])
        n_over, max_d = detect_distance_overflow(dists, overflow_threshold=150.0)
        assert n_over == 1

    def test_empty(self):
        n_over, max_d = detect_distance_overflow(np.array([]))
        assert n_over == 0
        assert max_d == 0.0

    def test_with_inf(self):
        dists = np.array([0.0, float("inf"), 5.0])
        n_over, max_d = detect_distance_overflow(dists)
        # inf is not finite, so only finite values checked
        assert max_d == 5.0


class TestAnalyzeGraphDensity:
    def test_sparse_graph(self):
        density, rec = analyze_graph_density(100, 50)
        assert density < 0.1
        assert rec == "sparse"

    def test_dense_graph(self):
        density, rec = analyze_graph_density(10, 60)
        # max_edges = 10*9 = 90, density = 60/90 ≈ 0.67
        assert density > 0.5
        assert rec == "dense"

    def test_moderate_graph(self):
        density, rec = analyze_graph_density(10, 20)
        # max_edges = 90, density ≈ 0.22
        assert 0.1 <= density <= 0.5
        assert rec == "moderate"

    def test_single_node(self):
        density, rec = analyze_graph_density(1, 0)
        assert density == 0.0
        assert rec == "sparse"

    def test_empty_graph(self):
        density, rec = analyze_graph_density(0, 0)
        assert density == 0.0
        assert rec == "sparse"


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestGraphOptimizationRules:
    def _get_rules(self):
        rs = GraphOptimizationExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_negative_weight_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_negative_weight_detection_before_relax"]
        rw = GraphRewriter()
        cdg = _graph_opt_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_negative_weights" in prims
        assert len(g.nodes) == 7  # 6 + 1

    def test_relaxation_convergence_applies(self):
        rules = self._get_rules()
        rule = rules["insert_relaxation_convergence_after_relax"]
        rw = GraphRewriter()
        cdg = _graph_opt_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "monitor_relaxation_convergence" in prims

    def test_distance_overflow_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_distance_overflow_detection_before_extract"]
        rw = GraphRewriter()
        cdg = _graph_opt_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_distance_overflow" in prims

    def test_graph_density_analysis_applies(self):
        rules = self._get_rules()
        rule = rules["insert_graph_density_analysis_before_relax"]
        rw = GraphRewriter()
        cdg = _graph_opt_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "analyze_graph_density" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestGraphOptimizationDiagnostics:
    def test_diagnose_negative_weights(self):
        rs = GraphOptimizationExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"min_edge_weight": -5.0}
        )
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_negative_weight_detection_before_relax" in names

    def test_positive_weights_no_trigger(self):
        rs = GraphOptimizationExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"min_edge_weight": 0.5}
        )
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ctx)
        neg_diags = [
            d for d in diags
            if d.rule_name == "insert_negative_weight_detection_before_relax"
        ]
        assert len(neg_diags) == 0

    def test_diagnose_early_convergence(self):
        rs = GraphOptimizationExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"relaxation_iterations": 5, "n_nodes": 100}
        )
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_relaxation_convergence_after_relax" in names

    def test_full_iterations_no_trigger(self):
        rs = GraphOptimizationExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"relaxation_iterations": 80, "n_nodes": 100}
        )
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ctx)
        conv_diags = [
            d for d in diags
            if d.rule_name == "insert_relaxation_convergence_after_relax"
        ]
        assert len(conv_diags) == 0

    def test_diagnose_distance_overflow(self):
        rs = GraphOptimizationExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"max_distance": 1e18}
        )
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_distance_overflow_detection_before_extract" in names

    def test_normal_distances_no_trigger(self):
        rs = GraphOptimizationExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"max_distance": 1000.0}
        )
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ctx)
        overflow_diags = [
            d for d in diags
            if d.rule_name == "insert_distance_overflow_detection_before_extract"
        ]
        assert len(overflow_diags) == 0

    def test_diagnose_dense_graph(self):
        rs = GraphOptimizationExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"graph_density": 0.8}
        )
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_graph_density_analysis_before_relax" in names

    def test_sparse_graph_no_trigger(self):
        rs = GraphOptimizationExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"graph_density": 0.05}
        )
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ctx)
        density_diags = [
            d for d in diags
            if d.rule_name == "insert_graph_density_analysis_before_relax"
        ]
        assert len(density_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = GraphOptimizationExpansionRuleSet()
        cdg = _graph_opt_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestGraphOptimizationIntegration:
    def test_full_expansion_with_all_diagnostics(self):
        """End-to-end: diagnostics fire, engine expands graph optimization CDG."""
        rs = GraphOptimizationExpansionRuleSet()
        engine = ExpansionEngine([rs])

        ctx = ExpansionContext(
            intermediates={
                "min_edge_weight": -3.0,
                "relaxation_iterations": 5,
                "n_nodes": 100,
                "max_distance": 1e18,
                "graph_density": 0.8,
            }
        )
        cdg = _graph_opt_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "detect_negative_weights",
            "monitor_relaxation_convergence",
            "detect_distance_overflow",
            "analyze_graph_density",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_dc_rules(self):
        """Graph optimization + D&C rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.divide_and_conquer import (
            DivideAndConquerExpansionRuleSet,
        )

        engine = ExpansionEngine([
            GraphOptimizationExpansionRuleSet(),
            DivideAndConquerExpansionRuleSet(),
        ])

        # Only graph opt data, no D&C data → only graph opt diags fire
        ctx = ExpansionContext(
            intermediates={"min_edge_weight": -1.0}
        )
        cdg = _graph_opt_cdg()
        result = engine.expand(cdg, ctx)

        # D&C rules should NOT have fired
        dc_atoms = {
            "measure_split_balance",
            "check_recursion_depth",
            "profile_merge_cost",
            "detect_subproblem_overlap",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & dc_atoms)
