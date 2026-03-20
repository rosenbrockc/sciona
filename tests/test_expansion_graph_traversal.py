"""Tests for the Graph Traversal expansion rules and runtime atoms."""

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
from sciona.principal.expansion_rules.graph_traversal import (
    GraphTraversalExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_graph_traversal import (
    check_connectivity,
    compact_visited_set,
    detect_cycles,
    detect_frontier_overflow,
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


def _traversal_cdg():
    """Build a minimal graph traversal CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("src", "Source", ConceptType.CUSTOM),
            _node("iv", "Init Visited", ConceptType.GRAPH_TRAVERSAL),
            _node("pn", "Pick Next", ConceptType.GRAPH_TRAVERSAL),
            _node("proc", "Process Node", ConceptType.GRAPH_TRAVERSAL),
            _node("uf", "Update Frontier", ConceptType.GRAPH_TRAVERSAL),
            _node("ct", "Check Termination", ConceptType.GRAPH_TRAVERSAL),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("src", "iv"),
            _edge("iv", "pn"),
            _edge("pn", "proc"),
            _edge("proc", "uf"),
            _edge("uf", "ct"),
            _edge("ct", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestDetectCycles:
    def test_dag_no_cycles(self):
        # Simple DAG: 0→1→2→3
        adj = np.array([[0, 1], [1, 2], [2, 3]])
        has_cycle, cycle_nodes = detect_cycles(adj, 4)
        assert not has_cycle
        assert len(cycle_nodes) == 0

    def test_simple_cycle(self):
        # 0→1→2→0
        adj = np.array([[0, 1], [1, 2], [2, 0]])
        has_cycle, cycle_nodes = detect_cycles(adj, 3)
        assert has_cycle
        assert len(cycle_nodes) > 0
        # All three nodes should be in the cycle
        assert set(cycle_nodes) == {0, 1, 2}

    def test_self_loop(self):
        # 0→0
        adj = np.array([[0, 0]])
        has_cycle, cycle_nodes = detect_cycles(adj, 1)
        assert has_cycle

    def test_empty_graph(self):
        adj = np.empty((0, 2), dtype=np.int64)
        has_cycle, cycle_nodes = detect_cycles(adj, 0)
        assert not has_cycle
        assert len(cycle_nodes) == 0

    def test_disconnected_with_cycle(self):
        # Component 1: 0→1 (no cycle)
        # Component 2: 2→3→2 (cycle)
        adj = np.array([[0, 1], [2, 3], [3, 2]])
        has_cycle, cycle_nodes = detect_cycles(adj, 4)
        assert has_cycle
        assert 2 in cycle_nodes
        assert 3 in cycle_nodes


class TestCheckConnectivity:
    def test_connected_graph(self):
        # 0→1→2
        adj = np.array([[0, 1], [1, 2]])
        n_comp, labels = check_connectivity(adj, 3)
        assert n_comp == 1
        assert len(set(labels)) == 1

    def test_disconnected_graph(self):
        # Component 1: 0→1, Component 2: 2→3
        adj = np.array([[0, 1], [2, 3]])
        n_comp, labels = check_connectivity(adj, 4)
        assert n_comp == 2
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_isolated_nodes(self):
        # 0→1, node 2 is isolated
        adj = np.array([[0, 1]])
        n_comp, labels = check_connectivity(adj, 3)
        assert n_comp == 2

    def test_single_node(self):
        adj = np.empty((0, 2), dtype=np.int64)
        n_comp, labels = check_connectivity(adj, 1)
        assert n_comp == 1
        assert labels[0] == 0

    def test_empty_graph(self):
        n_comp, labels = check_connectivity(np.empty((0, 2), dtype=np.int64), 0)
        assert n_comp == 0
        assert len(labels) == 0


class TestCompactVisitedSet:
    def test_basic(self):
        visited = np.array([0, 2, 4])
        compact = compact_visited_set(visited, 5)
        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(compact, expected)

    def test_all_visited(self):
        visited = np.array([0, 1, 2, 3])
        compact = compact_visited_set(visited, 4)
        assert np.all(compact)

    def test_empty(self):
        visited = np.array([], dtype=np.int64)
        compact = compact_visited_set(visited, 5)
        assert not np.any(compact)
        assert len(compact) == 5

    def test_out_of_bounds_ignored(self):
        visited = np.array([0, 1, 100])
        compact = compact_visited_set(visited, 5)
        assert compact[0] and compact[1]
        assert sum(compact) == 2


class TestDetectFrontierOverflow:
    def test_no_overflow(self):
        # n=100, sqrt=10, max frontier = 5
        frontier_sizes = np.array([2, 3, 5, 4, 3])
        mask, max_f = detect_frontier_overflow(frontier_sizes, 100)
        assert not np.any(mask)
        assert max_f == 5

    def test_overflow(self):
        # n=100, sqrt=10, some frontiers > 10
        frontier_sizes = np.array([5, 15, 3, 20])
        mask, max_f = detect_frontier_overflow(frontier_sizes, 100)
        assert mask[1] and mask[3]
        assert not mask[0] and not mask[2]
        assert max_f == 20

    def test_empty(self):
        frontier_sizes = np.array([], dtype=np.int64)
        mask, max_f = detect_frontier_overflow(frontier_sizes, 100)
        assert len(mask) == 0
        assert max_f == 0


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestGraphTraversalRules:
    def _get_rules(self):
        rs = GraphTraversalExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_cycle_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_cycle_detection_before_init"]
        rw = GraphRewriter()
        cdg = _traversal_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_cycles" in prims
        assert len(g.nodes) == 8  # 7 + 1

    def test_connectivity_check_applies(self):
        rules = self._get_rules()
        rule = rules["insert_connectivity_check_before_init"]
        rw = GraphRewriter()
        cdg = _traversal_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "check_connectivity" in prims

    def test_frontier_overflow_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_frontier_overflow_detection_after_update"]
        rw = GraphRewriter()
        cdg = _traversal_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_frontier_overflow" in prims

    def test_visited_compaction_applies(self):
        rules = self._get_rules()
        rule = rules["insert_visited_compaction_after_update"]
        rw = GraphRewriter()
        cdg = _traversal_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "compact_visited_set" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestGraphTraversalDiagnostics:
    def test_diagnose_cycle(self):
        rs = GraphTraversalExpansionRuleSet()
        adj = np.array([[0, 1], [1, 2], [2, 0]])
        ctx = ExpansionContext(
            intermediates={"adjacency": adj, "n_nodes": 3}
        )
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_cycle_detection_before_init" in names

    def test_no_cycle_no_trigger(self):
        rs = GraphTraversalExpansionRuleSet()
        adj = np.array([[0, 1], [1, 2]])
        ctx = ExpansionContext(
            intermediates={"adjacency": adj, "n_nodes": 3}
        )
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ctx)
        cycle_diags = [
            d for d in diags
            if d.rule_name == "insert_cycle_detection_before_init"
        ]
        assert len(cycle_diags) == 0

    def test_diagnose_disconnected(self):
        rs = GraphTraversalExpansionRuleSet()
        adj = np.array([[0, 1], [2, 3]])
        ctx = ExpansionContext(
            intermediates={"adjacency": adj, "n_nodes": 4}
        )
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_connectivity_check_before_init" in names

    def test_connected_no_trigger(self):
        rs = GraphTraversalExpansionRuleSet()
        adj = np.array([[0, 1], [1, 2]])
        ctx = ExpansionContext(
            intermediates={"adjacency": adj, "n_nodes": 3}
        )
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ctx)
        conn_diags = [
            d for d in diags
            if d.rule_name == "insert_connectivity_check_before_init"
        ]
        assert len(conn_diags) == 0

    def test_diagnose_frontier_overflow(self):
        rs = GraphTraversalExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={
                "frontier_sizes": np.array([5, 50, 3]),
                "n_nodes": 100,
            }
        )
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_frontier_overflow_detection_after_update" in names

    def test_normal_frontier_no_trigger(self):
        rs = GraphTraversalExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={
                "frontier_sizes": np.array([2, 3, 4]),
                "n_nodes": 100,
            }
        )
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ctx)
        frontier_diags = [
            d for d in diags
            if d.rule_name == "insert_frontier_overflow_detection_after_update"
        ]
        assert len(frontier_diags) == 0

    def test_diagnose_visited_density(self):
        rs = GraphTraversalExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"visited_ratio": 0.95}
        )
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_visited_compaction_after_update" in names

    def test_low_visited_ratio_no_trigger(self):
        rs = GraphTraversalExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"visited_ratio": 0.3}
        )
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ctx)
        compact_diags = [
            d for d in diags
            if d.rule_name == "insert_visited_compaction_after_update"
        ]
        assert len(compact_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = GraphTraversalExpansionRuleSet()
        cdg = _traversal_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestGraphTraversalIntegration:
    def test_full_expansion_with_cyclic_disconnected_graph(self):
        """End-to-end: diagnostics fire, engine expands traversal CDG."""
        rs = GraphTraversalExpansionRuleSet()
        engine = ExpansionEngine([rs])

        # Cyclic + disconnected graph + frontier overflow + high visited ratio
        adj = np.array([[0, 1], [1, 2], [2, 0], [3, 4]])  # cycle in 0-1-2, disconnected 3-4
        ctx = ExpansionContext(
            intermediates={
                "adjacency": adj,
                "n_nodes": 5,
                "frontier_sizes": np.array([2, 20, 3]),
                "visited_ratio": 0.9,
            }
        )
        cdg = _traversal_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "detect_cycles",
            "check_connectivity",
            "detect_frontier_overflow",
            "compact_visited_set",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_mcmc_rules(self):
        """Graph traversal + MCMC rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.mcmc import (
            MCMCExpansionRuleSet,
        )

        engine = ExpansionEngine([
            GraphTraversalExpansionRuleSet(),
            MCMCExpansionRuleSet(),
        ])

        # Only graph data, no MCMC data → only graph diags fire
        adj = np.array([[0, 1], [1, 2], [2, 0]])
        ctx = ExpansionContext(
            intermediates={"adjacency": adj, "n_nodes": 3}
        )
        cdg = _traversal_cdg()
        result = engine.expand(cdg, ctx)

        # MCMC rules should NOT have fired
        mcmc_atoms = {
            "detect_divergent_transitions",
            "compute_dual_averaging_step_size",
            "estimate_mass_matrix",
            "compute_convergence_diagnostics",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & mcmc_atoms)
