"""Tests for the Dynamic Programming expansion rules and runtime atoms."""

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
from sciona.principal.expansion_rules.dynamic_programming import (
    DynamicProgrammingExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_dynamic_programming import (
    compress_dp_table,
    detect_table_sparsity,
    prune_infeasible_states,
    validate_subproblem_overlap,
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


def _dp_cdg():
    """Build a minimal DP CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("src", "Source", ConceptType.CUSTOM),
            _node("ds", "Define Subproblems", ConceptType.DYNAMIC_PROGRAMMING),
            _node("bc", "Base Case", ConceptType.DYNAMIC_PROGRAMMING),
            _node("rec", "Recurrence", ConceptType.DYNAMIC_PROGRAMMING),
            _node("mem", "Memoize", ConceptType.DYNAMIC_PROGRAMMING),
            _node("ext", "Extract Solution", ConceptType.DYNAMIC_PROGRAMMING),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("src", "ds"),
            _edge("ds", "bc"),
            _edge("bc", "rec"),
            _edge("rec", "mem"),
            _edge("mem", "ext"),
            _edge("ext", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestDetectTableSparsity:
    def test_dense_table(self):
        table = np.array([[1.0, 2.0], [3.0, 4.0]])
        density, indices = detect_table_sparsity(table)
        assert density == 1.0
        assert len(indices) == 4

    def test_sparse_table(self):
        table = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        density, indices = detect_table_sparsity(table)
        assert density == pytest.approx(2.0 / 9.0)
        assert len(indices) == 2

    def test_nan_treated_as_unfilled(self):
        table = np.array([[1.0, np.nan], [np.nan, 2.0]])
        density, indices = detect_table_sparsity(table)
        assert density == pytest.approx(0.5)
        assert len(indices) == 2

    def test_with_fill_mask(self):
        table = np.zeros((3, 3))
        mask = np.array([[True, False, False], [False, True, False], [False, False, True]])
        density, indices = detect_table_sparsity(table, fill_mask=mask)
        assert density == pytest.approx(3.0 / 9.0)

    def test_empty_table(self):
        table = np.empty((0,))
        density, indices = detect_table_sparsity(table)
        assert density == 0.0
        assert len(indices) == 0


class TestPruneInfeasibleStates:
    def test_all_feasible(self):
        shape = (5, 5)
        bounds = np.array([[0.0, 4.0], [0.0, 4.0]])
        constraints = bounds.copy()
        mask, n_pruned = prune_infeasible_states(shape, constraints, bounds)
        assert mask.shape == shape
        assert np.all(mask)
        assert n_pruned == 0

    def test_partial_pruning(self):
        shape = (10, 10)
        bounds = np.array([[2.0, 7.0], [3.0, 8.0]])
        constraints = bounds.copy()
        mask, n_pruned = prune_infeasible_states(shape, constraints, bounds)
        assert mask.shape == shape
        assert n_pruned > 0
        # Cells outside [2,7] x [3,8] should be infeasible
        assert not mask[0, 0]
        assert not mask[1, 9]
        assert mask[3, 5]

    def test_no_feasible(self):
        shape = (5, 5)
        # Bounds that exclude all indices
        bounds = np.array([[10.0, 20.0], [10.0, 20.0]])
        constraints = bounds.copy()
        mask, n_pruned = prune_infeasible_states(shape, constraints, bounds)
        assert not np.any(mask)
        assert n_pruned == 25


class TestCompressDpTable:
    def test_basic_compression(self):
        table = np.arange(50).reshape(10, 5)
        compressed, ratio = compress_dp_table(table, 3)
        assert compressed.shape == (3, 5)
        np.testing.assert_array_equal(compressed, table[-3:])
        assert ratio == pytest.approx(0.7)

    def test_reuse_distance_exceeds_rows(self):
        table = np.arange(10).reshape(5, 2)
        compressed, ratio = compress_dp_table(table, 10)
        np.testing.assert_array_equal(compressed, table)
        assert ratio == 0.0

    def test_reuse_distance_one(self):
        table = np.arange(20).reshape(10, 2)
        compressed, ratio = compress_dp_table(table, 1)
        assert compressed.shape == (1, 2)
        np.testing.assert_array_equal(compressed, table[-1:])
        assert ratio == pytest.approx(0.9)

    def test_zero_reuse_distance(self):
        table = np.arange(10).reshape(5, 2)
        compressed, ratio = compress_dp_table(table, 0)
        np.testing.assert_array_equal(compressed, table)
        assert ratio == 0.0


class TestValidateSubproblemOverlap:
    def test_high_overlap(self):
        # Each subproblem called ~5 times on average
        counts = np.array([5, 4, 6, 5, 3, 7])
        ratio, has_overlap = validate_subproblem_overlap(counts)
        assert ratio > 1.5
        assert has_overlap

    def test_low_overlap(self):
        # Each subproblem called exactly once
        counts = np.array([1, 1, 1, 1, 1])
        ratio, has_overlap = validate_subproblem_overlap(counts)
        assert ratio == pytest.approx(1.0)
        assert not has_overlap

    def test_empty_counts(self):
        counts = np.array([], dtype=np.float64)
        ratio, has_overlap = validate_subproblem_overlap(counts)
        assert ratio == 0.0
        assert not has_overlap

    def test_zero_counts_ignored(self):
        # Some subproblems never called; only active ones count
        counts = np.array([0, 0, 3, 0, 4])
        ratio, has_overlap = validate_subproblem_overlap(counts)
        assert ratio == pytest.approx(3.5)
        assert has_overlap

    def test_borderline(self):
        # Exactly 1.5 should NOT trigger overlap
        counts = np.array([1.5, 1.5, 1.5])
        ratio, has_overlap = validate_subproblem_overlap(counts)
        assert ratio == pytest.approx(1.5)
        assert not has_overlap


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestDynamicProgrammingRules:
    def _get_rules(self):
        rs = DynamicProgrammingExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_sparsity_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_sparsity_detection_before_recurrence"]
        rw = GraphRewriter()
        cdg = _dp_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_table_sparsity" in prims
        assert len(g.nodes) == 8  # 7 + 1

    def test_constraint_pruning_applies(self):
        rules = self._get_rules()
        rule = rules["insert_constraint_pruning_before_recurrence"]
        rw = GraphRewriter()
        cdg = _dp_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "prune_infeasible_states" in prims

    def test_table_compression_applies(self):
        rules = self._get_rules()
        rule = rules["insert_table_compression_after_memoize"]
        rw = GraphRewriter()
        cdg = _dp_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "compress_dp_table" in prims

    def test_overlap_validation_applies(self):
        rules = self._get_rules()
        rule = rules["insert_overlap_validation_before_base_case"]
        rw = GraphRewriter()
        cdg = _dp_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "validate_subproblem_overlap" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestDynamicProgrammingDiagnostics:
    def test_diagnose_sparse_table(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(intermediates={"table_density": 0.15})
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_sparsity_detection_before_recurrence" in names

    def test_dense_table_no_trigger(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(intermediates={"table_density": 0.8})
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        sparsity_diags = [
            d for d in diags
            if d.rule_name == "insert_sparsity_detection_before_recurrence"
        ]
        assert len(sparsity_diags) == 0

    def test_diagnose_infeasible_states(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(intermediates={"infeasible_fraction": 0.7})
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_constraint_pruning_before_recurrence" in names

    def test_low_infeasible_no_trigger(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(intermediates={"infeasible_fraction": 0.3})
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        pruning_diags = [
            d for d in diags
            if d.rule_name == "insert_constraint_pruning_before_recurrence"
        ]
        assert len(pruning_diags) == 0

    def test_diagnose_table_memory(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"table_memory_mb": 250.0, "reuse_distance": 3}
        )
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_table_compression_after_memoize" in names

    def test_small_table_no_trigger(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"table_memory_mb": 50.0, "reuse_distance": 3}
        )
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        mem_diags = [
            d for d in diags
            if d.rule_name == "insert_table_compression_after_memoize"
        ]
        assert len(mem_diags) == 0

    def test_no_reuse_distance_no_trigger(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"table_memory_mb": 250.0}
        )
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        mem_diags = [
            d for d in diags
            if d.rule_name == "insert_table_compression_after_memoize"
        ]
        assert len(mem_diags) == 0

    def test_diagnose_low_overlap(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"call_counts": np.array([1, 1, 1, 1])}
        )
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_overlap_validation_before_base_case" in names

    def test_high_overlap_no_trigger(self):
        rs = DynamicProgrammingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"call_counts": np.array([5, 4, 6, 5])}
        )
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ctx)
        overlap_diags = [
            d for d in diags
            if d.rule_name == "insert_overlap_validation_before_base_case"
        ]
        assert len(overlap_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = DynamicProgrammingExpansionRuleSet()
        cdg = _dp_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestDynamicProgrammingIntegration:
    def test_full_expansion_with_sparse_low_overlap(self):
        """End-to-end: diagnostics fire, engine expands DP CDG."""
        rs = DynamicProgrammingExpansionRuleSet()
        engine = ExpansionEngine([rs])

        ctx = ExpansionContext(
            intermediates={
                "table_density": 0.1,
                "infeasible_fraction": 0.8,
                "table_memory_mb": 500.0,
                "reuse_distance": 2,
                "call_counts": np.array([1, 1, 1, 1]),
            }
        )
        cdg = _dp_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "detect_table_sparsity",
            "prune_infeasible_states",
            "compress_dp_table",
            "validate_subproblem_overlap",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_graph_traversal_rules(self):
        """DP + graph traversal rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.graph_traversal import (
            GraphTraversalExpansionRuleSet,
        )

        engine = ExpansionEngine([
            DynamicProgrammingExpansionRuleSet(),
            GraphTraversalExpansionRuleSet(),
        ])

        # Only DP data, no graph data → only DP diags fire
        ctx = ExpansionContext(
            intermediates={"table_density": 0.1}
        )
        cdg = _dp_cdg()
        result = engine.expand(cdg, ctx)

        # Graph traversal rules should NOT have fired
        graph_atoms = {
            "detect_cycles",
            "check_connectivity",
            "detect_frontier_overflow",
            "compact_visited_set",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & graph_atoms)
