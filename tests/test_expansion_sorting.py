"""Tests for the Sorting expansion rules and runtime atoms."""

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
from sciona.principal.expansion_rules.sorting import (
    SortingExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_sorting import (
    analyze_comparison_count,
    analyze_swap_count,
    measure_presortedness,
    validate_stability,
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


def _sorting_cdg():
    """Build a minimal sorting CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("src", "Source", ConceptType.CUSTOM),
            _node("cmp", "Compare", ConceptType.SORTING),
            _node("swp", "Swap", ConceptType.SORTING),
            _node("rec", "Recurse/Iterate", ConceptType.SORTING),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("src", "cmp"),
            _edge("cmp", "swp"),
            _edge("swp", "rec"),
            _edge("rec", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestMeasurePresortedness:
    def test_sorted_input(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ratio, n_inv = measure_presortedness(data)
        assert ratio == 0.0
        assert n_inv == 0

    def test_reverse_sorted(self):
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ratio, n_inv = measure_presortedness(data)
        assert ratio == 1.0
        assert n_inv == 4

    def test_partially_sorted(self):
        data = np.array([1.0, 3.0, 2.0, 4.0, 5.0])
        ratio, n_inv = measure_presortedness(data)
        assert 0.0 < ratio < 1.0
        assert n_inv == 1

    def test_single_element(self):
        ratio, n_inv = measure_presortedness(np.array([42.0]))
        assert ratio == 0.0
        assert n_inv == 0

    def test_empty(self):
        ratio, n_inv = measure_presortedness(np.array([]))
        assert ratio == 0.0
        assert n_inv == 0


class TestAnalyzeComparisonCount:
    def test_normal_count(self):
        # n=1000, expected ~20000, actual=15000
        ratio, is_exc = analyze_comparison_count(15000, 1000)
        assert ratio < 1.0
        assert not is_exc

    def test_excessive_count(self):
        # n=1000, expected ~20000, actual=500000 (quadratic)
        ratio, is_exc = analyze_comparison_count(500000, 1000)
        assert ratio > 1.0
        assert is_exc

    def test_single_element(self):
        ratio, is_exc = analyze_comparison_count(0, 1)
        assert ratio == 0.0
        assert not is_exc


class TestAnalyzeSwapCount:
    def test_normal_count(self):
        ratio, is_exc = analyze_swap_count(10000, 1000)
        assert ratio < 1.0
        assert not is_exc

    def test_excessive_count(self):
        ratio, is_exc = analyze_swap_count(500000, 1000)
        assert ratio > 1.0
        assert is_exc

    def test_single_element(self):
        ratio, is_exc = analyze_swap_count(0, 1)
        assert ratio == 0.0
        assert not is_exc


class TestValidateStability:
    def test_stable_sort(self):
        keys = np.array([1.0, 1.0, 2.0, 2.0])
        orig = np.array([0, 1, 2, 3])
        srt = np.array([0, 1, 2, 3])  # equal keys in original order
        violations, is_stable = validate_stability(keys, orig, srt)
        assert violations == 0
        assert is_stable

    def test_unstable_sort(self):
        keys = np.array([1.0, 1.0, 2.0, 2.0])
        orig = np.array([0, 1, 2, 3])
        srt = np.array([1, 0, 3, 2])  # equal keys reversed
        violations, is_stable = validate_stability(keys, orig, srt)
        assert violations == 2
        assert not is_stable

    def test_no_equal_keys(self):
        keys = np.array([1.0, 2.0, 3.0])
        orig = np.array([0, 1, 2])
        srt = np.array([0, 1, 2])
        violations, is_stable = validate_stability(keys, orig, srt)
        assert violations == 0
        assert is_stable

    def test_single_element(self):
        violations, is_stable = validate_stability(
            np.array([1.0]), np.array([0]), np.array([0])
        )
        assert violations == 0
        assert is_stable

    def test_empty(self):
        violations, is_stable = validate_stability(
            np.array([]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        )
        assert violations == 0
        assert is_stable


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestSortingRules:
    def _get_rules(self):
        rs = SortingExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_presortedness_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_presortedness_detection_before_compare"]
        rw = GraphRewriter()
        cdg = _sorting_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "measure_presortedness" in prims
        assert len(g.nodes) == 6  # 5 + 1

    def test_comparison_count_analysis_applies(self):
        rules = self._get_rules()
        rule = rules["insert_comparison_count_analysis_after_compare"]
        rw = GraphRewriter()
        cdg = _sorting_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "analyze_comparison_count" in prims

    def test_swap_count_analysis_applies(self):
        rules = self._get_rules()
        rule = rules["insert_swap_count_analysis_after_swap"]
        rw = GraphRewriter()
        cdg = _sorting_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "analyze_swap_count" in prims

    def test_stability_validation_applies(self):
        rules = self._get_rules()
        rule = rules["insert_stability_validation_after_recurse"]
        rw = GraphRewriter()
        cdg = _sorting_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "validate_stability" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestSortingDiagnostics:
    def test_diagnose_presortedness(self):
        rs = SortingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"disorder_ratio": 0.02}
        )
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_presortedness_detection_before_compare" in names

    def test_unsorted_no_trigger(self):
        rs = SortingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"disorder_ratio": 0.5}
        )
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ctx)
        presort_diags = [
            d for d in diags
            if d.rule_name == "insert_presortedness_detection_before_compare"
        ]
        assert len(presort_diags) == 0

    def test_diagnose_excessive_comparisons(self):
        rs = SortingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"n_comparisons": 500000, "n_elements": 1000}
        )
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_comparison_count_analysis_after_compare" in names

    def test_normal_comparisons_no_trigger(self):
        rs = SortingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"n_comparisons": 5000, "n_elements": 1000}
        )
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ctx)
        comp_diags = [
            d for d in diags
            if d.rule_name == "insert_comparison_count_analysis_after_compare"
        ]
        assert len(comp_diags) == 0

    def test_diagnose_excessive_swaps(self):
        rs = SortingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"n_swaps": 500000, "n_elements": 1000}
        )
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_swap_count_analysis_after_swap" in names

    def test_normal_swaps_no_trigger(self):
        rs = SortingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"n_swaps": 5000, "n_elements": 1000}
        )
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ctx)
        swap_diags = [
            d for d in diags
            if d.rule_name == "insert_swap_count_analysis_after_swap"
        ]
        assert len(swap_diags) == 0

    def test_diagnose_stability_violations(self):
        rs = SortingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"n_stability_violations": 5}
        )
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_stability_validation_after_recurse" in names

    def test_stable_sort_no_trigger(self):
        rs = SortingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"n_stability_violations": 0}
        )
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ctx)
        stability_diags = [
            d for d in diags
            if d.rule_name == "insert_stability_validation_after_recurse"
        ]
        assert len(stability_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = SortingExpansionRuleSet()
        cdg = _sorting_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestSortingIntegration:
    def test_full_expansion_with_all_diagnostics(self):
        """End-to-end: diagnostics fire, engine expands sorting CDG."""
        rs = SortingExpansionRuleSet()
        engine = ExpansionEngine([rs])

        ctx = ExpansionContext(
            intermediates={
                "disorder_ratio": 0.02,
                "n_comparisons": 500000,
                "n_elements": 1000,
                "n_swaps": 500000,
                "n_stability_violations": 3,
            }
        )
        cdg = _sorting_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "measure_presortedness",
            "analyze_comparison_count",
            "analyze_swap_count",
            "validate_stability",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_graph_opt_rules(self):
        """Sorting + Graph Optimization rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.graph_optimization import (
            GraphOptimizationExpansionRuleSet,
        )

        engine = ExpansionEngine([
            SortingExpansionRuleSet(),
            GraphOptimizationExpansionRuleSet(),
        ])

        # Only sorting data, no graph opt data → only sorting diags fire
        ctx = ExpansionContext(
            intermediates={"disorder_ratio": 0.02}
        )
        cdg = _sorting_cdg()
        result = engine.expand(cdg, ctx)

        # Graph opt rules should NOT have fired
        graph_opt_atoms = {
            "detect_negative_weights",
            "monitor_relaxation_convergence",
            "detect_distance_overflow",
            "analyze_graph_density",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & graph_opt_atoms)
