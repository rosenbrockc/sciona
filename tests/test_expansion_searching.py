"""Tests for the Searching expansion rules and runtime atoms."""

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
from sciona.principal.expansion_rules.searching import (
    SearchingExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_searching import (
    analyze_distribution_uniformity,
    analyze_iteration_count,
    detect_midpoint_overflow,
    validate_sorted_order,
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


def _searching_cdg():
    """Build a minimal searching CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("src", "Source", ConceptType.CUSTOM),
            _node("ib", "Init Bounds", ConceptType.SEARCHING),
            _node("pr", "Probe", ConceptType.SEARCHING),
            _node("nr", "Narrow", ConceptType.SEARCHING),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("src", "ib"),
            _edge("ib", "pr"),
            _edge("pr", "nr"),
            _edge("nr", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestValidateSortedOrder:
    def test_sorted(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        violations, is_sorted = validate_sorted_order(data)
        assert violations == 0
        assert is_sorted

    def test_unsorted(self):
        data = np.array([1.0, 3.0, 2.0, 4.0])
        violations, is_sorted = validate_sorted_order(data)
        assert violations == 1
        assert not is_sorted

    def test_reverse_sorted(self):
        data = np.array([4.0, 3.0, 2.0, 1.0])
        violations, is_sorted = validate_sorted_order(data)
        assert violations == 3
        assert not is_sorted

    def test_single(self):
        violations, is_sorted = validate_sorted_order(np.array([42.0]))
        assert violations == 0
        assert is_sorted

    def test_empty(self):
        violations, is_sorted = validate_sorted_order(np.array([]))
        assert violations == 0
        assert is_sorted


class TestAnalyzeDistributionUniformity:
    def test_uniform(self):
        data = np.linspace(0, 100, 101)
        score, rec = analyze_distribution_uniformity(data)
        assert score > 0.9
        assert rec == "interpolation"

    def test_highly_skewed(self):
        # Exponential spacing — very non-uniform
        data = np.sort(np.exp(np.linspace(0, 10, 100)))
        score, rec = analyze_distribution_uniformity(data)
        assert score < 0.5
        assert rec == "binary"

    def test_all_same(self):
        data = np.array([5.0, 5.0, 5.0])
        score, rec = analyze_distribution_uniformity(data)
        assert score == 0.0

    def test_two_elements(self):
        score, rec = analyze_distribution_uniformity(np.array([1.0, 2.0]))
        assert score == 1.0
        assert rec == "binary"  # too small for interpolation


class TestDetectMidpointOverflow:
    def test_no_overflow(self):
        would_overflow, mid = detect_midpoint_overflow(0, 100)
        assert not would_overflow
        assert mid == 50

    def test_large_values_no_overflow(self):
        would_overflow, mid = detect_midpoint_overflow(0, 2**62)
        assert not would_overflow

    def test_overflow_risk(self):
        hi = np.iinfo(np.int64).max
        lo = np.iinfo(np.int64).max // 2 + 1
        would_overflow, mid = detect_midpoint_overflow(lo, hi)
        assert would_overflow
        # safe_mid should still be correct
        assert lo <= mid <= hi


class TestAnalyzeIterationCount:
    def test_normal(self):
        # n=1024, log2=10, expected_max=20, iters=8
        ratio, is_exc = analyze_iteration_count(8, 1024)
        assert ratio < 1.0
        assert not is_exc

    def test_excessive(self):
        # n=1024, expected_max=20, iters=50
        ratio, is_exc = analyze_iteration_count(50, 1024)
        assert ratio > 1.0
        assert is_exc

    def test_single_element(self):
        ratio, is_exc = analyze_iteration_count(1, 1)
        assert ratio == 0.0
        assert not is_exc


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestSearchingRules:
    def _get_rules(self):
        rs = SearchingExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_sorted_order_validation_applies(self):
        rules = self._get_rules()
        rule = rules["insert_sorted_order_validation_before_init"]
        rw = GraphRewriter()
        cdg = _searching_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "validate_sorted_order" in prims
        assert len(g.nodes) == 6

    def test_distribution_uniformity_applies(self):
        rules = self._get_rules()
        rule = rules["insert_distribution_uniformity_analysis_before_init"]
        rw = GraphRewriter()
        cdg = _searching_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "analyze_distribution_uniformity" in prims

    def test_midpoint_overflow_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_midpoint_overflow_detection_before_probe"]
        rw = GraphRewriter()
        cdg = _searching_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_midpoint_overflow" in prims

    def test_iteration_count_analysis_applies(self):
        rules = self._get_rules()
        rule = rules["insert_iteration_count_analysis_after_narrow"]
        rw = GraphRewriter()
        cdg = _searching_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "analyze_iteration_count" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestSearchingDiagnostics:
    def test_diagnose_unsorted(self):
        rs = SearchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"sort_violations": 5}
        )
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_sorted_order_validation_before_init" in names

    def test_sorted_no_trigger(self):
        rs = SearchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"sort_violations": 0}
        )
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ctx)
        sort_diags = [
            d for d in diags
            if d.rule_name == "insert_sorted_order_validation_before_init"
        ]
        assert len(sort_diags) == 0

    def test_diagnose_non_uniform(self):
        rs = SearchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"uniformity_score": 0.2}
        )
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_distribution_uniformity_analysis_before_init" in names

    def test_uniform_no_trigger(self):
        rs = SearchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"uniformity_score": 0.9}
        )
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ctx)
        uni_diags = [
            d for d in diags
            if d.rule_name == "insert_distribution_uniformity_analysis_before_init"
        ]
        assert len(uni_diags) == 0

    def test_diagnose_overflow_risk(self):
        rs = SearchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"max_index": np.iinfo(np.int64).max - 1}
        )
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_midpoint_overflow_detection_before_probe" in names

    def test_small_indices_no_trigger(self):
        rs = SearchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"max_index": 1000}
        )
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ctx)
        overflow_diags = [
            d for d in diags
            if d.rule_name == "insert_midpoint_overflow_detection_before_probe"
        ]
        assert len(overflow_diags) == 0

    def test_diagnose_excessive_iterations(self):
        rs = SearchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"search_iterations": 100, "n_elements": 1024}
        )
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_iteration_count_analysis_after_narrow" in names

    def test_normal_iterations_no_trigger(self):
        rs = SearchingExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"search_iterations": 8, "n_elements": 1024}
        )
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ctx)
        iter_diags = [
            d for d in diags
            if d.rule_name == "insert_iteration_count_analysis_after_narrow"
        ]
        assert len(iter_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = SearchingExpansionRuleSet()
        cdg = _searching_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestSearchingIntegration:
    def test_full_expansion_with_all_diagnostics(self):
        rs = SearchingExpansionRuleSet()
        engine = ExpansionEngine([rs])

        ctx = ExpansionContext(
            intermediates={
                "sort_violations": 3,
                "uniformity_score": 0.2,
                "max_index": np.iinfo(np.int64).max - 1,
                "search_iterations": 100,
                "n_elements": 1024,
            }
        )
        cdg = _searching_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "validate_sorted_order",
            "analyze_distribution_uniformity",
            "detect_midpoint_overflow",
            "analyze_iteration_count",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_string_matching_rules(self):
        from sciona.principal.expansion_rules.string_matching import (
            StringMatchingExpansionRuleSet,
        )

        engine = ExpansionEngine([
            SearchingExpansionRuleSet(),
            StringMatchingExpansionRuleSet(),
        ])

        ctx = ExpansionContext(
            intermediates={"sort_violations": 3}
        )
        cdg = _searching_cdg()
        result = engine.expand(cdg, ctx)

        sm_atoms = {
            "analyze_alphabet_size",
            "check_pattern_text_ratio",
            "measure_hash_collision_rate",
            "validate_failure_function",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & sm_atoms)
