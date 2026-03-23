"""Tests for the Divide and Conquer expansion rules and runtime atoms."""

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
from sciona.principal.expansion_rules.divide_and_conquer import (
    DivideAndConquerExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_divide_and_conquer import (
    check_recursion_depth,
    detect_subproblem_overlap,
    measure_split_balance,
    profile_merge_cost,
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


def _dc_cdg():
    """Build a minimal D&C CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("src", "Source", ConceptType.CUSTOM),
            _node("sp", "Split", ConceptType.DIVIDE_AND_CONQUER),
            _node("rl", "Recurse Left", ConceptType.DIVIDE_AND_CONQUER),
            _node("rr", "Recurse Right", ConceptType.DIVIDE_AND_CONQUER),
            _node("mg", "Merge", ConceptType.DIVIDE_AND_CONQUER),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("src", "sp"),
            _edge("sp", "rl"),
            _edge("sp", "rr"),
            _edge("rl", "mg"),
            _edge("rr", "mg"),
            _edge("mg", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestMeasureSplitBalance:
    def test_balanced_splits(self):
        left = np.array([50.0, 25.0, 12.0])
        right = np.array([50.0, 25.0, 13.0])
        mean_bal, ratios = measure_split_balance(left, right)
        assert mean_bal > 0.9
        assert len(ratios) == 3

    def test_unbalanced_splits(self):
        left = np.array([90.0, 80.0])
        right = np.array([10.0, 20.0])
        mean_bal, ratios = measure_split_balance(left, right)
        assert mean_bal < 0.3
        assert ratios[0] == pytest.approx(10.0 / 90.0)

    def test_empty(self):
        mean_bal, ratios = measure_split_balance(np.array([]), np.array([]))
        assert mean_bal == 1.0
        assert len(ratios) == 0

    def test_zero_size_subproblem(self):
        left = np.array([10.0, 0.0])
        right = np.array([0.0, 5.0])
        mean_bal, ratios = measure_split_balance(left, right)
        assert len(ratios) == 2
        # (10, 0) → 0/10 = 0.0, (0, 5) → 0/5 = 0.0
        assert ratios[0] == 0.0
        assert ratios[1] == 0.0

    def test_mismatched_lengths(self):
        left = np.array([50.0, 25.0, 12.0])
        right = np.array([50.0])
        mean_bal, ratios = measure_split_balance(left, right)
        # Only first pair considered
        assert len(ratios) == 1


class TestCheckRecursionDepth:
    def test_normal_depth(self):
        # n=1024, log2=10, expected_max=20, depth=15
        ratio, is_excessive = check_recursion_depth(15, 1024)
        assert ratio < 1.0
        assert not is_excessive

    def test_excessive_depth(self):
        # n=1024, log2=10, expected_max=20, depth=50
        ratio, is_excessive = check_recursion_depth(50, 1024)
        assert ratio > 1.0
        assert is_excessive

    def test_single_element(self):
        ratio, is_excessive = check_recursion_depth(1, 1)
        assert ratio == 0.0
        assert not is_excessive

    def test_small_input(self):
        # n=4, log2=2, expected_max=4, depth=3
        ratio, is_excessive = check_recursion_depth(3, 4)
        assert not is_excessive


class TestProfileMergeCost:
    def test_merge_dominated(self):
        merge = np.array([80.0, 70.0, 60.0])
        total = np.array([100.0, 100.0, 100.0])
        mean_frac, fractions = profile_merge_cost(merge, total)
        assert mean_frac == pytest.approx(0.7)
        assert fractions[0] == pytest.approx(0.8)

    def test_low_merge_cost(self):
        merge = np.array([10.0, 5.0])
        total = np.array([100.0, 100.0])
        mean_frac, fractions = profile_merge_cost(merge, total)
        assert mean_frac < 0.1

    def test_empty(self):
        mean_frac, fractions = profile_merge_cost(np.array([]), np.array([]))
        assert mean_frac == 0.0
        assert len(fractions) == 0

    def test_zero_total_time(self):
        merge = np.array([10.0])
        total = np.array([0.0])
        mean_frac, fractions = profile_merge_cost(merge, total)
        assert fractions[0] == 0.0


class TestDetectSubproblemOverlap:
    def test_no_overlap(self):
        hashes = np.array([1, 2, 3, 4, 5])
        ratio, n_dup = detect_subproblem_overlap(hashes)
        assert ratio == 0.0
        assert n_dup == 0

    def test_high_overlap(self):
        hashes = np.array([1, 1, 1, 2, 2, 3])
        ratio, n_dup = detect_subproblem_overlap(hashes)
        assert n_dup == 3  # 2 extra 1s + 1 extra 2
        assert ratio == pytest.approx(3 / 6)

    def test_empty(self):
        ratio, n_dup = detect_subproblem_overlap(np.array([], dtype=np.int64))
        assert ratio == 0.0
        assert n_dup == 0

    def test_all_same(self):
        hashes = np.array([42, 42, 42, 42])
        ratio, n_dup = detect_subproblem_overlap(hashes)
        assert n_dup == 3
        assert ratio == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestDivideAndConquerRules:
    def _get_rules(self):
        rs = DivideAndConquerExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_split_balance_applies(self):
        rules = self._get_rules()
        rule = rules["insert_split_balance_after_split"]
        rw = GraphRewriter()
        cdg = _dc_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "measure_split_balance" in prims
        assert len(g.nodes) == 7  # 6 + 1

    def test_recursion_depth_check_applies(self):
        rules = self._get_rules()
        rule = rules["insert_recursion_depth_check_before_recurse"]
        rw = GraphRewriter()
        cdg = _dc_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "check_recursion_depth" in prims

    def test_merge_cost_profiling_applies(self):
        rules = self._get_rules()
        rule = rules["insert_merge_cost_profiling_after_merge"]
        rw = GraphRewriter()
        cdg = _dc_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "profile_merge_cost" in prims

    def test_subproblem_overlap_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_subproblem_overlap_detection_before_split"]
        rw = GraphRewriter()
        cdg = _dc_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_subproblem_overlap" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestDivideAndConquerDiagnostics:
    def test_diagnose_unbalanced_splits(self):
        rs = DivideAndConquerExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"split_balance": 0.2}
        )
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_split_balance_after_split" in names

    def test_balanced_splits_no_trigger(self):
        rs = DivideAndConquerExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"split_balance": 0.9}
        )
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ctx)
        balance_diags = [
            d for d in diags
            if d.rule_name == "insert_split_balance_after_split"
        ]
        assert len(balance_diags) == 0

    def test_diagnose_excessive_depth(self):
        rs = DivideAndConquerExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"recursion_depth": 50, "input_size": 1024}
        )
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_recursion_depth_check_before_recurse" in names

    def test_normal_depth_no_trigger(self):
        rs = DivideAndConquerExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"recursion_depth": 10, "input_size": 1024}
        )
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ctx)
        depth_diags = [
            d for d in diags
            if d.rule_name == "insert_recursion_depth_check_before_recurse"
        ]
        assert len(depth_diags) == 0

    def test_diagnose_merge_dominated(self):
        rs = DivideAndConquerExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"merge_fraction": 0.75}
        )
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_merge_cost_profiling_after_merge" in names

    def test_low_merge_fraction_no_trigger(self):
        rs = DivideAndConquerExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"merge_fraction": 0.2}
        )
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ctx)
        merge_diags = [
            d for d in diags
            if d.rule_name == "insert_merge_cost_profiling_after_merge"
        ]
        assert len(merge_diags) == 0

    def test_diagnose_subproblem_overlap(self):
        rs = DivideAndConquerExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"subproblem_overlap_ratio": 0.3}
        )
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_subproblem_overlap_detection_before_split" in names

    def test_no_overlap_no_trigger(self):
        rs = DivideAndConquerExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"subproblem_overlap_ratio": 0.05}
        )
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ctx)
        overlap_diags = [
            d for d in diags
            if d.rule_name == "insert_subproblem_overlap_detection_before_split"
        ]
        assert len(overlap_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = DivideAndConquerExpansionRuleSet()
        cdg = _dc_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestDivideAndConquerIntegration:
    def test_full_expansion_with_all_diagnostics(self):
        """End-to-end: diagnostics fire, engine expands D&C CDG."""
        rs = DivideAndConquerExpansionRuleSet()
        engine = ExpansionEngine([rs])

        ctx = ExpansionContext(
            intermediates={
                "split_balance": 0.1,
                "recursion_depth": 100,
                "input_size": 1024,
                "merge_fraction": 0.8,
                "subproblem_overlap_ratio": 0.4,
            }
        )
        cdg = _dc_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "measure_split_balance",
            "check_recursion_depth",
            "profile_merge_cost",
            "detect_subproblem_overlap",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_greedy_rules(self):
        """D&C + Greedy rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.greedy import (
            GreedyExpansionRuleSet,
        )

        engine = ExpansionEngine([
            DivideAndConquerExpansionRuleSet(),
            GreedyExpansionRuleSet(),
        ])

        # Only D&C data, no greedy data → only D&C diags fire
        ctx = ExpansionContext(
            intermediates={"split_balance": 0.1}
        )
        cdg = _dc_cdg()
        result = engine.expand(cdg, ctx)

        # Greedy rules should NOT have fired
        greedy_atoms = {
            "validate_matroid_exchange",
            "detect_criterion_ties",
            "estimate_solution_quality",
            "detect_redundant_feasibility",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & greedy_atoms)
