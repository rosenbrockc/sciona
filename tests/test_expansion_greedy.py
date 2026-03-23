"""Tests for the Greedy expansion rules and runtime atoms."""

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
from sciona.principal.expansion_rules.greedy import (
    GreedyExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_greedy import (
    detect_criterion_ties,
    detect_redundant_feasibility,
    estimate_solution_quality,
    validate_matroid_exchange,
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


def _greedy_cdg():
    """Build a minimal greedy CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("src", "Source", ConceptType.CUSTOM),
            _node("sc", "Sort Candidates", ConceptType.GREEDY),
            _node("gc", "Greedy Choice", ConceptType.GREEDY),
            _node("fc", "Feasibility Check", ConceptType.GREEDY),
            _node("us", "Update Solution", ConceptType.GREEDY),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("src", "sc"),
            _edge("sc", "gc"),
            _edge("gc", "fc"),
            _edge("fc", "us"),
            _edge("us", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestValidateMatroidExchange:
    def test_matroid_sets(self):
        # Sets where exchange property holds trivially
        sets = [np.array([0, 1]), np.array([0, 1, 2]), np.array([1, 2, 3])]
        ratio, is_matroid = validate_matroid_exchange(sets, 4)
        assert ratio > 0.0
        assert isinstance(is_matroid, bool)

    def test_single_set(self):
        sets = [np.array([0, 1, 2])]
        ratio, is_matroid = validate_matroid_exchange(sets, 5)
        assert ratio == 1.0
        assert is_matroid

    def test_empty_sets(self):
        ratio, is_matroid = validate_matroid_exchange([], 5)
        assert ratio == 1.0
        assert is_matroid

    def test_equal_size_sets(self):
        # All sets same size — no pairs with different sizes
        sets = [np.array([0, 1]), np.array([2, 3])]
        ratio, is_matroid = validate_matroid_exchange(sets, 4)
        assert ratio == 1.0
        assert is_matroid

    def test_zero_ground_set(self):
        ratio, is_matroid = validate_matroid_exchange([np.array([])], 0)
        assert ratio == 1.0
        assert is_matroid


class TestDetectCriterionTies:
    def test_no_ties(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        n_ties, groups = detect_criterion_ties(scores)
        assert n_ties == 0
        assert len(groups) == 4

    def test_all_tied(self):
        scores = np.array([5.0, 5.0, 5.0])
        n_ties, groups = detect_criterion_ties(scores)
        assert n_ties == 3
        # All should have the same group label
        assert len(set(groups)) == 1

    def test_partial_ties(self):
        scores = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
        n_ties, groups = detect_criterion_ties(scores)
        assert n_ties == 4  # Two pairs of ties

    def test_empty(self):
        n_ties, groups = detect_criterion_ties(np.array([]))
        assert n_ties == 0
        assert len(groups) == 0

    def test_custom_tolerance(self):
        scores = np.array([1.0, 1.05, 2.0])
        # Default tolerance (1e-8) — these are not tied
        n_ties_strict, _ = detect_criterion_ties(scores, tie_tolerance=1e-8)
        assert n_ties_strict == 0
        # Loose tolerance — first two are tied
        n_ties_loose, _ = detect_criterion_ties(scores, tie_tolerance=0.1)
        assert n_ties_loose == 2


class TestEstimateSolutionQuality:
    def test_optimal(self):
        ratio, is_opt = estimate_solution_quality(100.0, 100.0)
        assert ratio == 1.0
        assert is_opt

    def test_near_optimal(self):
        ratio, is_opt = estimate_solution_quality(99.5, 100.0)
        assert ratio == pytest.approx(0.995)
        assert is_opt

    def test_suboptimal(self):
        ratio, is_opt = estimate_solution_quality(70.0, 100.0)
        assert ratio == pytest.approx(0.7)
        assert not is_opt

    def test_zero_bound(self):
        ratio, is_opt = estimate_solution_quality(0.0, 0.0)
        assert ratio == 1.0
        assert is_opt

    def test_zero_bound_nonzero_value(self):
        ratio, is_opt = estimate_solution_quality(5.0, 0.0)
        assert ratio == 0.0
        assert not is_opt


class TestDetectRedundantFeasibility:
    def test_all_pass(self):
        history = np.array([True, True, True, True])
        rate, is_redundant = detect_redundant_feasibility(history)
        assert rate == 1.0
        assert is_redundant

    def test_some_fail(self):
        history = np.array([True, False, True, True])
        rate, is_redundant = detect_redundant_feasibility(history)
        assert rate == 0.75
        assert not is_redundant

    def test_all_fail(self):
        history = np.array([False, False, False])
        rate, is_redundant = detect_redundant_feasibility(history)
        assert rate == 0.0
        assert not is_redundant

    def test_empty(self):
        history = np.array([], dtype=bool)
        rate, is_redundant = detect_redundant_feasibility(history)
        assert rate == 1.0
        assert is_redundant


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestGreedyRules:
    def _get_rules(self):
        rs = GreedyExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_matroid_validation_applies(self):
        rules = self._get_rules()
        rule = rules["insert_matroid_validation_before_greedy_choice"]
        rw = GraphRewriter()
        cdg = _greedy_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "validate_matroid_exchange" in prims
        assert len(g.nodes) == 7  # 6 + 1

    def test_tie_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_tie_detection_after_sort"]
        rw = GraphRewriter()
        cdg = _greedy_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_criterion_ties" in prims

    def test_quality_bound_applies(self):
        rules = self._get_rules()
        rule = rules["insert_quality_bound_after_update"]
        rw = GraphRewriter()
        cdg = _greedy_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "estimate_solution_quality" in prims

    def test_redundant_feasibility_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_redundant_feasibility_detection_before_feasibility"]
        rw = GraphRewriter()
        cdg = _greedy_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_redundant_feasibility" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestGreedyDiagnostics:
    def test_diagnose_matroid_violation(self):
        rs = GreedyExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"exchange_ratio": 0.5}
        )
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_matroid_validation_before_greedy_choice" in names

    def test_high_exchange_ratio_no_trigger(self):
        rs = GreedyExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"exchange_ratio": 0.98}
        )
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ctx)
        matroid_diags = [
            d for d in diags
            if d.rule_name == "insert_matroid_validation_before_greedy_choice"
        ]
        assert len(matroid_diags) == 0

    def test_diagnose_criterion_ties(self):
        rs = GreedyExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"tie_fraction": 0.3}
        )
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_tie_detection_after_sort" in names

    def test_low_tie_fraction_no_trigger(self):
        rs = GreedyExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"tie_fraction": 0.05}
        )
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ctx)
        tie_diags = [
            d for d in diags
            if d.rule_name == "insert_tie_detection_after_sort"
        ]
        assert len(tie_diags) == 0

    def test_diagnose_solution_quality(self):
        rs = GreedyExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"greedy_value": 60.0, "relaxation_bound": 100.0}
        )
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_quality_bound_after_update" in names

    def test_good_quality_no_trigger(self):
        rs = GreedyExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"greedy_value": 95.0, "relaxation_bound": 100.0}
        )
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ctx)
        quality_diags = [
            d for d in diags
            if d.rule_name == "insert_quality_bound_after_update"
        ]
        assert len(quality_diags) == 0

    def test_diagnose_redundant_feasibility(self):
        rs = GreedyExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"feasibility_pass_rate": 1.0}
        )
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_redundant_feasibility_detection_before_feasibility" in names

    def test_failing_feasibility_no_trigger(self):
        rs = GreedyExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"feasibility_pass_rate": 0.8}
        )
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ctx)
        redundant_diags = [
            d for d in diags
            if d.rule_name == "insert_redundant_feasibility_detection_before_feasibility"
        ]
        assert len(redundant_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = GreedyExpansionRuleSet()
        cdg = _greedy_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestGreedyIntegration:
    def test_full_expansion_with_all_diagnostics(self):
        """End-to-end: diagnostics fire, engine expands greedy CDG."""
        rs = GreedyExpansionRuleSet()
        engine = ExpansionEngine([rs])

        ctx = ExpansionContext(
            intermediates={
                "exchange_ratio": 0.5,
                "tie_fraction": 0.3,
                "greedy_value": 60.0,
                "relaxation_bound": 100.0,
                "feasibility_pass_rate": 1.0,
            }
        )
        cdg = _greedy_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "validate_matroid_exchange",
            "detect_criterion_ties",
            "estimate_solution_quality",
            "detect_redundant_feasibility",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_dp_rules(self):
        """Greedy + DP rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.dynamic_programming import (
            DynamicProgrammingExpansionRuleSet,
        )

        engine = ExpansionEngine([
            GreedyExpansionRuleSet(),
            DynamicProgrammingExpansionRuleSet(),
        ])

        # Only greedy data, no DP data → only greedy diags fire
        ctx = ExpansionContext(
            intermediates={"exchange_ratio": 0.5}
        )
        cdg = _greedy_cdg()
        result = engine.expand(cdg, ctx)

        # DP rules should NOT have fired
        dp_atoms = {
            "detect_table_sparsity",
            "prune_infeasible_states",
            "compress_dp_table",
            "validate_subproblem_overlap",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & dp_atoms)
