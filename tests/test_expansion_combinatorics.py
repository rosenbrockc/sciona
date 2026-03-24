"""Tests for the Combinatorial Optimization expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.combinatorics import CombinatoricsExpansionRuleSet
from sciona.expansion_atoms.runtime_combinatorics import (
    analyze_branching_factor, monitor_bound_tightness,
    detect_symmetry, check_pruning_effectiveness,
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

def _combinatorics_cdg():
    return _cdg(
        [_node("src", "Source"),
         _node("bnd", "Bound", ConceptType.COMBINATORICS),
         _node("bra", "Branch", ConceptType.COMBINATORICS),
         _node("pru", "Prune", ConceptType.COMBINATORICS),
         _node("sel", "Select", ConceptType.COMBINATORICS),
         _node("out", "Output")],
        [_edge("src", "bnd"), _edge("bnd", "bra"), _edge("bra", "pru"),
         _edge("pru", "sel"), _edge("sel", "out")],
    )


class TestAnalyzeBranchingFactor:
    def test_manageable(self):
        counts = np.array([2, 3, 2, 4, 3], dtype=float)
        mean, ok = analyze_branching_factor(counts)
        assert ok
        assert mean < 10

    def test_explosive(self):
        counts = np.array([20, 30, 25, 15], dtype=float)
        mean, ok = analyze_branching_factor(counts)
        assert not ok
        assert mean > 10

    def test_empty(self):
        mean, ok = analyze_branching_factor(np.array([]))
        assert ok


class TestMonitorBoundTightness:
    def test_tight(self):
        ub = np.array([10.0, 8.0, 6.5])
        lb = np.array([5.0, 5.5, 6.0])
        gap, tight = monitor_bound_tightness(ub, lb)
        assert tight

    def test_loose(self):
        ub = np.array([100.0, 90.0])
        lb = np.array([1.0, 2.0])
        gap, tight = monitor_bound_tightness(ub, lb)
        assert not tight

    def test_empty(self):
        gap, tight = monitor_bound_tightness(np.array([]), np.array([]))
        assert tight


class TestDetectSymmetry:
    def test_no_symmetry(self):
        pairs = np.arange(100, dtype=float)
        frac, has = detect_symmetry(pairs, 5)
        assert not has

    def test_high_symmetry(self):
        pairs = np.arange(10, dtype=float)
        frac, has = detect_symmetry(pairs, 5)
        assert has
        assert frac > 0.3

    def test_empty(self):
        frac, has = detect_symmetry(np.array([]), 0)
        assert not has


class TestCheckPruningEffectiveness:
    def test_effective(self):
        rate, ok = check_pruning_effectiveness(100, 50)
        assert ok
        assert rate == 0.5

    def test_ineffective(self):
        rate, ok = check_pruning_effectiveness(1000, 5)
        assert not ok
        assert rate < 0.1

    def test_zero_total(self):
        rate, ok = check_pruning_effectiveness(0, 0)
        assert not ok


class TestCombinatoricsRules:
    def _get_rules(self):
        return {r.name: r for r in CombinatoricsExpansionRuleSet().rules()}

    def test_branching_factor_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_branching_factor_analysis_after_branch"], _combinatorics_cdg())
        assert not result.is_failure
        assert "analyze_branching_factor" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_bound_tightness_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_bound_tightness_monitoring_after_bound"], _combinatorics_cdg())
        assert not result.is_failure

    def test_symmetry_detection_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_symmetry_detection_before_branch"], _combinatorics_cdg())
        assert not result.is_failure

    def test_pruning_effectiveness_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_pruning_effectiveness_after_prune"], _combinatorics_cdg())
        assert not result.is_failure


class TestCombinatoricsDiagnostics:
    def test_diagnose_branching_factor(self):
        diags = CombinatoricsExpansionRuleSet().diagnose(_combinatorics_cdg(), ExpansionContext(intermediates={"mean_branching_factor": 15.0}))
        assert "insert_branching_factor_analysis_after_branch" in {d.rule_name for d in diags}

    def test_manageable_no_trigger(self):
        diags = CombinatoricsExpansionRuleSet().diagnose(_combinatorics_cdg(), ExpansionContext(intermediates={"mean_branching_factor": 3.0}))
        assert not [d for d in diags if d.rule_name == "insert_branching_factor_analysis_after_branch"]

    def test_diagnose_bound_tightness(self):
        diags = CombinatoricsExpansionRuleSet().diagnose(_combinatorics_cdg(), ExpansionContext(intermediates={"bound_gap_ratio": 0.8}))
        assert "insert_bound_tightness_monitoring_after_bound" in {d.rule_name for d in diags}

    def test_diagnose_symmetry(self):
        diags = CombinatoricsExpansionRuleSet().diagnose(_combinatorics_cdg(), ExpansionContext(intermediates={"symmetry_fraction": 0.5}))
        assert "insert_symmetry_detection_before_branch" in {d.rule_name for d in diags}

    def test_diagnose_pruning(self):
        diags = CombinatoricsExpansionRuleSet().diagnose(_combinatorics_cdg(), ExpansionContext(intermediates={"pruning_rate": 0.02}))
        assert "insert_pruning_effectiveness_after_prune" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert CombinatoricsExpansionRuleSet().diagnose(_combinatorics_cdg(), ExpansionContext()) == []


class TestCombinatoricsIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([CombinatoricsExpansionRuleSet()]).expand(
            _combinatorics_cdg(), ExpansionContext(intermediates={"mean_branching_factor": 20.0, "pruning_rate": 0.01}))
        assert result.expanded
