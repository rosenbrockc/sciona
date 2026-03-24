"""Tests for the Linear Algebra expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.linear_algebra import LinearAlgebraExpansionRuleSet
from sciona.expansion_atoms.runtime_linear_algebra import (
    check_matrix_conditioning, validate_decomposition_accuracy,
    detect_rank_deficiency, monitor_iterative_convergence,
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

def _linear_algebra_cdg():
    return _cdg(
        [_node("src", "Source"),
         _node("fac", "Factorize", ConceptType.ALGEBRA),
         _node("sol", "Solve/Transform", ConceptType.ALGEBRA),
         _node("val", "Validate", ConceptType.ALGEBRA),
         _node("out", "Output")],
        [_edge("src", "fac"), _edge("fac", "sol"), _edge("sol", "val"), _edge("val", "out")],
    )


class TestCheckMatrixConditioning:
    def test_well_conditioned(self):
        A = np.eye(3)
        cond, ok = check_matrix_conditioning(A)
        assert ok
        assert cond == pytest.approx(1.0)

    def test_ill_conditioned(self):
        A = np.diag([1.0, 1e-14])
        cond, ok = check_matrix_conditioning(A)
        assert not ok
        assert cond > 1e12

    def test_empty(self):
        cond, ok = check_matrix_conditioning(np.array([]).reshape(0, 0))
        assert ok


class TestValidateDecompositionAccuracy:
    def test_exact(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        err, ok = validate_decomposition_accuracy(A, A)
        assert ok
        assert err == pytest.approx(0.0, abs=1e-15)

    def test_inaccurate(self):
        A = np.eye(3)
        bad = A + np.ones((3, 3))
        err, ok = validate_decomposition_accuracy(A, bad)
        assert not ok

    def test_empty(self):
        err, ok = validate_decomposition_accuracy(np.array([]), np.array([]))
        assert ok


class TestDetectRankDeficiency:
    def test_full_rank(self):
        sv = np.array([5.0, 3.0, 1.0])
        rank, ok = detect_rank_deficiency(sv, 3)
        assert ok
        assert rank == 3

    def test_rank_deficient(self):
        sv = np.array([5.0, 3.0, 1e-15])
        rank, ok = detect_rank_deficiency(sv, 3)
        assert not ok
        assert rank == 2

    def test_empty(self):
        rank, ok = detect_rank_deficiency(np.array([]), 0)
        assert ok


class TestMonitorIterativeConvergence:
    def test_converging(self):
        norms = np.array([100, 50, 25, 12.5, 6.25], dtype=float)
        rate, ok = monitor_iterative_convergence(norms)
        assert ok
        assert rate < 0.99

    def test_stalling(self):
        norms = np.array([100, 99.5, 99.0, 98.6, 98.2], dtype=float)
        rate, ok = monitor_iterative_convergence(norms)
        assert not ok
        assert rate >= 0.99

    def test_single(self):
        rate, ok = monitor_iterative_convergence(np.array([1.0]))
        assert ok


class TestLinearAlgebraRules:
    def _get_rules(self):
        return {r.name: r for r in LinearAlgebraExpansionRuleSet().rules()}

    def test_conditioning_check_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_conditioning_check_before_factorize"], _linear_algebra_cdg())
        assert not result.is_failure
        assert "check_matrix_conditioning" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_decomposition_accuracy_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_decomposition_accuracy_after_factorize"], _linear_algebra_cdg())
        assert not result.is_failure

    def test_rank_deficiency_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_rank_deficiency_detection_before_solve"], _linear_algebra_cdg())
        assert not result.is_failure

    def test_iterative_convergence_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_iterative_convergence_after_solve"], _linear_algebra_cdg())
        assert not result.is_failure


class TestLinearAlgebraDiagnostics:
    def test_diagnose_ill_conditioned(self):
        diags = LinearAlgebraExpansionRuleSet().diagnose(_linear_algebra_cdg(), ExpansionContext(intermediates={"matrix_condition_number": 1e14}))
        assert "insert_conditioning_check_before_factorize" in {d.rule_name for d in diags}

    def test_well_conditioned_no_trigger(self):
        diags = LinearAlgebraExpansionRuleSet().diagnose(_linear_algebra_cdg(), ExpansionContext(intermediates={"matrix_condition_number": 100.0}))
        assert not [d for d in diags if d.rule_name == "insert_conditioning_check_before_factorize"]

    def test_diagnose_decomposition_residual(self):
        diags = LinearAlgebraExpansionRuleSet().diagnose(_linear_algebra_cdg(), ExpansionContext(intermediates={"decomposition_residual": 1e-5}))
        assert "insert_decomposition_accuracy_after_factorize" in {d.rule_name for d in diags}

    def test_diagnose_rank_deficit(self):
        diags = LinearAlgebraExpansionRuleSet().diagnose(_linear_algebra_cdg(), ExpansionContext(intermediates={"rank_deficit": 2}))
        assert "insert_rank_deficiency_detection_before_solve" in {d.rule_name for d in diags}

    def test_diagnose_iterative_convergence(self):
        diags = LinearAlgebraExpansionRuleSet().diagnose(_linear_algebra_cdg(), ExpansionContext(intermediates={"iterative_convergence_rate": 0.995}))
        assert "insert_iterative_convergence_after_solve" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert LinearAlgebraExpansionRuleSet().diagnose(_linear_algebra_cdg(), ExpansionContext()) == []


class TestLinearAlgebraIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([LinearAlgebraExpansionRuleSet()]).expand(
            _linear_algebra_cdg(), ExpansionContext(intermediates={"matrix_condition_number": 1e14, "rank_deficit": 1}))
        assert result.expanded
