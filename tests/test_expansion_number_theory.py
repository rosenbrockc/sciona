"""Tests for the Number Theory expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.number_theory import NumberTheoryExpansionRuleSet
from sciona.expansion_atoms.runtime_number_theory import (
    validate_input_range, monitor_gcd_convergence,
    check_small_prime_divisors, detect_modular_overflow,
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

def _number_theory_cdg():
    return _cdg(
        [_node("src", "Source"), _node("rd", "Reduce", ConceptType.NUMBER_THEORY),
         _node("it", "Iterate", ConceptType.NUMBER_THEORY), _node("cl", "Conclude", ConceptType.NUMBER_THEORY),
         _node("out", "Output")],
        [_edge("src", "rd"), _edge("rd", "it"), _edge("it", "cl"), _edge("cl", "out")],
    )


class TestValidateInputRange:
    def test_safe_values(self):
        vals = np.array([100, 200, 300])
        n_risk, safe = validate_input_range(vals)
        assert n_risk == 0
        assert safe

    def test_overflow_risk(self):
        vals = np.array([2**33, 100])
        n_risk, safe = validate_input_range(vals)
        assert n_risk == 1
        assert not safe

    def test_empty(self):
        n_risk, safe = validate_input_range(np.array([], dtype=np.int64))
        assert n_risk == 0
        assert safe


class TestMonitorGcdConvergence:
    def test_fast_convergence(self):
        rems = np.array([100, 30, 10, 0])
        n_steps, ratio = monitor_gcd_convergence(rems)
        assert ratio < 0.618

    def test_slow_convergence(self):
        # Fibonacci-like: each remainder is ~0.618 of previous
        rems = np.array([89, 55, 34, 21, 13, 8, 5, 3, 2, 1], dtype=float)
        n_steps, ratio = monitor_gcd_convergence(rems)
        assert ratio > 0.5

    def test_single_step(self):
        n_steps, ratio = monitor_gcd_convergence(np.array([10]))
        assert n_steps == 1


class TestCheckSmallPrimeDivisors:
    def test_composite(self):
        has_factor, factor = check_small_prime_divisors(100)
        assert has_factor
        assert factor == 2

    def test_prime(self):
        has_factor, factor = check_small_prime_divisors(97)
        assert not has_factor
        assert factor == 0

    def test_small_number(self):
        has_factor, factor = check_small_prime_divisors(1)
        assert has_factor


class TestDetectModularOverflow:
    def test_no_overflow(self):
        would, bits = detect_modular_overflow(100, 50, 1000)
        assert not would

    def test_overflow(self):
        would, bits = detect_modular_overflow(2**40, 100, 2**40)
        assert would

    def test_zero_values(self):
        would, bits = detect_modular_overflow(0, 10, 100)
        assert not would


class TestNumberTheoryRules:
    def _get_rules(self):
        return {r.name: r for r in NumberTheoryExpansionRuleSet().rules()}

    def test_input_range_validation_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_input_range_validation_before_reduce"], _number_theory_cdg())
        assert not result.is_failure
        assert "validate_input_range" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_gcd_convergence_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_gcd_convergence_monitoring_after_iterate"], _number_theory_cdg())
        assert not result.is_failure

    def test_small_prime_check_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_small_prime_check_before_conclude"], _number_theory_cdg())
        assert not result.is_failure

    def test_modular_overflow_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_modular_overflow_detection_before_reduce"], _number_theory_cdg())
        assert not result.is_failure


class TestNumberTheoryDiagnostics:
    def test_diagnose_input_range(self):
        diags = NumberTheoryExpansionRuleSet().diagnose(_number_theory_cdg(), ExpansionContext(intermediates={"input_overflow_risk_count": 3}))
        assert "insert_input_range_validation_before_reduce" in {d.rule_name for d in diags}

    def test_safe_range_no_trigger(self):
        diags = NumberTheoryExpansionRuleSet().diagnose(_number_theory_cdg(), ExpansionContext(intermediates={"input_overflow_risk_count": 0}))
        assert not [d for d in diags if d.rule_name == "insert_input_range_validation_before_reduce"]

    def test_diagnose_slow_gcd(self):
        diags = NumberTheoryExpansionRuleSet().diagnose(_number_theory_cdg(), ExpansionContext(intermediates={"gcd_reduction_ratio": 0.7}))
        assert "insert_gcd_convergence_monitoring_after_iterate" in {d.rule_name for d in diags}

    def test_fast_gcd_no_trigger(self):
        diags = NumberTheoryExpansionRuleSet().diagnose(_number_theory_cdg(), ExpansionContext(intermediates={"gcd_reduction_ratio": 0.3}))
        assert not [d for d in diags if d.rule_name == "insert_gcd_convergence_monitoring_after_iterate"]

    def test_diagnose_composite(self):
        diags = NumberTheoryExpansionRuleSet().diagnose(_number_theory_cdg(), ExpansionContext(intermediates={"has_small_factor": True}))
        assert "insert_small_prime_check_before_conclude" in {d.rule_name for d in diags}

    def test_diagnose_modular_overflow(self):
        diags = NumberTheoryExpansionRuleSet().diagnose(_number_theory_cdg(), ExpansionContext(intermediates={"modular_overflow": True}))
        assert "insert_modular_overflow_detection_before_reduce" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert NumberTheoryExpansionRuleSet().diagnose(_number_theory_cdg(), ExpansionContext()) == []


class TestNumberTheoryIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([NumberTheoryExpansionRuleSet()]).expand(
            _number_theory_cdg(), ExpansionContext(intermediates={"input_overflow_risk_count": 2, "modular_overflow": True}))
        assert result.expanded
