"""Tests for the Information Theory expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.information_theory import InformationTheoryExpansionRuleSet
from sciona.expansion_atoms.runtime_information_theory import (
    analyze_sample_sufficiency,
    check_distribution_support,
    detect_numerical_underflow,
    validate_information_inequality,
)


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
    return DependencyEdge(source_id=src, target_id=tgt, output_name="out", input_name="in", source_type="ndarray", target_type="ndarray")


def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})


def _information_theory_cdg():
    return _cdg(
        [
            _node("src", "Source"),
            _node("est", "Estimate Distribution", ConceptType.INFORMATION_THEORY),
            _node("cmp", "Compute Entropy/Divergence", ConceptType.INFORMATION_THEORY),
            _node("val", "Validate Bounds", ConceptType.INFORMATION_THEORY),
            _node("out", "Output"),
        ],
        [_edge("src", "est"), _edge("est", "cmp"), _edge("cmp", "val"), _edge("val", "out")],
    )


class TestCheckDistributionSupport:
    def test_full_support(self):
        frac, ok = check_distribution_support(np.array([0.2, 0.3, 0.5]))
        assert ok
        assert frac == 0.0

    def test_missing_support(self):
        frac, ok = check_distribution_support(np.array([0.2, 0.0, 0.8]))
        assert not ok
        assert frac > 0.0

    def test_empty(self):
        frac, ok = check_distribution_support(np.array([]))
        assert ok


class TestAnalyzeSampleSufficiency:
    def test_sufficient(self):
        sps, ok = analyze_sample_sufficiency(100, 10)
        assert ok
        assert sps == 10.0

    def test_insufficient(self):
        sps, ok = analyze_sample_sufficiency(10, 10)
        assert not ok
        assert sps < 5.0

    def test_zero_support(self):
        sps, ok = analyze_sample_sufficiency(0, 0)
        assert not ok


class TestDetectNumericalUnderflow:
    def test_stable(self):
        frac, ok = detect_numerical_underflow(np.array([-1.0, -2.0, -3.0]))
        assert ok
        assert frac == 0.0

    def test_underflow(self):
        frac, ok = detect_numerical_underflow(np.array([-1000.0, -10.0]))
        assert not ok
        assert frac > 0.05

    def test_empty(self):
        frac, ok = detect_numerical_underflow(np.array([]))
        assert ok


class TestValidateInformationInequality:
    def test_holds(self):
        violation, ok = validate_information_inequality(np.array([0.4, 0.2]), np.array([0.5, 0.3]))
        assert ok
        assert violation == 0.0

    def test_violated(self):
        violation, ok = validate_information_inequality(np.array([0.6, 0.2]), np.array([0.5, 0.1]))
        assert not ok
        assert violation > 0.0

    def test_empty(self):
        violation, ok = validate_information_inequality(np.array([]), np.array([]))
        assert ok


class TestInformationTheoryRules:
    def _get_rules(self):
        return {r.name: r for r in InformationTheoryExpansionRuleSet().rules()}

    def test_distribution_support_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_distribution_support_check_before_compute"],
            _information_theory_cdg(),
        )
        assert not result.is_failure
        assert "check_distribution_support" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_sample_sufficiency_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_sample_sufficiency_analysis_before_compute"],
            _information_theory_cdg(),
        )
        assert not result.is_failure

    def test_underflow_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_numerical_underflow_detection_after_compute"],
            _information_theory_cdg(),
        )
        assert not result.is_failure

    def test_information_inequality_applies(self):
        result = GraphRewriter().apply_rule(
            self._get_rules()["insert_information_inequality_validation_after_validate"],
            _information_theory_cdg(),
        )
        assert not result.is_failure


class TestInformationTheoryDiagnostics:
    def test_diagnose_support(self):
        diags = InformationTheoryExpansionRuleSet().diagnose(
            _information_theory_cdg(),
            ExpansionContext(intermediates={"zero_mass_fraction": 0.1}),
        )
        assert "insert_distribution_support_check_before_compute" in {d.rule_name for d in diags}

    def test_diagnose_samples(self):
        diags = InformationTheoryExpansionRuleSet().diagnose(
            _information_theory_cdg(),
            ExpansionContext(intermediates={"samples_per_symbol": 2.0}),
        )
        assert "insert_sample_sufficiency_analysis_before_compute" in {d.rule_name for d in diags}

    def test_diagnose_underflow(self):
        diags = InformationTheoryExpansionRuleSet().diagnose(
            _information_theory_cdg(),
            ExpansionContext(intermediates={"underflow_fraction": 0.2}),
        )
        assert "insert_numerical_underflow_detection_after_compute" in {d.rule_name for d in diags}

    def test_diagnose_inequality(self):
        diags = InformationTheoryExpansionRuleSet().diagnose(
            _information_theory_cdg(),
            ExpansionContext(intermediates={"max_information_inequality_violation": 1e-6}),
        )
        assert "insert_information_inequality_validation_after_validate" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert InformationTheoryExpansionRuleSet().diagnose(_information_theory_cdg(), ExpansionContext()) == []


class TestInformationTheoryIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([InformationTheoryExpansionRuleSet()]).expand(
            _information_theory_cdg(),
            ExpansionContext(intermediates={"zero_mass_fraction": 0.1, "underflow_fraction": 0.2}),
        )
        assert result.expanded
