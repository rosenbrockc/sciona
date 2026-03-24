"""Tests for the quadrature expansion rules and runtime atoms."""

import numpy as np

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.expansion_atoms.runtime_quadrature import (
    analyze_integrand_smoothness,
    check_domain_coverage,
    detect_singularity,
    monitor_convergence_rate,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.quadrature import QuadratureExpansionRuleSet


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


def _quadrature_cdg():
    return _cdg(
        [
            _node("src", "Source"),
            _node("smp", "Sample Points", ConceptType.QUADRATURE),
            _node("evl", "Evaluate Integrand", ConceptType.QUADRATURE),
            _node("ref", "Estimate Error/Refine", ConceptType.QUADRATURE),
            _node("out", "Output"),
        ],
        [_edge("src", "smp"), _edge("smp", "evl"), _edge("evl", "ref"), _edge("ref", "out")],
    )


def test_smoothness():
    deriv, ok = analyze_integrand_smoothness(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    assert ok
    deriv, ok = analyze_integrand_smoothness(np.array([0.0, 1e7]), np.array([0.0, 1.0]))
    assert deriv > 1e6
    assert not ok


def test_singularity():
    maximum, ok = detect_singularity(np.array([1.0, 2.0]))
    assert ok
    maximum, ok = detect_singularity(np.array([1e11]))
    assert maximum > 1e10
    assert not ok


def test_convergence_rate():
    rate, ok = monitor_convergence_rate(np.array([1.0, 0.25, 0.0625]))
    assert ok
    rate, ok = monitor_convergence_rate(np.array([1.0, 0.8, 0.7]))
    assert rate >= 0.5
    assert not ok


def test_domain_coverage():
    gap, ok = check_domain_coverage(np.linspace(0.0, 1.0, 11), np.array([0.0, 1.0]))
    assert ok
    gap, ok = check_domain_coverage(np.array([0.0, 0.9]), np.array([0.0, 1.0]))
    assert gap > 0.1
    assert not ok


def test_quadrature_rules_apply():
    rules = {r.name: r for r in QuadratureExpansionRuleSet().rules()}
    assert not GraphRewriter().apply_rule(rules["insert_domain_coverage_check_after_sample"], _quadrature_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_integrand_smoothness_analysis_before_evaluate"], _quadrature_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_singularity_detection_after_evaluate"], _quadrature_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_convergence_monitor_after_refine"], _quadrature_cdg()).is_failure


def test_quadrature_diagnostics():
    diags = QuadratureExpansionRuleSet().diagnose(
        _quadrature_cdg(),
        ExpansionContext(
            intermediates={
                "max_gap_ratio": 0.25,
                "integrand_max_derivative": 1e7,
                "integrand_max_value": 1e11,
                "quadrature_convergence_rate": 0.8,
            }
        ),
    )
    rule_names = {d.rule_name for d in diags}
    assert "insert_domain_coverage_check_after_sample" in rule_names
    assert "insert_integrand_smoothness_analysis_before_evaluate" in rule_names
    assert "insert_singularity_detection_after_evaluate" in rule_names
    assert "insert_convergence_monitor_after_refine" in rule_names


def test_quadrature_engine_expands():
    result = ExpansionEngine([QuadratureExpansionRuleSet()]).expand(
        _quadrature_cdg(),
        ExpansionContext(intermediates={"max_gap_ratio": 0.25, "integrand_max_value": 1e11}),
    )
    assert result.expanded
