"""Tests for the randomized expansion rules and runtime atoms."""

import numpy as np

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.expansion_atoms.runtime_randomized import (
    analyze_sketch_accuracy,
    check_concentration_bound,
    monitor_sample_coverage,
    validate_hash_independence,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.randomized import RandomizedExpansionRuleSet


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


def _randomized_cdg():
    return _cdg(
        [
            _node("src", "Source"),
            _node("gen", "Generate Samples", ConceptType.RANDOMIZED),
            _node("skh", "Sketch/Hash", ConceptType.RANDOMIZED),
            _node("est", "Estimate", ConceptType.RANDOMIZED),
            _node("out", "Output"),
        ],
        [_edge("src", "gen"), _edge("gen", "skh"), _edge("skh", "est"), _edge("est", "out")],
    )


def test_hash_independence():
    ratio, ok = validate_hash_independence(2.0, 2.0)
    assert ok
    ratio, ok = validate_hash_independence(10.0, 2.0)
    assert ratio > 2.0
    assert not ok


def test_sketch_accuracy():
    error, ok = analyze_sketch_accuracy(np.array([10.0, 20.0]), np.array([10.5, 19.5]))
    assert ok
    error, ok = analyze_sketch_accuracy(np.array([10.0, 20.0]), np.array([20.0, 40.0]))
    assert error > 0.1
    assert not ok


def test_sample_coverage():
    coverage, ok = monitor_sample_coverage(np.array([1, 2, 3]), 10)
    assert ok
    coverage, ok = monitor_sample_coverage(np.array([], dtype=int), 10)
    assert coverage == 0.0
    assert not ok


def test_concentration():
    rate, ok = check_concentration_bound(np.array([0.01, 0.02, 0.03]), 0.05)
    assert ok
    rate, ok = check_concentration_bound(np.array([0.1, 0.2, 0.01]), 0.05)
    assert rate > 0.05
    assert not ok


def test_randomized_rules_apply():
    rules = {r.name: r for r in RandomizedExpansionRuleSet().rules()}
    assert not GraphRewriter().apply_rule(rules["insert_sample_coverage_monitor_after_generate"], _randomized_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_hash_independence_validation_after_sketch"], _randomized_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_sketch_accuracy_analysis_before_estimate"], _randomized_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_concentration_check_after_estimate"], _randomized_cdg()).is_failure


def test_randomized_diagnostics():
    diags = RandomizedExpansionRuleSet().diagnose(
        _randomized_cdg(),
        ExpansionContext(
            intermediates={
                "sample_coverage": 0.01,
                "collision_ratio": 4.0,
                "sketch_relative_error": 0.2,
                "concentration_violation_rate": 0.2,
            }
        ),
    )
    rule_names = {d.rule_name for d in diags}
    assert "insert_sample_coverage_monitor_after_generate" in rule_names
    assert "insert_hash_independence_validation_after_sketch" in rule_names
    assert "insert_sketch_accuracy_analysis_before_estimate" in rule_names
    assert "insert_concentration_check_after_estimate" in rule_names


def test_randomized_engine_expands():
    result = ExpansionEngine([RandomizedExpansionRuleSet()]).expand(
        _randomized_cdg(),
        ExpansionContext(intermediates={"sample_coverage": 0.01, "collision_ratio": 4.0}),
    )
    assert result.expanded
