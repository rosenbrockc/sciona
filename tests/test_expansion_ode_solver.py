"""Tests for the ODE solver expansion rules and runtime atoms."""

import numpy as np

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.expansion_atoms.runtime_ode_solver import (
    check_energy_conservation,
    detect_stiffness,
    monitor_step_rejection_rate,
    validate_order_of_accuracy,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.ode_solver import ODESolverExpansionRuleSet


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


def _ode_cdg():
    return _cdg(
        [
            _node("src", "Source"),
            _node("eval", "Evaluate Derivative", ConceptType.ODE_SOLVER),
            _node("adv", "Advance State", ConceptType.ODE_SOLVER),
            _node("err", "Estimate Error", ConceptType.ODE_SOLVER),
            _node("adp", "Adapt Step Size", ConceptType.ODE_SOLVER),
            _node("out", "Output"),
        ],
        [_edge("src", "eval"), _edge("eval", "adv"), _edge("adv", "err"), _edge("err", "adp"), _edge("adp", "out")],
    )


def test_step_rejection_rate():
    rate, ok = monitor_step_rejection_rate(np.array([1, 1, 0, 0]))
    assert rate == 0.5
    assert ok
    rate, ok = monitor_step_rejection_rate(np.array([0, 0, 0, 1]))
    assert rate > 0.5
    assert not ok


def test_detect_stiffness():
    ratio, stiff = detect_stiffness(np.array([1.0, 1e8]))
    assert ratio > 1e6
    assert stiff
    ratio, stiff = detect_stiffness(np.array([1.0, 10.0]))
    assert not stiff


def test_energy_conservation():
    drift, ok = check_energy_conservation(np.array([1.0, 1.0 + 1e-8]))
    assert ok
    drift, ok = check_energy_conservation(np.array([1.0, 1.1]))
    assert drift > 1e-6
    assert not ok


def test_order_validation():
    order, ok = validate_order_of_accuracy(np.array([1e-1, 2.5e-2, 6.25e-3]), np.array([0.4, 0.2, 0.1]), 2.0)
    assert order > 1.5
    assert ok
    order, ok = validate_order_of_accuracy(np.array([1e-1, 5e-2, 2.5e-2]), np.array([0.4, 0.2, 0.1]), 2.0)
    assert order < 1.6
    assert not ok


def test_ode_rules_apply():
    rules = {r.name: r for r in ODESolverExpansionRuleSet().rules()}
    assert not GraphRewriter().apply_rule(rules["insert_stiffness_detection_before_advance"], _ode_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_energy_conservation_check_after_advance"], _ode_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_order_validation_after_estimate_error"], _ode_cdg()).is_failure
    assert not GraphRewriter().apply_rule(rules["insert_step_rejection_monitor_after_adapt"], _ode_cdg()).is_failure


def test_ode_diagnostics():
    diags = ODESolverExpansionRuleSet().diagnose(
        _ode_cdg(),
        ExpansionContext(
            intermediates={
                "stiffness_ratio": 1e8,
                "energy_drift": 1e-4,
                "empirical_convergence_order": 0.4,
                "step_rejection_rate": 0.75,
            }
        ),
    )
    rule_names = {d.rule_name for d in diags}
    assert "insert_stiffness_detection_before_advance" in rule_names
    assert "insert_energy_conservation_check_after_advance" in rule_names
    assert "insert_order_validation_after_estimate_error" in rule_names
    assert "insert_step_rejection_monitor_after_adapt" in rule_names


def test_ode_engine_expands():
    result = ExpansionEngine([ODESolverExpansionRuleSet()]).expand(
        _ode_cdg(),
        ExpansionContext(intermediates={"stiffness_ratio": 1e8, "step_rejection_rate": 0.75}),
    )
    assert result.expanded
