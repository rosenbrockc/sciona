"""Expansion rules for the ODE Solver family."""

from __future__ import annotations

import logging

import numpy as np

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionDiagnostic

logger = logging.getLogger(__name__)

_DOMAIN = "ode_solver"

_EVALUATE_DERIVATIVE = "Evaluate Derivative"
_ADVANCE_STATE = "Advance State"
_ESTIMATE_ERROR = "Estimate Error"
_ADAPT_STEP_SIZE = "Adapt Step Size"


def _node(node_id: str, name: str, concept_type: ConceptType, *, matched_primitive: str | None = None,
          inputs: list[IOSpec] | None = None, outputs: list[IOSpec] | None = None,
          description: str = "", type_signature: str = "") -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=description or name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
        inputs=inputs or [],
        outputs=outputs or [],
        type_signature=type_signature or f"{name} -> result",
    )


def _edge(source_id: str, target_id: str, output_name: str = "out", input_name: str = "in",
          type_desc: str = "ndarray") -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name=output_name,
        input_name=input_name,
        source_type=type_desc,
        target_type=type_desc,
    )


def _build_insert_stiffness_detection() -> RewriteRule:
    src = _node("src", _EVALUATE_DERIVATIVE, ConceptType.ODE_SOLVER)
    advance = _node("advance", _ADVANCE_STATE, ConceptType.ODE_SOLVER)
    lhs = CDGExport(nodes=[src, advance], edges=[_edge("src", "advance")])
    interface = CDGExport(nodes=[src, advance], edges=[])
    detect = _node(
        "stiffness",
        "Detect Stiffness",
        ConceptType.ODE_SOLVER,
        matched_primitive="detect_stiffness",
        inputs=[IOSpec(name="jacobian_eigenvalues", type_desc="ndarray")],
        outputs=[IOSpec(name="stiffness_ratio", type_desc="float"), IOSpec(name="is_stiff", type_desc="bool")],
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[src, detect, advance], edges=[_edge("src", "stiffness"), _edge("stiffness", "advance")])
    return RewriteRule(
        name="insert_stiffness_detection_before_advance",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "advance": "advance"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "advance": "advance"}, edge_map={}),
        priority=3,
    )


def _build_insert_energy_conservation_check() -> RewriteRule:
    advance = _node("advance", _ADVANCE_STATE, ConceptType.ODE_SOLVER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[advance, sink], edges=[_edge("advance", "sink")])
    interface = CDGExport(nodes=[advance, sink], edges=[])
    energy = _node(
        "energy",
        "Check Energy Conservation",
        ConceptType.ODE_SOLVER,
        matched_primitive="check_energy_conservation",
        inputs=[IOSpec(name="energy_values", type_desc="ndarray")],
        outputs=[IOSpec(name="energy_drift", type_desc="float"), IOSpec(name="is_conserved", type_desc="bool")],
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[advance, energy, sink], edges=[_edge("advance", "energy"), _edge("energy", "sink")])
    return RewriteRule(
        name="insert_energy_conservation_check_after_advance",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"advance": "advance", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"advance": "advance", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_order_validation() -> RewriteRule:
    estimate = _node("estimate", _ESTIMATE_ERROR, ConceptType.ODE_SOLVER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[estimate, sink], edges=[_edge("estimate", "sink")])
    interface = CDGExport(nodes=[estimate, sink], edges=[])
    order = _node(
        "order",
        "Validate Order Of Accuracy",
        ConceptType.ODE_SOLVER,
        matched_primitive="validate_order_of_accuracy",
        inputs=[
            IOSpec(name="errors", type_desc="ndarray"),
            IOSpec(name="step_sizes", type_desc="ndarray"),
            IOSpec(name="expected_order", type_desc="float"),
        ],
        outputs=[IOSpec(name="empirical_order", type_desc="float"), IOSpec(name="order_ok", type_desc="bool")],
        type_signature="ndarray, ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[estimate, order, sink], edges=[_edge("estimate", "order"), _edge("order", "sink")])
    return RewriteRule(
        name="insert_order_validation_after_estimate_error",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"estimate": "estimate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"estimate": "estimate", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_step_rejection_monitor() -> RewriteRule:
    adapt = _node("adapt", _ADAPT_STEP_SIZE, ConceptType.ODE_SOLVER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[adapt, sink], edges=[_edge("adapt", "sink")])
    interface = CDGExport(nodes=[adapt, sink], edges=[])
    reject = _node(
        "reject",
        "Monitor Step Rejection Rate",
        ConceptType.ODE_SOLVER,
        matched_primitive="monitor_step_rejection_rate",
        inputs=[IOSpec(name="accepted", type_desc="ndarray")],
        outputs=[IOSpec(name="rejection_rate", type_desc="float"), IOSpec(name="is_excessive", type_desc="bool")],
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[adapt, reject, sink], edges=[_edge("adapt", "reject"), _edge("reject", "sink")])
    return RewriteRule(
        name="insert_step_rejection_monitor_after_adapt",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"adapt": "adapt", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"adapt": "adapt", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _diagnose_stiffness(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("stiffness_ratio")
    if value is None:
        return None
    try:
        ratio = float(value)
    except (ValueError, TypeError):
        return None
    if ratio > 1e6:
        return ExpansionDiagnostic(
            rule_name="insert_stiffness_detection_before_advance",
            severity=max(0.35, min(1.0, np.log10(max(ratio, 1.0)) / 8.0)),
            evidence=f"Stiffness ratio {ratio:.2e} exceeds 1e6.",
            metric_name="stiffness_ratio",
            metric_value=ratio,
            threshold=1e6,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_energy(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("energy_drift")
    if value is None:
        return None
    try:
        drift = float(value)
    except (ValueError, TypeError):
        return None
    if drift > 1e-6:
        return ExpansionDiagnostic(
            rule_name="insert_energy_conservation_check_after_advance",
            severity=max(0.35, min(1.0, drift / 1e-4)),
            evidence=f"Energy drift {drift:.2e} exceeds 1e-6.",
            metric_name="energy_drift",
            metric_value=drift,
            threshold=1e-6,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_order(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("empirical_convergence_order")
    if value is None:
        return None
    try:
        order = float(value)
    except (ValueError, TypeError):
        return None
    if order < 0.8:
        return ExpansionDiagnostic(
            rule_name="insert_order_validation_after_estimate_error",
            severity=max(0.35, min(1.0, (0.8 - order) / 0.8)),
            evidence=f"Empirical convergence order {order:.3f} is below 0.8.",
            metric_name="empirical_convergence_order",
            metric_value=order,
            threshold=0.8,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_rejection(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("step_rejection_rate")
    if value is None:
        return None
    try:
        rate = float(value)
    except (ValueError, TypeError):
        return None
    if rate > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_step_rejection_monitor_after_adapt",
            severity=max(0.35, min(1.0, rate)),
            evidence=f"Step rejection rate {rate:.3f} exceeds 0.5.",
            metric_name="step_rejection_rate",
            metric_value=rate,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


class ODESolverExpansionRuleSet:
    @property
    def name(self) -> str:
        return "ode_solver"

    @property
    def domain(self) -> str:
        return _DOMAIN

    def rules(self) -> list[RewriteRule]:
        return [
            _build_insert_stiffness_detection(),
            _build_insert_energy_conservation_check(),
            _build_insert_order_validation(),
            _build_insert_step_rejection_monitor(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics = [
            _diagnose_stiffness(cdg, context),
            _diagnose_energy(cdg, context),
            _diagnose_order(cdg, context),
            _diagnose_rejection(cdg, context),
        ]
        return [d for d in diagnostics if d is not None]
