"""Expansion rules for the Continuous Optimization family (Gradient Descent, Newton, L-BFGS).

Continuous Optimization skeleton topology (4 nodes, linear pipeline):

    Initialize → Compute Gradient → Update Parameters → Check Convergence

Expansion insertion points:
  - After Compute Gradient: vanishing gradient detection
  - Before Update Parameters: loss landscape analysis
  - After Update Parameters: constraint violation check
  - After Check Convergence: convergence rate monitoring
"""

from __future__ import annotations

import logging

import numpy as np

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
)

logger = logging.getLogger(__name__)

_DOMAIN = "optimization"

_INITIALIZE = "Initialize"
_COMPUTE_GRADIENT = "Compute Gradient"
_UPDATE_PARAMETERS = "Update Parameters"
_CHECK_CONVERGENCE = "Check Convergence"


def _node(
    node_id: str, name: str, concept_type: ConceptType, *,
    matched_primitive: str | None = None, inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None, description: str = "",
    type_signature: str = "",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id, name=name, description=description or name,
        concept_type=concept_type, status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive, inputs=inputs or [],
        outputs=outputs or [], type_signature=type_signature or f"{name} -> result",
    )


def _edge(
    source_id: str, target_id: str, output_name: str = "out",
    input_name: str = "in", type_desc: str = "ndarray",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id, target_id=target_id, output_name=output_name,
        input_name=input_name, source_type=type_desc, target_type=type_desc,
    )


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_vanishing_gradient_detection() -> RewriteRule:
    gradient = _node("gradient", _COMPUTE_GRADIENT, ConceptType.OPTIMIZATION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[gradient, sink], edges=[_edge("gradient", "sink")])
    interface = CDGExport(nodes=[gradient, sink], edges=[])

    vanishing = _node(
        "vanishing", "Detect Vanishing Gradient", ConceptType.OPTIMIZATION,
        matched_primitive="detect_vanishing_gradient",
        inputs=[IOSpec(name="gradients", type_desc="ndarray")],
        outputs=[IOSpec(name="min_norm", type_desc="float"),
                 IOSpec(name="is_vanishing", type_desc="bool")],
        description="Check if gradient norms have collapsed.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[gradient, vanishing, sink],
        edges=[_edge("gradient", "vanishing"), _edge("vanishing", "sink")],
    )

    return RewriteRule(
        name="insert_vanishing_gradient_detection_after_compute_gradient",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"gradient": "gradient", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"gradient": "gradient", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_loss_landscape_analysis() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    update = _node("update", _UPDATE_PARAMETERS, ConceptType.OPTIMIZATION)
    lhs = CDGExport(nodes=[src, update], edges=[_edge("src", "update")])
    interface = CDGExport(nodes=[src, update], edges=[])

    landscape = _node(
        "landscape", "Analyze Loss Landscape", ConceptType.OPTIMIZATION,
        matched_primitive="analyze_loss_landscape",
        inputs=[IOSpec(name="hessian_eigenvalues", type_desc="ndarray")],
        outputs=[IOSpec(name="condition_number", type_desc="float"),
                 IOSpec(name="is_ill_conditioned", type_desc="bool")],
        description="Estimate local curvature via Hessian spectral analysis.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, landscape, update],
        edges=[_edge("src", "landscape"), _edge("landscape", "update")],
    )

    return RewriteRule(
        name="insert_loss_landscape_analysis_before_update",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "update": "update"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "update": "update"}, edge_map={}),
        priority=2,
    )


def _build_insert_constraint_violation_check() -> RewriteRule:
    update = _node("update", _UPDATE_PARAMETERS, ConceptType.OPTIMIZATION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    constraint = _node(
        "constraint", "Check Constraint Violation", ConceptType.OPTIMIZATION,
        matched_primitive="check_constraint_violation",
        inputs=[IOSpec(name="values", type_desc="ndarray"),
                IOSpec(name="bounds", type_desc="ndarray")],
        outputs=[IOSpec(name="max_violation", type_desc="float"),
                 IOSpec(name="is_feasible", type_desc="bool")],
        description="Check feasibility gap for constrained problems.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[update, constraint, sink],
        edges=[_edge("update", "constraint"), _edge("constraint", "sink")],
    )

    return RewriteRule(
        name="insert_constraint_violation_check_after_update",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_convergence_rate_monitoring() -> RewriteRule:
    converge = _node("converge", _CHECK_CONVERGENCE, ConceptType.OPTIMIZATION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[converge, sink], edges=[_edge("converge", "sink")])
    interface = CDGExport(nodes=[converge, sink], edges=[])

    rate = _node(
        "rate", "Monitor Convergence Rate", ConceptType.OPTIMIZATION,
        matched_primitive="monitor_convergence_rate",
        inputs=[IOSpec(name="objective_history", type_desc="ndarray")],
        outputs=[IOSpec(name="convergence_order", type_desc="float"),
                 IOSpec(name="is_converging", type_desc="bool")],
        description="Estimate empirical convergence order.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[converge, rate, sink],
        edges=[_edge("converge", "rate"), _edge("rate", "sink")],
    )

    return RewriteRule(
        name="insert_convergence_rate_monitoring_after_check",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"converge": "converge", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"converge": "converge", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_vanishing_gradient(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    min_norm = intermediates.get("gradient_min_norm")
    if min_norm is None:
        return None
    try:
        mn = float(min_norm)
    except (ValueError, TypeError):
        return None
    if mn < 1e-15:
        return ExpansionDiagnostic(
            rule_name="insert_vanishing_gradient_detection_after_compute_gradient",
            severity=1.0,
            evidence=f"Gradient min norm {mn:.2e} is below 1e-15 — vanishing gradient",
            metric_name="gradient_min_norm", metric_value=mn, threshold=1e-15,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_loss_landscape(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    cond = intermediates.get("hessian_condition_number")
    if cond is None:
        return None
    try:
        c = float(cond)
    except (ValueError, TypeError):
        return None
    if c > 1e10:
        return ExpansionDiagnostic(
            rule_name="insert_loss_landscape_analysis_before_update",
            severity=min(1.0, np.log10(max(c, 1)) / 12.0),
            evidence=f"Hessian condition number {c:.2e} exceeds 1e10 — ill-conditioned landscape",
            metric_name="hessian_condition_number", metric_value=c, threshold=1e10,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_constraint_violation(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    violation = intermediates.get("max_constraint_violation")
    if violation is None:
        return None
    try:
        v = float(violation)
    except (ValueError, TypeError):
        return None
    if v > 0:
        return ExpansionDiagnostic(
            rule_name="insert_constraint_violation_check_after_update",
            severity=min(1.0, v / 1.0),
            evidence=f"Maximum constraint violation {v:.4f} > 0 — infeasible solution",
            metric_name="max_constraint_violation", metric_value=v, threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_convergence_rate(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    order = intermediates.get("convergence_order")
    if order is None:
        return None
    try:
        o = float(order)
    except (ValueError, TypeError):
        return None
    if o < 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_convergence_rate_monitoring_after_check",
            severity=min(1.0, (0.5 - o) / 0.5),
            evidence=f"Convergence order {o:.3f} is below 0.5 — very slow convergence",
            metric_name="convergence_order", metric_value=o, threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class OptimizationExpansionRuleSet:
    """Expansion rules for continuous optimization pipelines (GD, Newton, L-BFGS)."""

    name = "optimization"
    domain = "optimization"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_vanishing_gradient_detection(),
            _build_insert_loss_landscape_analysis(),
            _build_insert_constraint_violation_check(),
            _build_insert_convergence_rate_monitoring(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_vanishing_gradient, _diagnose_loss_landscape,
                    _diagnose_constraint_violation, _diagnose_convergence_rate]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
