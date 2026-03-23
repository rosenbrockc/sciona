"""Expansion rules for the Belief Propagation family (Sum-Product, Max-Product, Loopy BP).

Belief Propagation skeleton topology (4 nodes, cyclic with memoization):

    Variable to Factor → Factor to Variable → Marginal Computation
    Memoization State (linked to both message nodes)

Expansion insertion points:
  - After Factor to Variable: message convergence monitoring
  - After Marginal Computation: belief normalization validation
  - After Variable to Factor: message damping analysis
  - Before Variable to Factor: graph cycle detection
"""

from __future__ import annotations

import logging

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

_DOMAIN = "belief_propagation"

_VAR_TO_FACTOR = "Variable to Factor"
_FACTOR_TO_VAR = "Factor to Variable"
_MARGINAL = "Marginal Computation"
_MEMO = "Memoization State"


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


def _build_insert_message_convergence_monitoring() -> RewriteRule:
    f2v = _node("f2v", _FACTOR_TO_VAR, ConceptType.MESSAGE_PASSING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[f2v, sink], edges=[_edge("f2v", "sink")])
    interface = CDGExport(nodes=[f2v, sink], edges=[])

    convergence = _node(
        "convergence", "Monitor Message Convergence", ConceptType.MESSAGE_PASSING,
        matched_primitive="monitor_message_convergence",
        inputs=[IOSpec(name="message_deltas", type_desc="ndarray"), IOSpec(name="tolerance", type_desc="float")],
        outputs=[IOSpec(name="final_delta", type_desc="float"), IOSpec(name="has_converged", type_desc="bool")],
        description="Monitor convergence of message passing iterations.",
        type_signature="ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[f2v, convergence, sink], edges=[_edge("f2v", "convergence"), _edge("convergence", "sink")])

    return RewriteRule(
        name="insert_message_convergence_monitoring_after_factor_to_var", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"f2v": "f2v", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"f2v": "f2v", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_belief_normalization_validation() -> RewriteRule:
    marginal = _node("marginal", _MARGINAL, ConceptType.MESSAGE_PASSING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[marginal, sink], edges=[_edge("marginal", "sink")])
    interface = CDGExport(nodes=[marginal, sink], edges=[])

    normalize = _node(
        "normalize", "Validate Belief Normalization", ConceptType.MESSAGE_PASSING,
        matched_primitive="validate_belief_normalization",
        inputs=[IOSpec(name="beliefs", type_desc="ndarray"), IOSpec(name="tolerance", type_desc="float")],
        outputs=[IOSpec(name="max_deviation", type_desc="float"), IOSpec(name="is_normalized", type_desc="bool")],
        description="Validate that beliefs (marginals) are properly normalized.",
        type_signature="ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[marginal, normalize, sink], edges=[_edge("marginal", "normalize"), _edge("normalize", "sink")])

    return RewriteRule(
        name="insert_belief_normalization_validation_after_marginal", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"marginal": "marginal", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"marginal": "marginal", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_message_damping_analysis() -> RewriteRule:
    v2f = _node("v2f", _VAR_TO_FACTOR, ConceptType.MESSAGE_PASSING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[v2f, sink], edges=[_edge("v2f", "sink")])
    interface = CDGExport(nodes=[v2f, sink], edges=[])

    damping = _node(
        "damping", "Analyze Message Damping", ConceptType.MESSAGE_PASSING,
        matched_primitive="analyze_message_damping",
        inputs=[IOSpec(name="message_history", type_desc="ndarray")],
        outputs=[IOSpec(name="oscillation_score", type_desc="float"), IOSpec(name="needs_damping", type_desc="bool")],
        description="Analyze whether messages oscillate, suggesting damping is needed.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[v2f, damping, sink], edges=[_edge("v2f", "damping"), _edge("damping", "sink")])

    return RewriteRule(
        name="insert_message_damping_analysis_after_var_to_factor", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"v2f": "v2f", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"v2f": "v2f", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_cycle_detection() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    v2f = _node("v2f", _VAR_TO_FACTOR, ConceptType.MESSAGE_PASSING)
    lhs = CDGExport(nodes=[src, v2f], edges=[_edge("src", "v2f")])
    interface = CDGExport(nodes=[src, v2f], edges=[])

    cycles = _node(
        "cycles", "Detect Graph Cycles", ConceptType.MESSAGE_PASSING,
        matched_primitive="detect_graph_cycles",
        inputs=[IOSpec(name="adjacency", type_desc="ndarray")],
        outputs=[IOSpec(name="n_extra_edges", type_desc="int"), IOSpec(name="is_tree", type_desc="bool")],
        description="Detect cycles in the factor graph.",
        type_signature="ndarray -> tuple[int, bool]",
    )
    rhs = CDGExport(nodes=[src, cycles, v2f], edges=[_edge("src", "cycles"), _edge("cycles", "v2f")])

    return RewriteRule(
        name="insert_cycle_detection_before_var_to_factor", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "v2f": "v2f"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "v2f": "v2f"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_convergence(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    delta = intermediates.get("message_final_delta")
    if delta is None:
        return None
    try:
        d = float(delta)
    except (ValueError, TypeError):
        return None
    if d > 1e-6:
        return ExpansionDiagnostic(
            rule_name="insert_message_convergence_monitoring_after_factor_to_var",
            severity=min(1.0, d * 1e4), evidence=f"Message delta {d:.2e} exceeds 1e-6 — not converged",
            metric_name="message_final_delta", metric_value=d, threshold=1e-6, source_domain=_DOMAIN,
        )
    return None


def _diagnose_normalization(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    dev = intermediates.get("belief_max_deviation")
    if dev is None:
        return None
    try:
        d = float(dev)
    except (ValueError, TypeError):
        return None
    if d > 1e-8:
        return ExpansionDiagnostic(
            rule_name="insert_belief_normalization_validation_after_marginal",
            severity=min(1.0, d * 1e6), evidence=f"Belief normalization deviation {d:.2e} exceeds 1e-8",
            metric_name="belief_max_deviation", metric_value=d, threshold=1e-8, source_domain=_DOMAIN,
        )
    return None


def _diagnose_damping(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    score = intermediates.get("oscillation_score")
    if score is None:
        return None
    try:
        s = float(score)
    except (ValueError, TypeError):
        return None
    if s > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_message_damping_analysis_after_var_to_factor",
            severity=min(1.0, s * 3), evidence=f"Oscillation score {s:.3f} exceeds 0.1 — messages oscillating",
            metric_name="oscillation_score", metric_value=s, threshold=0.1, source_domain=_DOMAIN,
        )
    return None


def _diagnose_cycles(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    extra = intermediates.get("n_extra_edges")
    if extra is None:
        return None
    try:
        e = int(extra)
    except (ValueError, TypeError):
        return None
    if e > 0:
        return ExpansionDiagnostic(
            rule_name="insert_cycle_detection_before_var_to_factor",
            severity=min(1.0, e / 10.0), evidence=f"{e} extra edge(s) beyond spanning tree — loopy BP, results approximate",
            metric_name="n_extra_edges", metric_value=float(e), threshold=0.0, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class BeliefPropagationExpansionRuleSet:
    """Expansion rules for belief propagation pipelines (Sum-Product, Max-Product, Loopy BP)."""

    name = "belief_propagation"
    domain = "belief_propagation"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_message_convergence_monitoring(),
            _build_insert_belief_normalization_validation(),
            _build_insert_message_damping_analysis(),
            _build_insert_cycle_detection(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_convergence, _diagnose_normalization, _diagnose_damping, _diagnose_cycles]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
