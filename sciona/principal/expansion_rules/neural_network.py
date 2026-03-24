"""Expansion rules for the Neural Network family.

Neural Network skeleton topology (4 nodes, linear with feedback):

    Forward Pass -> Loss Computation -> Backward Pass -> Parameter Update

Expansion insertion points:
  - After Backward Pass: gradient explosion detection
  - After Forward Pass: activation statistics analysis
  - After Loss Computation: loss convergence monitoring
  - After Parameter Update: weight distribution check
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

_DOMAIN = "neural_network"

_FORWARD_PASS = "Forward Pass"
_LOSS_COMPUTATION = "Loss Computation"
_BACKWARD_PASS = "Backward Pass"
_PARAMETER_UPDATE = "Parameter Update"


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


def _build_insert_gradient_explosion_detection() -> RewriteRule:
    backward = _node("backward", _BACKWARD_PASS, ConceptType.NEURAL_NETWORK)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[backward, sink], edges=[_edge("backward", "sink")])
    interface = CDGExport(nodes=[backward, sink], edges=[])

    explosion = _node(
        "explosion", "Detect Gradient Explosion", ConceptType.NEURAL_NETWORK,
        matched_primitive="detect_gradient_explosion",
        inputs=[IOSpec(name="gradients", type_desc="ndarray")],
        outputs=[IOSpec(name="max_norm", type_desc="float"),
                 IOSpec(name="is_exploding", type_desc="bool")],
        description="Check if gradient norms have exploded.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[backward, explosion, sink],
        edges=[_edge("backward", "explosion"), _edge("explosion", "sink")],
    )

    return RewriteRule(
        name="insert_gradient_explosion_detection_after_backward",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"backward": "backward", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"backward": "backward", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_activation_statistics() -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[forward, sink], edges=[_edge("forward", "sink")])
    interface = CDGExport(nodes=[forward, sink], edges=[])

    activation = _node(
        "activation", "Analyze Activation Statistics", ConceptType.NEURAL_NETWORK,
        matched_primitive="analyze_activation_statistics",
        inputs=[IOSpec(name="activations", type_desc="ndarray")],
        outputs=[IOSpec(name="dead_fraction", type_desc="float"),
                 IOSpec(name="has_dead_neurons", type_desc="bool")],
        description="Analyze fraction of dead neurons in activations.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[forward, activation, sink],
        edges=[_edge("forward", "activation"), _edge("activation", "sink")],
    )

    return RewriteRule(
        name="insert_activation_statistics_after_forward",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_loss_convergence_monitoring() -> RewriteRule:
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[loss, sink], edges=[_edge("loss", "sink")])
    interface = CDGExport(nodes=[loss, sink], edges=[])

    convergence = _node(
        "convergence", "Monitor Loss Convergence", ConceptType.NEURAL_NETWORK,
        matched_primitive="monitor_loss_convergence",
        inputs=[IOSpec(name="loss_history", type_desc="ndarray")],
        outputs=[IOSpec(name="plateau_ratio", type_desc="float"),
                 IOSpec(name="is_plateaued", type_desc="bool")],
        description="Detect loss plateau via relative change.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[loss, convergence, sink],
        edges=[_edge("loss", "convergence"), _edge("convergence", "sink")],
    )

    return RewriteRule(
        name="insert_loss_convergence_monitoring_after_loss",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"loss": "loss", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"loss": "loss", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_weight_distribution_check() -> RewriteRule:
    update = _node("update", _PARAMETER_UPDATE, ConceptType.NEURAL_NETWORK)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    weight = _node(
        "weight", "Check Weight Distribution", ConceptType.NEURAL_NETWORK,
        matched_primitive="check_weight_distribution",
        inputs=[IOSpec(name="weights", type_desc="ndarray")],
        outputs=[IOSpec(name="norm_ratio", type_desc="float"),
                 IOSpec(name="is_balanced", type_desc="bool")],
        description="Check max/min layer norm ratio.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[update, weight, sink],
        edges=[_edge("update", "weight"), _edge("weight", "sink")],
    )

    return RewriteRule(
        name="insert_weight_distribution_check_after_update",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_gradient_explosion(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    norm = intermediates.get("gradient_max_norm")
    if norm is None:
        return None
    try:
        n = float(norm)
    except (ValueError, TypeError):
        return None
    if n > 100.0:
        return ExpansionDiagnostic(
            rule_name="insert_gradient_explosion_detection_after_backward",
            severity=min(1.0, np.log10(max(n, 1)) / 4.0),
            evidence=f"Gradient max norm {n:.2e} exceeds 100.0 — exploding gradient",
            metric_name="gradient_max_norm", metric_value=n, threshold=100.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_activation_statistics(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("dead_neuron_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_activation_statistics_after_forward",
            severity=min(1.0, f),
            evidence=f"Dead neuron fraction {f:.2%} exceeds 50% — dying ReLU problem",
            metric_name="dead_neuron_fraction", metric_value=f, threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_loss_convergence(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    ratio = intermediates.get("loss_plateau_ratio")
    if ratio is None:
        return None
    try:
        r = float(ratio)
    except (ValueError, TypeError):
        return None
    if r < 1e-6:
        return ExpansionDiagnostic(
            rule_name="insert_loss_convergence_monitoring_after_loss",
            severity=min(1.0, 1.0 - r / 1e-6),
            evidence=f"Loss plateau ratio {r:.2e} below 1e-6 — training has plateaued",
            metric_name="loss_plateau_ratio", metric_value=r, threshold=1e-6,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_weight_distribution(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    ratio = intermediates.get("weight_norm_ratio")
    if ratio is None:
        return None
    try:
        r = float(ratio)
    except (ValueError, TypeError):
        return None
    if r > 100.0:
        return ExpansionDiagnostic(
            rule_name="insert_weight_distribution_check_after_update",
            severity=min(1.0, np.log10(max(r, 1)) / 4.0),
            evidence=f"Weight norm ratio {r:.2e} exceeds 100.0 — imbalanced weights",
            metric_name="weight_norm_ratio", metric_value=r, threshold=100.0,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class NeuralNetworkExpansionRuleSet:
    """Expansion rules for neural network training pipelines."""

    name = "neural_network"
    domain = "neural_network"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_gradient_explosion_detection(),
            _build_insert_activation_statistics(),
            _build_insert_loss_convergence_monitoring(),
            _build_insert_weight_distribution_check(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_gradient_explosion, _diagnose_activation_statistics,
                    _diagnose_loss_convergence, _diagnose_weight_distribution]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
