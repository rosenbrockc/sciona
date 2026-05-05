"""Expansion rules for the Neural Network family.

Neural Network skeleton topology (4 nodes, linear with feedback):

    Forward Pass -> Loss Computation -> Backward Pass -> Parameter Update

Expansion insertion points:
  - After Backward Pass: gradient explosion detection
  - After Forward Pass: activation statistics analysis
  - After Loss Computation: loss convergence monitoring
  - After Parameter Update: weight distribution check
  - Around training loop: SWA, mixed precision, AWP, progressive resizing
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


def _build_insert_swa_checkpoint_averaging() -> RewriteRule:
    update = _node("update", _PARAMETER_UPDATE, ConceptType.NEURAL_NETWORK)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    swa = _node(
        "swa", "Stochastic Weight Averaging", ConceptType.NEURAL_NETWORK,
        matched_primitive="stochastic_weight_averaging",
        inputs=[IOSpec(name="checkpoints", type_desc="list[nn.Module]")],
        outputs=[IOSpec(name="averaged_model", type_desc="nn.Module")],
        description="Average late-training checkpoints to improve generalization.",
    )
    rhs = CDGExport(
        nodes=[update, swa, sink],
        edges=[_edge("update", "swa"), _edge("swa", "sink")],
    )
    return RewriteRule(
        name="insert_swa_checkpoint_averaging_after_update",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_mixed_precision_training() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[src, forward], edges=[_edge("src", "forward")])
    interface = CDGExport(nodes=[src, forward], edges=[])

    precision = _node(
        "mixed_precision", "Configure Mixed Precision", ConceptType.NEURAL_NETWORK,
        matched_primitive="torch.cuda.amp.autocast",
        inputs=[IOSpec(name="batch", type_desc="Tensor")],
        outputs=[IOSpec(name="scaled_batch", type_desc="Tensor")],
        description="Run forward and backward passes under fp16 or bf16 mixed precision.",
    )
    rhs = CDGExport(
        nodes=[src, precision, forward],
        edges=[_edge("src", "mixed_precision"), _edge("mixed_precision", "forward")],
    )
    return RewriteRule(
        name="insert_mixed_precision_training_before_forward",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "forward": "forward"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "forward": "forward"}, edge_map={}),
        priority=3,
    )


def _build_insert_adversarial_weight_perturbation() -> RewriteRule:
    backward = _node("backward", _BACKWARD_PASS, ConceptType.NEURAL_NETWORK)
    update = _node("update", _PARAMETER_UPDATE, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[backward, update], edges=[_edge("backward", "update")])
    interface = CDGExport(nodes=[backward, update], edges=[])

    awp = _node(
        "awp", "Adversarial Weight Perturbation", ConceptType.NEURAL_NETWORK,
        matched_primitive="adversarial_weight_perturbation",
        inputs=[IOSpec(name="gradients", type_desc="ndarray")],
        outputs=[IOSpec(name="perturbed_weights", type_desc="nn.Module")],
        description="Perturb model weights adversarially during training for robustness.",
    )
    rhs = CDGExport(
        nodes=[backward, awp, update],
        edges=[_edge("backward", "awp"), _edge("awp", "update")],
    )
    return RewriteRule(
        name="insert_adversarial_weight_perturbation_before_update",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"backward": "backward", "update": "update"}, edge_map={}),
        r_morphism=Morphism(node_map={"backward": "backward", "update": "update"}, edge_map={}),
        priority=3,
    )


def _build_insert_progressive_resizing_before_forward() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[src, forward], edges=[_edge("src", "forward")])
    interface = CDGExport(nodes=[src, forward], edges=[])

    resize = _node(
        "progressive_resize", "Progressive Image Resizing", ConceptType.NEURAL_NETWORK,
        matched_primitive="progressive_image_resizing",
        inputs=[IOSpec(name="images", type_desc="Tensor")],
        outputs=[IOSpec(name="resized_images", type_desc="Tensor")],
        description="Train over a schedule of increasing image resolutions.",
    )
    rhs = CDGExport(
        nodes=[src, resize, forward],
        edges=[_edge("src", "progressive_resize"), _edge("progressive_resize", "forward")],
    )
    return RewriteRule(
        name="insert_progressive_resizing_before_forward",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "forward": "forward"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "forward": "forward"}, edge_map={}),
        priority=2,
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


def _planning_text(context: ExpansionContext) -> str:
    artifact = context.planning_artifact or {}
    return str(artifact).lower() if isinstance(artifact, dict) else ""


def _truthy_intermediate(context: ExpansionContext, *keys: str) -> bool:
    values = context.intermediates or {}
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and value:
            return True
        if str(value or "").strip().lower() in {"1", "true", "yes", "required", "recommended"}:
            return True
    return False


def _diagnose_swa_checkpoint_averaging(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_swa", "use_swa", "stochastic_weight_averaging")
    planning = _planning_text(context)
    if not explicit and "stochastic weight averaging" not in planning and "swa" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_swa_checkpoint_averaging_after_update",
        severity=0.75,
        evidence="Stochastic Weight Averaging is required to average late-training checkpoints.",
        metric_name="requires_swa", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_mixed_precision_training(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_mixed_precision", "use_mixed_precision")
    planning = _planning_text(context)
    if not explicit and "mixed precision" not in planning and "bf16" not in planning and "fp16" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_mixed_precision_training_before_forward",
        severity=0.70,
        evidence="Mixed-precision fp16/bf16 training is required for this neural training loop.",
        metric_name="requires_mixed_precision", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_adversarial_weight_perturbation(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_awp", "use_adversarial_weight_perturbation")
    planning = _planning_text(context)
    if not explicit and "adversarial weight perturbation" not in planning and "awp" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_adversarial_weight_perturbation_before_update",
        severity=0.75,
        evidence="Adversarial Weight Perturbation is required for robustness during parameter updates.",
        metric_name="requires_awp", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_progressive_resizing(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_progressive_resizing", "use_progressive_resizing")
    planning = _planning_text(context)
    if not explicit and "progressive image resizing" not in planning and "progressive resizing" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_progressive_resizing_before_forward",
        severity=0.65,
        evidence="Progressive image resizing is required before forward passes.",
        metric_name="requires_progressive_resizing", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


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
            _build_insert_swa_checkpoint_averaging(),
            _build_insert_mixed_precision_training(),
            _build_insert_adversarial_weight_perturbation(),
            _build_insert_progressive_resizing_before_forward(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_gradient_explosion, _diagnose_activation_statistics,
                    _diagnose_loss_convergence, _diagnose_weight_distribution,
                    _diagnose_swa_checkpoint_averaging,
                    _diagnose_mixed_precision_training,
                    _diagnose_adversarial_weight_perturbation,
                    _diagnose_progressive_resizing]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
