"""Expansion rules for the Neural Network family.

Neural Network skeleton topology (4 nodes, linear with feedback):

    Forward Pass -> Loss Computation -> Backward Pass -> Parameter Update

Expansion insertion points:
  - After Backward Pass: gradient explosion detection
  - After Forward Pass: activation statistics analysis
  - After Loss Computation: loss convergence monitoring
  - After Parameter Update: weight distribution check
  - Around training loop: SWA, mixed precision, AWP, progressive resizing
  - Competition refinements: recurrent sequence backbones, GeM pooling,
    multi-sample dropout, hard-negative mining, ArcFace-style margin losses
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


def _build_insert_before_forward_rule(
    *,
    rule_name: str,
    node_id: str,
    node_name: str,
    matched_primitive: str,
    description: str,
    priority: int = 3,
) -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[src, forward], edges=[_edge("src", "forward")])
    interface = CDGExport(nodes=[src, forward], edges=[])

    inserted = _node(
        node_id,
        node_name,
        ConceptType.NEURAL_NETWORK,
        matched_primitive=matched_primitive,
        inputs=[IOSpec(name="batch", type_desc="Tensor")],
        outputs=[IOSpec(name="prepared_batch", type_desc="Tensor")],
        description=description,
    )
    rhs = CDGExport(
        nodes=[src, inserted, forward],
        edges=[_edge("src", node_id), _edge(node_id, "forward")],
    )
    return RewriteRule(
        name=rule_name,
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "forward": "forward"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "forward": "forward"}, edge_map={}),
        priority=priority,
    )


def _build_insert_forward_loss_rule(
    *,
    rule_name: str,
    node_id: str,
    node_name: str,
    matched_primitive: str,
    description: str,
    input_name: str = "features",
    output_name: str = "features",
    priority: int = 3,
) -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[forward, loss], edges=[_edge("forward", "loss")])
    interface = CDGExport(nodes=[forward, loss], edges=[])

    inserted = _node(
        node_id,
        node_name,
        ConceptType.NEURAL_NETWORK,
        matched_primitive=matched_primitive,
        inputs=[IOSpec(name=input_name, type_desc="Tensor")],
        outputs=[IOSpec(name=output_name, type_desc="Tensor")],
        description=description,
    )
    rhs = CDGExport(
        nodes=[forward, inserted, loss],
        edges=[_edge("forward", node_id), _edge(node_id, "loss")],
    )
    return RewriteRule(
        name=rule_name,
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        priority=priority,
    )


def _build_insert_training_augmentation_before_forward() -> RewriteRule:
    return _build_insert_before_forward_rule(
        rule_name="insert_training_augmentation_before_forward",
        node_id="training_augmentation",
        node_name="Training Data Augmentation",
        matched_primitive="mixup_cutmix_spatial_frequency_augmentation",
        description="Apply MixUp, CutMix, XYMasking, frequency masking, spatial transforms, color jitter, blur, or noise augmentation.",
    )


def _build_insert_domain_specific_finetuning_before_forward() -> RewriteRule:
    return _build_insert_before_forward_rule(
        rule_name="insert_domain_specific_finetuning_before_forward",
        node_id="domain_specific_finetuning",
        node_name="Domain-Specific Fine-Tuning",
        matched_primitive="domain_specific_sft_adapter_pretraining",
        description="Load domain-specific pretraining, supervised fine-tuning, or lightweight adapters before task forward passes.",
    )


def _build_insert_adaptive_batch_norm_before_forward() -> RewriteRule:
    return _build_insert_before_forward_rule(
        rule_name="insert_adaptive_batch_norm_before_forward",
        node_id="adaptive_batch_norm",
        node_name="Adaptive Batch Normalization",
        matched_primitive="adaptive_batch_normalization",
        description="Adapt batch-normalization statistics per domain, location, plate, or deployment split.",
        priority=2,
    )


def _build_insert_roi_cropping_before_forward() -> RewriteRule:
    return _build_insert_before_forward_rule(
        rule_name="insert_roi_cropping_before_forward",
        node_id="roi_cropping_detector",
        node_name="ROI Cropping Detector",
        matched_primitive="megadetector_roi_cropping",
        description="Crop task-relevant regions with a detector before the main model forward pass.",
        priority=3,
    )


def _build_insert_sequence_cnn_recurrent_backbone() -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[forward, loss], edges=[_edge("forward", "loss")])
    interface = CDGExport(nodes=[forward, loss], edges=[])

    sequence = _node(
        "sequence_cnn_recurrent_backbone",
        "CNN-Recurrent Sequence Backbone",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="cnn_bidirectional_lstm_gru_backbone",
        inputs=[IOSpec(name="sequence_features", type_desc="Tensor")],
        outputs=[IOSpec(name="sequence_embedding", type_desc="Tensor")],
        description="Encode temporal or ordered features with a 1D/2D CNN followed by bidirectional LSTM or GRU layers.",
    )
    rhs = CDGExport(
        nodes=[forward, sequence, loss],
        edges=[_edge("forward", "sequence_cnn_recurrent_backbone"), _edge("sequence_cnn_recurrent_backbone", "loss")],
    )
    return RewriteRule(
        name="insert_sequence_cnn_recurrent_backbone_before_loss",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        priority=3,
    )


def _build_insert_transformer_sequence_aggregation() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_transformer_sequence_aggregation_before_loss",
        node_id="transformer_sequence_aggregation",
        node_name="Transformer Sequence Aggregation",
        matched_primitive="transformer_sequence_level_aggregation",
        description="Aggregate frame, slice, token, or event-level features with a transformer or sequence-level aggregator.",
    )


def _build_insert_gem_pooling() -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[forward, loss], edges=[_edge("forward", "loss")])
    interface = CDGExport(nodes=[forward, loss], edges=[])

    pooling = _node(
        "gem_pooling",
        "Generalized Mean Pooling",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="generalized_mean_pooling",
        inputs=[IOSpec(name="feature_map", type_desc="Tensor")],
        outputs=[IOSpec(name="pooled_features", type_desc="Tensor")],
        description="Aggregate convolutional features with GeM or GAP-style global pooling before the task head.",
    )
    rhs = CDGExport(
        nodes=[forward, pooling, loss],
        edges=[_edge("forward", "gem_pooling"), _edge("gem_pooling", "loss")],
    )
    return RewriteRule(
        name="insert_gem_pooling_after_forward",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        priority=3,
    )


def _build_insert_optimizer_schedule_before_update() -> RewriteRule:
    backward = _node("backward", _BACKWARD_PASS, ConceptType.NEURAL_NETWORK)
    update = _node("update", _PARAMETER_UPDATE, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[backward, update], edges=[_edge("backward", "update")])
    interface = CDGExport(nodes=[backward, update], edges=[])

    schedule = _node(
        "optimizer_schedule",
        "Optimizer Schedule",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="llrd_cyclic_lr_adam_weight_decay",
        inputs=[IOSpec(name="gradients", type_desc="Tensor")],
        outputs=[IOSpec(name="scheduled_update", type_desc="Tensor")],
        description="Configure layer-wise learning-rate decay, cyclic learning rates, Adam decay, or weight decay before parameter updates.",
    )
    rhs = CDGExport(
        nodes=[backward, schedule, update],
        edges=[_edge("backward", "optimizer_schedule"), _edge("optimizer_schedule", "update")],
    )
    return RewriteRule(
        name="insert_optimizer_schedule_before_update",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"backward": "backward", "update": "update"}, edge_map={}),
        r_morphism=Morphism(node_map={"backward": "backward", "update": "update"}, edge_map={}),
        priority=3,
    )


def _build_insert_multi_sample_dropout() -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[forward, loss], edges=[_edge("forward", "loss")])
    interface = CDGExport(nodes=[forward, loss], edges=[])

    dropout = _node(
        "multi_sample_dropout",
        "Multi-Sample Dropout",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="multi_sample_dropout_head",
        inputs=[IOSpec(name="features", type_desc="Tensor")],
        outputs=[IOSpec(name="averaged_logits", type_desc="Tensor")],
        description="Average logits from multiple dropout masks to stabilize the task head.",
    )
    rhs = CDGExport(
        nodes=[forward, dropout, loss],
        edges=[_edge("forward", "multi_sample_dropout"), _edge("multi_sample_dropout", "loss")],
    )
    return RewriteRule(
        name="insert_multi_sample_dropout_before_loss",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        priority=2,
    )


def _build_insert_regularization_before_loss() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_regularization_before_loss",
        node_id="dropout_l2_regularization",
        node_name="Dropout and L2 Regularization",
        matched_primitive="dropout_spatial_dropout_l2_regularization",
        description="Apply dropout, spatial dropout, L2, or weight-decay regularization to stabilize the task head.",
        priority=2,
    )


def _build_insert_test_time_augmentation_after_forward() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_test_time_augmentation_after_forward",
        node_id="test_time_augmentation",
        node_name="Test-Time Augmentation",
        matched_primitive="test_time_augmentation_multiscale_flip",
        description="Average inference predictions over multi-scale, flipping, crop, or other safe test-time augmentations.",
        input_name="inference_batch",
        output_name="tta_predictions",
        priority=3,
    )


def _build_insert_hard_negative_mining() -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[forward, loss], edges=[_edge("forward", "loss")])
    interface = CDGExport(nodes=[forward, loss], edges=[])

    mining = _node(
        "hard_negative_mining",
        "Hard Negative Mining",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="hard_negative_triplet_mining",
        inputs=[IOSpec(name="embeddings", type_desc="Tensor")],
        outputs=[IOSpec(name="mined_pairs", type_desc="Tensor")],
        description="Select hard negatives or hard triplets before contrastive, triplet, or ranking loss computation.",
    )
    rhs = CDGExport(
        nodes=[forward, mining, loss],
        edges=[_edge("forward", "hard_negative_mining"), _edge("hard_negative_mining", "loss")],
    )
    return RewriteRule(
        name="insert_hard_negative_mining_before_loss",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        priority=3,
    )


def _build_insert_siamese_metric_backbone() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_siamese_metric_backbone_before_loss",
        node_id="siamese_metric_backbone",
        node_name="Siamese Metric Backbone",
        matched_primitive="siamese_resnet_vgg_metric_network",
        description="Encode paired examples with shared-weight Siamese backbones for metric or identity learning.",
    )


def _build_insert_graph_interaction_network() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_graph_interaction_network_before_loss",
        node_id="graph_interaction_network",
        node_name="Graph Interaction Network",
        matched_primitive="graph_neural_network_interaction_model",
        description="Model entity proximity, kinematics, contacts, or particle interactions with a graph neural network.",
    )


def _build_insert_pointrend_boundary_refinement() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_pointrend_boundary_refinement_after_forward",
        node_id="pointrend_boundary_refinement",
        node_name="PointRend Boundary Refinement",
        matched_primitive="pointrend_boundary_refinement",
        description="Refine segmentation or instance boundaries with point-based high-resolution predictions.",
    )


def _build_insert_cross_encoder_backbone() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_cross_encoder_backbone_before_loss",
        node_id="cross_encoder_backbone",
        node_name="Cross-Encoder Backbone",
        matched_primitive="deberta_large_cross_encoder",
        description="Jointly encode query-document, prompt-answer, or pairwise relevance inputs with a DeBERTa-style cross-encoder.",
    )


def _build_insert_preference_ranking_head() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_preference_ranking_head_before_loss",
        node_id="preference_ranking_head",
        node_name="Preference Ranking Head",
        matched_primitive="bradley_terry_preference_model",
        description="Train a Bradley-Terry or pairwise preference ranking head from comparisons.",
    )


def _build_insert_multilabel_sigmoid_head() -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[forward, loss], edges=[_edge("forward", "loss")])
    interface = CDGExport(nodes=[forward, loss], edges=[])

    head = _node(
        "multilabel_sigmoid_head",
        "Multi-Label Sigmoid Head",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="multilabel_sigmoid_classification_head",
        inputs=[IOSpec(name="features", type_desc="Tensor")],
        outputs=[IOSpec(name="label_logits", type_desc="Tensor")],
        description="Predict independent label logits for multi-label or auxiliary multi-task classification.",
    )
    rhs = CDGExport(
        nodes=[forward, head, loss],
        edges=[_edge("forward", "multilabel_sigmoid_head"), _edge("multilabel_sigmoid_head", "loss")],
    )
    return RewriteRule(
        name="insert_multilabel_sigmoid_head_before_loss",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        priority=3,
    )


def _build_insert_multilabel_focal_bce_loss() -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[forward, loss], edges=[_edge("forward", "loss")])
    interface = CDGExport(nodes=[forward, loss], edges=[])

    multilabel_loss = _node(
        "multilabel_focal_bce_loss",
        "Multi-Label Focal/BCE Loss",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="multilabel_focal_bce_label_smoothing_loss",
        inputs=[IOSpec(name="label_logits", type_desc="Tensor"), IOSpec(name="labels", type_desc="Tensor")],
        outputs=[IOSpec(name="loss", type_desc="Tensor")],
        description="Compute multi-label focal or BCE loss, optionally with label smoothing.",
    )
    rhs = CDGExport(
        nodes=[forward, multilabel_loss, loss],
        edges=[_edge("forward", "multilabel_focal_bce_loss"), _edge("multilabel_focal_bce_loss", "loss")],
    )
    return RewriteRule(
        name="insert_multilabel_focal_bce_loss_before_loss",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        priority=3,
    )


def _build_insert_dice_bce_loss_before_loss() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_dice_bce_loss_before_loss",
        node_id="dice_bce_loss",
        node_name="Dice/BCE Loss",
        matched_primitive="dice_binary_cross_entropy_loss",
        description="Use Dice, Binary Cross-Entropy, or Dice+BCE loss for segmentation and multilabel masks.",
        input_name="logits",
        output_name="loss",
    )


def _build_insert_multiple_instance_learning_head_before_loss() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_multiple_instance_learning_head_before_loss",
        node_id="multiple_instance_learning_head",
        node_name="Multiple Instance Learning Head",
        matched_primitive="chowder_multiple_instance_learning_attention_pooling",
        description="Aggregate bag, tile, or patch-level embeddings with a MIL head such as Chowder or attention pooling.",
        input_name="instance_embeddings",
        output_name="bag_logits",
    )


def _build_insert_specaugment_before_forward() -> RewriteRule:
    return _build_insert_before_forward_rule(
        rule_name="insert_specaugment_before_forward",
        node_id="specaugment",
        node_name="SpecAugment",
        matched_primitive="frequency_time_masking_specaugment",
        description="Apply frequency masking, time masking, or time-frequency masking to spectrogram-like training inputs.",
        priority=2,
    )


def _build_insert_non_maximum_suppression_after_forward() -> RewriteRule:
    return _build_insert_forward_loss_rule(
        rule_name="insert_non_maximum_suppression_after_forward",
        node_id="non_maximum_suppression",
        node_name="Non-Maximum Suppression",
        matched_primitive="class_aware_non_maximum_suppression_triplet_filtering",
        description="Filter overlapping boxes, detections, or triplets with class-aware non-maximum suppression after prediction.",
        input_name="detections",
        output_name="filtered_detections",
    )


def _build_insert_stochastic_depth_before_forward() -> RewriteRule:
    return _build_insert_before_forward_rule(
        rule_name="insert_stochastic_depth_before_forward",
        node_id="stochastic_depth",
        node_name="Stochastic Depth",
        matched_primitive="stochastic_depth_drop_path_transformer_regularization",
        description="Regularize residual, Swin, or transformer blocks with stochastic depth or DropPath during training.",
        priority=2,
    )


def _build_insert_arcface_margin_loss_before_loss() -> RewriteRule:
    forward = _node("forward", _FORWARD_PASS, ConceptType.NEURAL_NETWORK)
    loss = _node("loss", _LOSS_COMPUTATION, ConceptType.NEURAL_NETWORK)
    lhs = CDGExport(nodes=[forward, loss], edges=[_edge("forward", "loss")])
    interface = CDGExport(nodes=[forward, loss], edges=[])

    arcface = _node(
        "arcface_margin_loss",
        "ArcFace Margin Loss",
        ConceptType.NEURAL_NETWORK,
        matched_primitive="arcface_subcenter_margin_loss",
        inputs=[IOSpec(name="embeddings", type_desc="Tensor"), IOSpec(name="labels", type_desc="Tensor")],
        outputs=[IOSpec(name="loss", type_desc="Tensor")],
        description="Use additive angular margin loss, including ArcFace or sub-center ArcFace variants.",
    )
    rhs = CDGExport(
        nodes=[forward, arcface, loss],
        edges=[_edge("forward", "arcface_margin_loss"), _edge("arcface_margin_loss", "loss")],
    )
    return RewriteRule(
        name="insert_arcface_margin_loss_before_loss",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "loss": "loss"}, edge_map={}),
        priority=3,
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


def _diagnose_textual_rule(
    context: ExpansionContext,
    *,
    rule_name: str,
    metric_name: str,
    evidence: str,
    intermediate_keys: tuple[str, ...],
    planning_terms: tuple[str, ...],
    severity: float = 0.70,
) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, *intermediate_keys)
    planning = _planning_text(context)
    if not explicit and not any(term in planning for term in planning_terms):
        return None
    return ExpansionDiagnostic(
        rule_name=rule_name,
        severity=severity,
        evidence=evidence,
        metric_name=metric_name, metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


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


def _diagnose_training_augmentation(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_training_augmentation_before_forward",
        metric_name="requires_training_augmentation",
        evidence="Training data augmentation such as MixUp, CutMix, masking, spatial, color, blur, or noise transforms is required.",
        intermediate_keys=("requires_training_augmentation", "use_mixup_cutmix", "use_heavy_augmentation"),
        planning_terms=("mixup", "cutmix", "xymasking", "frequency masking", "heavy rotation", "heavy spatial", "color jitter", "motion-blur", "motion blur", "noise/blur"),
    )


def _diagnose_domain_specific_finetuning(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_domain_specific_finetuning_before_forward",
        metric_name="requires_domain_specific_finetuning",
        evidence="Domain-specific pretraining, SFT, or adapters are required before task training.",
        intermediate_keys=("requires_domain_specific_finetuning", "use_domain_adapters", "use_large_scale_pretraining"),
        planning_terms=("domain-specific fine-tuning", "specific fine-tuning", "organ-specific", "adapter", "xeno-canto", "large-scale pre-training", "sft"),
    )


def _diagnose_adaptive_batch_norm(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_adaptive_batch_norm_before_forward",
        metric_name="requires_adaptive_batch_norm",
        evidence="Adaptive batch normalization is required for domain/location/plate shift.",
        intermediate_keys=("requires_adaptive_batch_norm", "use_adaptive_batch_norm"),
        planning_terms=("adaptive batch normalization", "adaptive batch norm", "per-location", "per-plate"),
        severity=0.65,
    )


def _diagnose_roi_cropping(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_roi_cropping_before_forward",
        metric_name="requires_roi_cropping",
        evidence="Detector-based region cropping is required before the main forward pass.",
        intermediate_keys=("requires_roi_cropping", "use_megadetector_cropping"),
        planning_terms=("megadetector", "roi cropping", "animal cropping", "detector-based roi"),
    )


def _diagnose_sequence_cnn_recurrent_backbone(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_sequence_cnn_recurrent_backbone", "use_bilstm_gru_backbone")
    planning = _planning_text(context)
    planning_requires = any(token in planning for token in ("bidirectional lstm", "bilstm", "gru", "1d-cnn", "2d cnn"))
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_sequence_cnn_recurrent_backbone_before_loss",
        severity=0.75,
        evidence="A CNN plus bidirectional LSTM/GRU sequence backbone is required.",
        metric_name="requires_sequence_cnn_recurrent_backbone", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_transformer_sequence_aggregation(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_transformer_sequence_aggregation_before_loss",
        metric_name="requires_transformer_sequence_aggregation",
        evidence="Sequence-level transformer aggregation is required before loss computation.",
        intermediate_keys=("requires_transformer_sequence_aggregation", "use_sequence_level_aggregation"),
        planning_terms=("sequence-level feature aggregation", "transformer-based sequence aggregation", "sequence aggregation"),
    )


def _diagnose_gem_pooling(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_gem_pooling", "use_gem_pooling", "use_gap_pooling")
    planning = _planning_text(context)
    planning_requires = any(token in planning for token in ("generalized mean", "gem pooling", "global average pooling", "gap features"))
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_gem_pooling_after_forward",
        severity=0.70,
        evidence="GeM or GAP-style global pooling is required after the backbone forward pass.",
        metric_name="requires_gem_pooling", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_optimizer_schedule(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_optimizer_schedule_before_update",
        metric_name="requires_optimizer_schedule",
        evidence="Optimizer scheduling such as LLRD, CLR, Adam decay, or weight decay is required.",
        intermediate_keys=("requires_optimizer_schedule", "use_llrd", "use_cyclic_lr", "use_adam_decay"),
        planning_terms=("layer-wise learning rate", "llrd", "cyclic learning rate", "cyclic learning rates", "adam optimizer with decay", "weight decay"),
    )


def _diagnose_multi_sample_dropout(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_multi_sample_dropout", "use_multi_sample_dropout")
    planning = _planning_text(context)
    if not explicit and "multi-sample dropout" not in planning and "multi sample dropout" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_multi_sample_dropout_before_loss",
        severity=0.65,
        evidence="Multi-sample dropout is required to stabilize task-head logits.",
        metric_name="requires_multi_sample_dropout", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_regularization(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_regularization_before_loss",
        metric_name="requires_regularization",
        evidence="Dropout, spatial dropout, L2, or heavy regularization is required.",
        intermediate_keys=("requires_regularization", "use_dropout_l2_regularization"),
        planning_terms=("dropout and l2", "spatial dropout", "heavy dropout", "l2 regularization"),
        severity=0.65,
    )


def _diagnose_test_time_augmentation(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_test_time_augmentation_after_forward",
        metric_name="requires_test_time_augmentation",
        evidence="Test-Time Augmentation is required for inference-time prediction averaging.",
        intermediate_keys=("requires_test_time_augmentation", "use_test_time_augmentation", "use_tta"),
        planning_terms=("test-time augmentation", "test time augmentation", " tta", "multi-scale test-time", "tta with flipping", "tta with scaling"),
        severity=0.75,
    )


def _diagnose_hard_negative_mining(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_hard_negative_mining", "use_hard_negative_mining")
    planning = _planning_text(context)
    planning_requires = any(token in planning for token in ("hard negative", "hard triplet", "triplet mining"))
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_hard_negative_mining_before_loss",
        severity=0.75,
        evidence="Hard-negative or hard-triplet mining is required before loss computation.",
        metric_name="requires_hard_negative_mining", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_siamese_metric_backbone(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_siamese_metric_backbone_before_loss",
        metric_name="requires_siamese_metric_backbone",
        evidence="A Siamese metric-learning backbone is required.",
        intermediate_keys=("requires_siamese_metric_backbone", "use_siamese_network"),
        planning_terms=("siamese network", "siamese resnet", "siamese vgg"),
    )


def _diagnose_graph_interaction_network(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_graph_interaction_network_before_loss",
        metric_name="requires_graph_interaction_network",
        evidence="A graph neural interaction model is required.",
        intermediate_keys=("requires_graph_interaction_network", "use_gnn_interaction_model"),
        planning_terms=("graph neural network", "gnn", "interaction graph"),
    )


def _diagnose_pointrend_boundary_refinement(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_pointrend_boundary_refinement_after_forward",
        metric_name="requires_pointrend_boundary_refinement",
        evidence="PointRend boundary refinement is required for segmentation boundaries.",
        intermediate_keys=("requires_pointrend_boundary_refinement", "use_pointrend"),
        planning_terms=("pointrend", "boundary refinement"),
    )


def _diagnose_cross_encoder_backbone(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_cross_encoder_backbone_before_loss",
        metric_name="requires_cross_encoder_backbone",
        evidence="A DeBERTa-style cross-encoder backbone is required.",
        intermediate_keys=("requires_cross_encoder_backbone", "use_deberta_cross_encoder"),
        planning_terms=("cross-encoder", "cross encoder", "deberta-v3", "deberta-v4"),
    )


def _diagnose_preference_ranking_head(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_preference_ranking_head_before_loss",
        metric_name="requires_preference_ranking_head",
        evidence="Bradley-Terry or pairwise preference modeling is required.",
        intermediate_keys=("requires_preference_ranking_head", "use_bradley_terry"),
        planning_terms=("bradley-terry", "preference modeling", "preference ranking"),
    )


def _diagnose_multilabel_sigmoid_head(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_multilabel_sigmoid_head", "use_multilabel_sigmoid_head")
    planning = _planning_text(context)
    planning_requires = any(token in planning for token in ("multi-label sigmoid", "multilabel sigmoid", "multi-task sigmoid", "auxiliary head"))
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_multilabel_sigmoid_head_before_loss",
        severity=0.70,
        evidence="A multi-label or multi-task sigmoid head is required before loss computation.",
        metric_name="requires_multilabel_sigmoid_head", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_dice_bce_loss(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_dice_bce_loss_before_loss",
        metric_name="requires_dice_bce_loss",
        evidence="Dice, BCE, or Dice+BCE loss is required for mask or multilabel training.",
        intermediate_keys=("requires_dice_bce_loss", "use_dice_bce_loss", "use_bce_loss"),
        planning_terms=("binary cross-entropy", "bce loss", "dice + binary", "dice + bce", "dice loss"),
    )


def _diagnose_multiple_instance_learning_head(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_multiple_instance_learning_head_before_loss",
        metric_name="requires_multiple_instance_learning_head",
        evidence="A multiple-instance learning head is required for bag, tile, or patch aggregation.",
        intermediate_keys=("requires_multiple_instance_learning_head", "use_mil_head", "use_chowder_mil"),
        planning_terms=("multiple instance learning", "mil head", "chowder"),
    )


def _diagnose_specaugment(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_specaugment_before_forward",
        metric_name="requires_specaugment",
        evidence="SpecAugment frequency/time masking is required for spectrogram training.",
        intermediate_keys=("requires_specaugment", "use_specaugment"),
        planning_terms=("specaugment", "frequency masking", "time masking", "frequency/time masking", "time/frequency masking"),
    )


def _diagnose_non_maximum_suppression(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_non_maximum_suppression_after_forward",
        metric_name="requires_nms",
        evidence="Non-maximum suppression is required after detection or triplet prediction.",
        intermediate_keys=("requires_nms", "use_nms", "use_class_aware_nms"),
        planning_terms=("non-maximum suppression", "non maximum suppression", "class-aware nms", "nms"),
    )


def _diagnose_stochastic_depth(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    return _diagnose_textual_rule(
        context,
        rule_name="insert_stochastic_depth_before_forward",
        metric_name="requires_stochastic_depth",
        evidence="Stochastic depth or DropPath regularization is required for residual or transformer layers.",
        intermediate_keys=("requires_stochastic_depth", "use_stochastic_depth", "use_drop_path"),
        planning_terms=("stochastic depth", "drop path", "droppath", "swin layers", "transformer layers"),
    )


def _diagnose_multilabel_focal_bce_loss(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_multilabel_focal_bce_loss", "use_multilabel_focal_loss")
    planning = _planning_text(context)
    planning_requires = any(token in planning for token in ("multi-label focal", "multilabel focal", "multi-label bce", "bce loss with label smoothing"))
    if not explicit and not planning_requires:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_multilabel_focal_bce_loss_before_loss",
        severity=0.70,
        evidence="A multi-label focal/BCE loss variant is required.",
        metric_name="requires_multilabel_focal_bce_loss", metric_value=1.0, threshold=0.0,
        source_domain=_DOMAIN,
    )


def _diagnose_arcface_margin_loss(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    explicit = _truthy_intermediate(context, "requires_arcface_margin_loss", "use_arcface", "use_subcenter_arcface")
    planning = _planning_text(context)
    if not explicit and "arcface" not in planning and "additive angular margin" not in planning:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_arcface_margin_loss_before_loss",
        severity=0.75,
        evidence="ArcFace or sub-center ArcFace angular-margin loss is required.",
        metric_name="requires_arcface_margin_loss", metric_value=1.0, threshold=0.0,
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
            _build_insert_training_augmentation_before_forward(),
            _build_insert_domain_specific_finetuning_before_forward(),
            _build_insert_adaptive_batch_norm_before_forward(),
            _build_insert_roi_cropping_before_forward(),
            _build_insert_sequence_cnn_recurrent_backbone(),
            _build_insert_transformer_sequence_aggregation(),
            _build_insert_gem_pooling(),
            _build_insert_optimizer_schedule_before_update(),
            _build_insert_multi_sample_dropout(),
            _build_insert_regularization_before_loss(),
            _build_insert_test_time_augmentation_after_forward(),
            _build_insert_hard_negative_mining(),
            _build_insert_siamese_metric_backbone(),
            _build_insert_graph_interaction_network(),
            _build_insert_pointrend_boundary_refinement(),
            _build_insert_cross_encoder_backbone(),
            _build_insert_preference_ranking_head(),
            _build_insert_multilabel_sigmoid_head(),
            _build_insert_multilabel_focal_bce_loss(),
            _build_insert_dice_bce_loss_before_loss(),
            _build_insert_multiple_instance_learning_head_before_loss(),
            _build_insert_specaugment_before_forward(),
            _build_insert_non_maximum_suppression_after_forward(),
            _build_insert_stochastic_depth_before_forward(),
            _build_insert_arcface_margin_loss_before_loss(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_gradient_explosion, _diagnose_activation_statistics,
                    _diagnose_loss_convergence, _diagnose_weight_distribution,
                    _diagnose_swa_checkpoint_averaging,
                    _diagnose_mixed_precision_training,
                    _diagnose_adversarial_weight_perturbation,
                    _diagnose_progressive_resizing,
                    _diagnose_training_augmentation,
                    _diagnose_domain_specific_finetuning,
                    _diagnose_adaptive_batch_norm,
                    _diagnose_roi_cropping,
                    _diagnose_sequence_cnn_recurrent_backbone,
                    _diagnose_transformer_sequence_aggregation,
                    _diagnose_gem_pooling,
                    _diagnose_optimizer_schedule,
                    _diagnose_multi_sample_dropout,
                    _diagnose_regularization,
                    _diagnose_test_time_augmentation,
                    _diagnose_hard_negative_mining,
                    _diagnose_siamese_metric_backbone,
                    _diagnose_graph_interaction_network,
                    _diagnose_pointrend_boundary_refinement,
                    _diagnose_cross_encoder_backbone,
                    _diagnose_preference_ranking_head,
                    _diagnose_multilabel_sigmoid_head,
                    _diagnose_multilabel_focal_bce_loss,
                    _diagnose_dice_bce_loss,
                    _diagnose_multiple_instance_learning_head,
                    _diagnose_specaugment,
                    _diagnose_non_maximum_suppression,
                    _diagnose_stochastic_depth,
                    _diagnose_arcface_margin_loss]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)
