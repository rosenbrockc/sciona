from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_assets import (
    asset_backed_rule_sets,
    clear_local_expansion_asset_caches,
    load_local_expansion_assets_by_family,
)
from sciona.principal.expansion_rules.neural_network import NeuralNetworkExpansionRuleSet


def _edge(source_id: str, target_id: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name="out",
        input_name="in",
        source_type="Tensor",
        target_type="Tensor",
    )


def _node(node_id: str, name: str) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=name,
        concept_type=ConceptType.NEURAL_NETWORK,
        status=NodeStatus.ATOMIC,
    )


def _neural_training_cdg() -> CDGExport:
    src = AlgorithmicNode(
        node_id="src",
        name="source",
        description="training batches",
        concept_type=ConceptType.DATA_ASSEMBLY,
        status=NodeStatus.ATOMIC,
    )
    forward = _node("forward", "Forward Pass")
    loss = _node("loss", "Loss Computation")
    backward = _node("backward", "Backward Pass")
    update = _node("update", "Parameter Update")
    sink = AlgorithmicNode(
        node_id="sink",
        name="sink",
        description="trained model",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
    )
    return CDGExport(
        nodes=[src, forward, loss, backward, update, sink],
        edges=[
            _edge("src", "forward"),
            _edge("forward", "loss"),
            _edge("loss", "backward"),
            _edge("backward", "update"),
            _edge("update", "sink"),
        ],
    )


def _asset_backed_rule_set():
    clear_local_expansion_asset_caches()
    return asset_backed_rule_sets([NeuralNetworkExpansionRuleSet()])[0]


def test_neural_network_provider_asset_includes_mined_training_operations() -> None:
    clear_local_expansion_asset_caches()

    asset = load_local_expansion_assets_by_family()["neural_network"]

    assert {operation.rule_name for operation in asset.operations} >= {
        "insert_swa_checkpoint_averaging_after_update",
        "insert_mixed_precision_training_before_forward",
        "insert_adversarial_weight_perturbation_before_update",
        "insert_progressive_resizing_before_forward",
        "insert_sequence_cnn_recurrent_backbone_before_loss",
        "insert_lightweight_cnn_regression_head_before_loss",
        "insert_gem_pooling_after_forward",
        "insert_multi_sample_dropout_before_loss",
        "insert_hard_negative_mining_before_loss",
        "insert_multilabel_sigmoid_head_before_loss",
        "insert_multilabel_focal_bce_loss_before_loss",
        "insert_arcface_margin_loss_before_loss",
        "insert_training_augmentation_before_forward",
        "insert_domain_specific_finetuning_before_forward",
        "insert_adaptive_batch_norm_before_forward",
        "insert_roi_cropping_before_forward",
        "insert_transformer_sequence_aggregation_before_loss",
        "insert_optimizer_schedule_before_update",
        "insert_regularization_before_loss",
        "insert_siamese_metric_backbone_before_loss",
        "insert_large_backbone_scale_attention_before_loss",
        "insert_graph_interaction_network_before_loss",
        "insert_pointrend_boundary_refinement_after_forward",
        "insert_cross_encoder_backbone_before_loss",
        "insert_preference_ranking_head_before_loss",
        "insert_dice_bce_loss_before_loss",
        "insert_test_time_augmentation_after_forward",
        "insert_multiple_instance_learning_head_before_loss",
        "insert_specaugment_before_forward",
        "insert_non_maximum_suppression_after_forward",
        "insert_stochastic_depth_before_forward",
        "insert_coordinate_regression_head_before_loss",
    }


def test_mined_neural_network_expansion_rules_apply_to_training_loop() -> None:
    rule_set = _asset_backed_rule_set()

    swa = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_swa": True}),
    )
    assert "Stochastic Weight Averaging" in {node.name for node in swa.cdg.nodes}

    mixed_precision = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_mixed_precision": True}),
    )
    assert "Configure Mixed Precision" in {node.name for node in mixed_precision.cdg.nodes}

    awp = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_awp": True}),
    )
    assert "Adversarial Weight Perturbation" in {node.name for node in awp.cdg.nodes}

    resize = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_progressive_resizing": True}),
    )
    assert "Progressive Image Resizing" in {node.name for node in resize.cdg.nodes}


def test_second_pass_neural_network_expansion_rules_apply_to_training_loop() -> None:
    rule_set = _asset_backed_rule_set()

    sequence = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_sequence_cnn_recurrent_backbone": True}),
    )
    assert "CNN-Recurrent Sequence Backbone" in {node.name for node in sequence.cdg.nodes}

    gem = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_gem_pooling": True}),
    )
    assert "Generalized Mean Pooling" in {node.name for node in gem.cdg.nodes}

    dropout = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_multi_sample_dropout": True}),
    )
    assert "Multi-Sample Dropout" in {node.name for node in dropout.cdg.nodes}

    mining = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_hard_negative_mining": True}),
    )
    assert "Hard Negative Mining" in {node.name for node in mining.cdg.nodes}

    multilabel_head = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_multilabel_sigmoid_head": True}),
    )
    assert "Multi-Label Sigmoid Head" in {node.name for node in multilabel_head.cdg.nodes}

    multilabel_loss = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_multilabel_focal_bce_loss": True}),
    )
    assert "Multi-Label Focal/BCE Loss" in {node.name for node in multilabel_loss.cdg.nodes}

    arcface = ExpansionEngine([rule_set]).expand(
        _neural_training_cdg(),
        ExpansionContext(intermediates={"requires_arcface_margin_loss": True}),
    )
    assert "ArcFace Margin Loss" in {node.name for node in arcface.cdg.nodes}


def test_support_three_neural_network_expansion_rules_apply_to_training_loop() -> None:
    rule_set = _asset_backed_rule_set()
    cases = [
        ("requires_training_augmentation", "Training Data Augmentation"),
        ("requires_domain_specific_finetuning", "Domain-Specific Fine-Tuning"),
        ("requires_adaptive_batch_norm", "Adaptive Batch Normalization"),
        ("requires_roi_cropping", "ROI Cropping Detector"),
        ("requires_lightweight_cnn_regression_head", "Lightweight CNN Regression Head"),
        ("requires_transformer_sequence_aggregation", "Transformer Sequence Aggregation"),
        ("requires_optimizer_schedule", "Optimizer Schedule"),
        ("requires_regularization", "Dropout and L2 Regularization"),
        ("requires_test_time_augmentation", "Test-Time Augmentation"),
        ("requires_siamese_metric_backbone", "Siamese Metric Backbone"),
        ("requires_large_backbone_scale_attention", "Large-Backbone Scale Attention"),
        ("requires_graph_interaction_network", "Graph Interaction Network"),
        ("requires_pointrend_boundary_refinement", "PointRend Boundary Refinement"),
        ("requires_cross_encoder_backbone", "Cross-Encoder Backbone"),
        ("requires_preference_ranking_head", "Preference Ranking Head"),
        ("requires_dice_bce_loss", "Dice/BCE Loss"),
        ("requires_multiple_instance_learning_head", "Multiple Instance Learning Head"),
        ("requires_specaugment", "SpecAugment"),
        ("requires_nms", "Non-Maximum Suppression"),
        ("requires_stochastic_depth", "Stochastic Depth"),
        ("requires_coordinate_regression_head", "Coordinate Regression Head"),
    ]

    for intermediate_key, expected_node_name in cases:
        result = ExpansionEngine([rule_set]).expand(
            _neural_training_cdg(),
            ExpansionContext(intermediates={intermediate_key: True}),
        )
        assert expected_node_name in {node.name for node in result.cdg.nodes}
