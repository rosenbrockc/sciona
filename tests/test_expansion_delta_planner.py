from __future__ import annotations

from sciona.principal.expansion_assets import clear_local_expansion_asset_caches
from sciona.principal.expansion_delta_planner import (
    DeltaAdaptationKind,
    DeltaPlanningQuery,
    plan_expansion_delta,
)


def test_delta_planner_selects_direct_use_when_base_already_covers_solution() -> None:
    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("signal_event_rate",),
            matched_techniques=("filter signal", "detect events", "measure rate"),
            missing_techniques=(),
            base_coverage=1.0,
        )
    )

    assert plan.decision == DeltaAdaptationKind.DIRECT_USE
    assert plan.selected.operation_sequence is None
    assert plan.selected.path == ("base_cdg", "direct_use")
    assert not plan.should_compose_novel


def test_delta_planner_selects_expansion_pack_for_multi_operation_gap() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("ml_model_selection",),
            matched_techniques=("fit estimator", "score validation split"),
            missing_techniques=(
                "k-fold cross validated ensemble",
                "stacking meta learner",
            ),
            stage_names=("model_training", "prediction_ensemble"),
            base_coverage=0.50,
            max_operations_per_sequence=2,
        )
    )

    assert plan.decision == DeltaAdaptationKind.EXPANSION_PACK
    assert plan.selected.projected_coverage == 1.0
    assert plan.selected.missing_terms_after_plan == ()
    assert plan.selected.operation_rule_names[:2] == (
        "apply_kfold_ensemble",
        "apply_stacking_ensemble",
    )
    assert plan.selected.path == (
        "base_cdg",
        "apply_kfold_ensemble",
        "apply_stacking_ensemble",
        "adapted_cdg",
    )


def test_delta_planner_keeps_family_sequence_candidates_after_neural_inventory_growth() -> None:
    clear_local_expansion_asset_caches()

    # Mirrors the ISIC 2019 validation query: many neural stage/context tokens should
    # not truncate the lower-scored ML operation needed to complete the sequence.
    stage_names = (
        "microscope_augmentation",
        "Microscope Augmentation",
        "Apply domain-specific augmentations: circular black masks simulating dermoscope borders, fake hair overlay, and CutMix. Uses Albumentations and custom transforms.",
        "Add realistic distortions to training images like circular borders, fake hairs, and image mixing to help the model handle real-world dermoscopic variation.",
        "data_assembly",
        "image_backbone_training",
        "Image Backbone Training",
        "Train EfficientNet B3-B7 and SE-ResNeXt backbones on augmented dermoscopic images using Focal Loss to handle extreme class imbalance (<2% positive rate).",
        "Train several powerful image-recognition networks on the skin images, using a special loss function that pays extra attention to the rare cancer cases.",
        "neural_network",
        "metadata_concatenation",
        "Metadata Concatenation",
        "Extract the penultimate embedding layer from each CNN backbone and concatenate with encoded patient metadata (age, sex, anatomical location).",
        "Combine the image features extracted by the neural network with patient information like age, sex, and where the lesion is on the body.",
        "data_assembly",
        "meta_head",
        "Meta Head",
        "Pass the concatenated image embedding + metadata vector through a small MLP for final binary classification (melanoma vs. benign).",
        "Feed the combined image and patient features through a small neural network to produce the final cancer probability.",
        "neural_network",
        "pseudo_labeling",
        "Pseudo-Labeling",
        "Use trained models to predict on old 2018/2019 ISIC competition datasets, then add high-confidence pseudo-labeled samples to the training pool to increase effective dataset size.",
        "Apply trained models to older competition datasets to generate predicted labels, then add these extra labeled examples to the training data.",
        "analysis",
        "rank_average_ensemble",
        "Rank Average Ensemble",
        "Rank-average predictions across 15+ CNN backbone variants and seeds. Each model's predictions are converted to ranks, averaged, then converted back to probabilities.",
        "Combine predictions from many different models by converting each to rankings, averaging the rankings, to get a final robust prediction.",
        "analysis",
    )

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("medical_image_tabular", "medical_image_tabular", "neural_network"),
            matched_techniques=(
                "EfficientNet-B0 to B5 and ResNeXt-101",
                "Metadata fusion via MLP branch",
            ),
            missing_techniques=(
                "Label Smoothing for human-error noise",
                "Multi-stage transfer learning",
                "Oversampling rare classes (Melanoma)",
                "Test-Time Augmentation (TTA)",
            ),
            stage_names=stage_names,
            input_names=(
                "dermoscopic_images",
                "patient_metadata",
                "dermoscopic_images",
                "augmented_images",
                "image_embeddings",
                "patient_metadata",
                "fused_features",
                "external_images",
                "model_predictions",
                "model_predictions",
            ),
            output_names=(
                "melanoma_probability",
                "augmented_images",
                "image_embeddings",
                "fused_features",
                "model_predictions",
                "pseudo_labeled_dataset",
                "melanoma_probability",
            ),
            runtime_keys=(
                "dermoscopic_images",
                "patient_metadata",
                "dermoscopic_images",
                "augmented_images",
                "image_embeddings",
                "patient_metadata",
                "fused_features",
                "external_images",
                "model_predictions",
                "model_predictions",
            ),
            intermediate_keys=stage_names,
            base_coverage=0.333333,
            min_adapted_coverage=0.50,
            max_sequences=5,
            max_operations_per_sequence=2,
        )
    )

    assert plan.decision == DeltaAdaptationKind.EXPANSION_PACK
    assert plan.selected.operation_rule_names == (
        "apply_dl_backbone_substitution",
        "insert_balanced_sampling_before_training",
    )
    assert "Oversampling rare classes (Melanoma)" in plan.selected.covered_terms
    assert plan.selected.projected_coverage == 0.833333


def test_delta_planner_selects_refinement_for_single_low_intrusion_gap() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("ode_solver",),
            matched_techniques=("evaluate derivative", "advance state", "adapt step size"),
            missing_techniques=("detect stiffness before advancing state",),
            stage_names=("evaluate_derivative", "advance_state"),
            intermediate_keys=("stiffness_ratio",),
            base_coverage=0.75,
            max_operations_per_sequence=1,
        )
    )

    assert plan.decision == DeltaAdaptationKind.REFINEMENT
    assert plan.selected.operation_rule_names == ("insert_stiffness_detection_before_advance",)
    assert plan.selected.intrusion_cost <= 0.20
    assert plan.selected.missing_terms_after_plan == ()


def test_delta_planner_selects_single_expansion_for_insert_gap() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("signal_event_rate",),
            matched_techniques=("filter signal", "detect events", "measure rate"),
            missing_techniques=("remove jumps before filtering",),
            runtime_keys=("signal", "sampling_rate"),
            intermediate_keys=("events",),
            base_coverage=0.75,
            max_operations_per_sequence=1,
        )
    )

    assert plan.decision == DeltaAdaptationKind.EXPANSION
    assert plan.selected.operation_rule_names == ("insert_jump_removal_before_filter",)
    assert plan.selected.projected_coverage == 1.0


def test_delta_planner_selects_true_novel_when_no_operation_covers_gap() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("ml_model_selection",),
            matched_techniques=("fit estimator",),
            missing_techniques=("quantum lattice annealing with orbital mechanics",),
            base_coverage=0.50,
        )
    )

    assert plan.decision == DeltaAdaptationKind.TRUE_NOVEL
    assert plan.should_compose_novel
    assert plan.selected.operation_sequence is None
    assert plan.selected.missing_terms_after_plan == (
        "quantum lattice annealing with orbital mechanics",
    )
