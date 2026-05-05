from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion_assets import clear_local_expansion_asset_caches
from sciona.principal.expansion_retrieval import (
    ExpansionAssetRetriever,
    ExpansionRetrievalQuery,
    retrieve_expansion_sequences,
)


def _signal_event_rate_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="root",
        name="signal event rate",
        description="filter signal, detect events, measure rate",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.DECOMPOSED,
        children=["filter", "detect", "rate"],
        inputs=[
            IOSpec(name="signal", type_desc="ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        outputs=[IOSpec(name="rate", type_desc="ndarray")],
    )
    filt = AlgorithmicNode(
        node_id="filter",
        name="filter_signal",
        description="condition waveform",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        inputs=[IOSpec(name="signal", type_desc="ndarray")],
        outputs=[IOSpec(name="filtered", type_desc="ndarray")],
    )
    detect = AlgorithmicNode(
        node_id="detect",
        name="detect_events",
        description="detect peaks",
        concept_type=ConceptType.DATA_EXTRACTION,
        status=NodeStatus.ATOMIC,
        inputs=[IOSpec(name="filtered", type_desc="ndarray")],
        outputs=[IOSpec(name="events", type_desc="ndarray")],
    )
    rate = AlgorithmicNode(
        node_id="rate",
        name="measure_rate",
        description="measure event rate",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.ATOMIC,
        inputs=[IOSpec(name="events", type_desc="ndarray")],
        outputs=[IOSpec(name="rate", type_desc="ndarray")],
    )
    return CDGExport(
        nodes=[root, filt, detect, rate],
        edges=[
            DependencyEdge(
                source_id="filter",
                target_id="detect",
                output_name="filtered",
                input_name="filtered",
                source_type="ndarray",
                target_type="ndarray",
            ),
            DependencyEdge(
                source_id="detect",
                target_id="rate",
                output_name="events",
                input_name="events",
                source_type="ndarray",
                target_type="ndarray",
            ),
        ],
        metadata={"family": "signal_event_rate"},
    )


def test_retrieves_ranked_signal_expansion_sequence_from_cdg_and_techniques() -> None:
    clear_local_expansion_asset_caches()

    sequences = retrieve_expansion_sequences(
        ExpansionRetrievalQuery(
            missing_techniques=(
                "remove jumps before filtering",
                "reject outlier intervals after detection",
            ),
            runtime_keys=("signal", "sampling_rate"),
            intermediate_keys=("events",),
        ),
        cdg=_signal_event_rate_cdg(),
        max_sequences=3,
        max_operations_per_sequence=4,
    )

    assert sequences
    best = sequences[0]
    rule_names = {operation.rule_name for operation in best.operations}
    assert best.asset_family == "signal_event_rate"
    assert "insert_jump_removal_before_filter" in rule_names
    assert "insert_outlier_rejection_after_detection" in rule_names
    assert set(best.covered_terms) == {
        "reject outlier intervals after detection",
        "remove jumps before filtering",
    }


def test_retrieves_ml_sequence_with_prerequisite_order() -> None:
    clear_local_expansion_asset_caches()

    sequences = ExpansionAssetRetriever().retrieve_sequences(
        ExpansionRetrievalQuery(
            families=("ml_model_selection",),
            missing_techniques=(
                "k-fold cross validated ensemble",
                "stacking meta learner",
            ),
            stage_names=("model_training", "prediction_ensemble"),
        ),
        max_sequences=3,
    )

    assert sequences
    best = sequences[0]
    assert best.asset_family == "ml_model_selection"
    assert [operation.rule_name for operation in best.operations[:2]] == [
        "apply_kfold_ensemble",
        "apply_stacking_ensemble",
    ]
    assert best.intrusion_cost > 0


def test_operation_retrieval_can_rank_by_family_stage_and_context() -> None:
    clear_local_expansion_asset_caches()

    matches = ExpansionAssetRetriever().retrieve_operations(
        ExpansionRetrievalQuery(
            families=("ode_solver",),
            missing_techniques=("detect stiffness before advancing state",),
            stage_names=("evaluate_derivative", "advance_state"),
            intermediate_keys=("stiffness_ratio",),
        ),
        max_results=5,
    )

    assert matches
    assert matches[0].asset_family == "ode_solver"
    assert matches[0].rule_name == "insert_stiffness_detection_before_advance"
    assert "family:ode_solver" in matches[0].reasons


def test_retrieves_mined_ml_gap_operations() -> None:
    clear_local_expansion_asset_caches()

    sequences = ExpansionAssetRetriever().retrieve_sequences(
        ExpansionRetrievalQuery(
            families=("ml_model_selection",),
            missing_techniques=(
                "CatBoost and LightGBM ensemble",
                "Recursive Feature Elimination (RFE)",
                "Log-target transformation",
                "Target Encoding with smoothing",
            ),
            stage_names=("feature_engineering", "model_training", "prediction_ensemble"),
        ),
        max_sequences=3,
        max_operations_per_sequence=4,
    )

    assert sequences
    rule_names = {operation.rule_name for operation in sequences[0].operations}
    assert {
        "apply_tree_ensemble_blend",
        "insert_recursive_feature_elimination_before_estimator",
        "insert_log_target_transform_before_estimator",
        "insert_smoothed_target_encoding_before_estimator",
    }.issubset(rule_names)


def test_retrieves_mined_neural_network_gap_operations() -> None:
    clear_local_expansion_asset_caches()

    sequences = ExpansionAssetRetriever().retrieve_sequences(
        ExpansionRetrievalQuery(
            families=("neural_network", "deep_learning"),
            missing_techniques=(
                "Stochastic Weight Averaging (SWA)",
                "Mixed-precision training (bf16)",
                "Adversarial Weight Perturbation (AWP)",
                "Progressive Image Resizing",
            ),
            stage_names=("forward pass", "backward pass", "parameter update"),
        ),
        max_sequences=3,
        max_operations_per_sequence=4,
    )

    assert sequences
    rule_names = {operation.rule_name for operation in sequences[0].operations}
    assert {
        "insert_swa_checkpoint_averaging_after_update",
        "insert_mixed_precision_training_before_forward",
        "insert_adversarial_weight_perturbation_before_update",
        "insert_progressive_resizing_before_forward",
    }.issubset(rule_names)
