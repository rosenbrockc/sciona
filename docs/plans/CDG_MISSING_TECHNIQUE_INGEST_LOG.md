# Missing Technique CDG Ingest Log

## Batch 1: High-Support Reusable Operations

Source: `validation_results_expansion_gap_mining.json`.

Deterministic approval: Phase 9 mining clustered these as recurring
`candidate_reusable_operation` gaps with support >= 5.

Semantic comparison to closest existing operations:

| Missing technique cluster | Closest existing operation(s) | Verdict | Action |
| --- | --- | --- | --- |
| Stochastic Weight Averaging (SWA) | `insert_weight_distribution_check_after_update`, `insert_loss_convergence_monitoring_after_loss` | Unique. Existing operations observe training health but do not average late checkpoints. | Added `insert_swa_checkpoint_averaging_after_update`. |
| Mixed-precision training (bf16/fp16) | `insert_gradient_explosion_detection_after_backward`, `apply_dl_backbone_substitution` | Unique. Existing operations monitor gradients or swap backbones but do not change numeric execution policy. | Added `insert_mixed_precision_training_before_forward`. |
| Adversarial Weight Perturbation (AWP) | `insert_weight_distribution_check_after_update`, `insert_gradient_explosion_detection_after_backward` | Unique. Existing operations observe weights/gradients but do not perturb weights during training. | Added `insert_adversarial_weight_perturbation_before_update`. |
| Progressive Image Resizing | `apply_dl_backbone_substitution` | Unique but adjacent. Existing backbone substitution mentions transfer-learning/TTA; progressive resizing is a training curriculum. | Added `insert_progressive_resizing_before_forward`. |
| CatBoost/LightGBM/XGBoost tree ensembles | `replace_estimator_from_recommendation`, `apply_stacking_ensemble` | Unique. Existing operations pick one estimator or stack arbitrary level-one models; mined gaps require heterogeneous tree blending metadata. | Added `apply_tree_ensemble_blend`. |
| Recursive Feature Elimination (RFE) | `insert_dimensionality_reduction_before_estimator` | Unique. Dimensionality reduction changes representation; RFE selects an estimator-driven feature subset. | Added `insert_recursive_feature_elimination_before_estimator`. |
| Log-target transformation | `insert_preprocessing_before_estimator` | Unique. Existing preprocessing transforms features; log-target transform changes target training space and requires inverse prediction transform. | Added `insert_log_target_transform_before_estimator`. |
| Smoothed target encoding | `insert_preprocessing_before_estimator` | Unique. Existing preprocessing is feature scaling/power transform; smoothed target encoding is fold-aware categorical encoding with leakage constraints. | Added `insert_smoothed_target_encoding_before_estimator`. |

Metadata enrichment: the provider asset descriptions include the exact mined
phrases and close variants so deterministic retrieval can match future
solution summaries without requiring one-off competition patches.

