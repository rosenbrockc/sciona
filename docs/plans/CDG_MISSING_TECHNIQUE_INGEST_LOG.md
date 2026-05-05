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

## Batch 2: Second Support-Tier Pass

Source: `/tmp/sciona_expansion_gap_mining_after_batch1.json`, mined from
`validation_results_expansion_full.json` after Batch 1.

Deterministic approval: the miner produced 120 reusable-looking clusters with
support >= 2. The high-support tier contained 21 clusters with support >= 4
(5 clusters at support 5 and 16 clusters at support 4).

Semantic comparison to closest existing operations:

| Missing technique cluster | Closest existing operation(s) | Verdict | Action |
| --- | --- | --- | --- |
| 2D CNN + Bidirectional LSTM/GRU | `apply_dl_backbone_substitution` | Unique. Existing backbone substitution covers transfer learning, not sequence-specific CNN plus recurrent context modeling. | Added `insert_sequence_cnn_recurrent_backbone_before_loss`. |
| EfficientNet/ResNet/DenseNet/Inception/VGG/Swin/DeiT backbone ensembles | `apply_dl_backbone_substitution`, `apply_stacking_ensemble` | Unique but adjacent. Single-backbone substitution and generic stacking do not preserve pretrained-backbone ensemble intent. | Added `apply_pretrained_backbone_ensemble`; enriched single-backbone metadata with exact backbone families. |
| Generalized Mean pooling / GAP features | `insert_activation_statistics_after_forward` | Unique. Activation statistics observes activations; GeM/GAP changes feature aggregation. | Added `insert_gem_pooling_after_forward`. |
| DELF and XGBoost pair reranking | `apply_stacking_ensemble` | Unique. Stacking blends predictions; reranking reorders first-pass retrieval candidates using local or pairwise evidence. | Added `insert_retrieval_reranking_after_prediction`. |
| Macro F1 / MCC thresholding | `insert_constraint_injection` | Unique. Constraint correction is not metric-calibrated threshold selection. | Added `insert_metric_optimized_thresholding_after_prediction`. |
| GroupKFold by patient/session/cultivar | `force_cv_strategy`, `apply_kfold_ensemble` | Already conceptually covered, but deterministic triggers were too narrow. | Enriched `force_cv_strategy` with group-aware keys and planning-text diagnostics. |
| Multi-sample Dropout | `insert_activation_statistics_after_forward` | Unique. Existing diagnostics observe activations; multi-sample dropout averages multiple head passes. | Added `insert_multi_sample_dropout_before_loss`. |
| Hard negative / hard triplet mining | `insert_metric_optimized_thresholding_after_prediction` | Unique. Thresholding is post-prediction; hard-negative mining changes training pairs before loss. | Added `insert_hard_negative_mining_before_loss`. |
| LightGBM custom/quantile/RMSE loss | `replace_estimator_from_recommendation` | Unique. Estimator replacement chooses a model class; metric-aligned objective changes the loss being optimized. | Added `replace_loss_with_metric_aligned_objective`. |
| ArcFace and sub-center ArcFace | `insert_hard_negative_mining_before_loss` | Unique but same family. Both are angular-margin loss refinements, so one operation covers both. | Added `insert_arcface_margin_loss_before_loss`. |
| Permutation-importance feature selection | `insert_recursive_feature_elimination_before_estimator`, `insert_dimensionality_reduction_before_estimator` | Unique. RFE is recursive estimator-based selection; dimensionality reduction changes representation. | Added `insert_permutation_importance_feature_selection_before_estimator`. |
| Balanced rare-class over-sampling | `force_cv_strategy` | Unique. Validation splitting does not rebalance training examples. | Added `insert_balanced_sampling_before_training`. |
| Multi-label sigmoid auxiliary head | `apply_dl_backbone_substitution` | Unique. Backbone substitution does not add an independent per-label prediction head. | Added `insert_multilabel_sigmoid_head_before_loss`. |
| Multi-label focal/BCE/label-smoothing loss | `replace_loss_with_metric_aligned_objective` | Unique enough to keep in neural-network loss space; the ML objective rule covers estimator objectives, not DL per-label losses. | Added `insert_multilabel_focal_bce_loss_before_loss`. |
| StandardScaler for numerical features | `insert_preprocessing_before_estimator` | Covered, but deterministic triggers were too narrow. | Enriched preprocessing triggers for StandardScaler and generic feature scaling. |

Batch outcome: all 21 support >= 4 clusters are now either implemented as a
new reusable expansion/refinement operation or intentionally collapsed into a
broader existing/new operation with metadata and diagnostics.

## Batch 3: Support-3 Candidate Pass

Source: `/tmp/sciona_expansion_gap_mining_after_batch2.json`, mined after
Batch 2. The refreshed deterministic pass marked 48 clusters as covered and
left 30 support-3 clusters plus two metadata-only support-4 clusters as
candidate reusable operations.

Semantic comparison to closest existing operations:

| Missing technique cluster | Closest existing operation(s) | Verdict | Action |
| --- | --- | --- | --- |
| GroupKFold / Stratified GroupKFold variants | `force_cv_strategy`, `apply_kfold_ensemble` | Covered conceptually, but deterministic triggers missed stratified/group-specific terms. | Enriched `force_cv_strategy` metadata and diagnostics. |
| StandardScaler numerical preprocessing | `insert_preprocessing_before_estimator` | Covered conceptually, but deterministic triggers missed exact StandardScaler phrasing. | Enriched preprocessing metadata and diagnostics. |
| MixUp/CutMix/frequency/spatial/color/blur augmentation | `insert_progressive_resizing_before_forward` | Unique. Progressive resizing is a curriculum; these are data augmentation policies. | Added `insert_training_augmentation_before_forward`. |
| Sequence-level / transformer aggregation | `insert_sequence_cnn_recurrent_backbone_before_loss` | Unique enough. CNN/RNN backbones model sequence context differently from transformer aggregation. | Added `insert_transformer_sequence_aggregation_before_loss`. |
| Pseudo-labeling on unlabeled/test/external/domain data | `insert_balanced_sampling_before_training` | Unique. Sampling changes class frequency; pseudo-labeling expands supervision. | Added `insert_pseudo_labeling_loop_before_training`. |
| Iterative imputation | `insert_preprocessing_before_estimator` | Unique. Generic scaling/power transforms do not model missing values. | Added `insert_iterative_imputation_before_estimator`. |
| LLRD / CLR / Adam decay / weight decay | `insert_mixed_precision_training_before_forward` | Unique. Numeric policy is separate from optimizer scheduling. | Added `insert_optimizer_schedule_before_update`. |
| Domain-specific SFT/adapters and Xeno-Canto pretraining | `apply_dl_backbone_substitution` | Unique but adjacent. Backbone substitution chooses transfer architecture; this refines initialization/adaptation. | Added `insert_domain_specific_finetuning_before_forward`. |
| Bandpass/Butterworth/whitening filter phrases | `signal_filter` base CDG and `insert_passband_ripple_measurement_after_validate` | Not a new expansion; the base signal-filter CDG already represents filtering. | Enriched signal-filter metadata with exact phrases. |
| Adaptive Batch Normalization | `insert_activation_statistics_after_forward` | Unique. Activation statistics observes; AdaBN changes normalization state. | Added `insert_adaptive_batch_norm_before_forward`. |
| Boosted-tree early stopping | `apply_tree_ensemble_blend`, `replace_loss_with_metric_aligned_objective` | Unique. It constrains training iterations with validation feedback. | Added `insert_tree_early_stopping_validation`. |
| Dynamic ArcFace margin scaling | `insert_arcface_margin_loss_before_loss` | Covered by ArcFace-family margin-loss operation. | Collapsed into existing ArcFace metadata/diagnostics. |
| Siamese networks | `insert_hard_negative_mining_before_loss` | Unique. Mining selects examples; Siamese backbones change pair representation. | Added `insert_siamese_metric_backbone_before_loss`. |
| BCE / Dice+BCE loss | `insert_multilabel_focal_bce_loss_before_loss` | Unique enough for segmentation and mask losses. | Added `insert_dice_bce_loss_before_loss`. |
| Database Augmentation (DBA) | `insert_retrieval_reranking_after_prediction` | Unique. Reranking orders candidates; DBA smooths retrieval embeddings. | Added `insert_database_augmentation_for_retrieval`. |
| GNN interaction models | `apply_dl_backbone_substitution` | Unique. Graph message passing changes relational structure. | Added `insert_graph_interaction_network_before_loss`. |
| Dropout/L2 regularization | `insert_multi_sample_dropout_before_loss` | Unique enough. Multi-sample dropout averages predictions; generic regularization covers dropout/L2 policies. | Added `insert_regularization_before_loss`. |
| PointRend boundary refinement | `insert_dice_bce_loss_before_loss` | Unique. Loss changes training objective; PointRend refines boundary prediction. | Added `insert_pointrend_boundary_refinement_after_forward`. |
| MegaDetector / ROI cropping | `apply_dl_backbone_substitution` | Unique. It changes input region selection before the main model. | Added `insert_roi_cropping_before_forward`. |
| Feature Hashing | `insert_smoothed_target_encoding_before_estimator` | Unique. Target encoding uses target means; hashing bounds sparse feature dimensionality. | Added `insert_feature_hashing_before_estimator`. |
| Chain-of-thought prompt augmentation | `insert_prompt_reasoning_augmentation_before_training` | Unique for LLM-style tasks but still represented as a pre-training/pre-inference augmentation. | Added `insert_prompt_reasoning_augmentation_before_training`. |
| Bradley-Terry preference ranking | `insert_metric_optimized_thresholding_after_prediction` | Unique. Thresholding is post-prediction; preference modeling changes pairwise supervision. | Added `insert_preference_ranking_head_before_loss`. |
| DeBERTa cross-encoder | `apply_pretrained_backbone_ensemble` | Unique enough. Cross-encoders jointly encode pairs for relevance/ranking. | Added `insert_cross_encoder_backbone_before_loss`. |

Batch outcome: the support-3 tier is now represented by reusable operations or
collapsed into existing operations with explicit metadata hooks. Manifest
closure after this batch reports 34 assets and 182 operations.
