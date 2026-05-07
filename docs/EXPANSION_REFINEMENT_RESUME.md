# Expansion/Refinement Work Resume

Last updated: 2026-05-07

## Current Goal

Continue improving CDG expansion/refinement coverage without overfitting validation by adding competition-specific aliases or descriptions. Metadata additions should be reusable, general, and tied to stable operation contracts.

## Guardrail

Allowed:

- Dejargonized context that explains when a reusable operation applies.
- Standard technique names when they are broadly recognized, such as `LightGBM`, `XGBoost`, `CatBoost`, `StratifiedGroupKFold`, `MAP@K`, `QWK`, `Levenshtein`, `Jaccard`, or `cosine similarity`.
- Trigger keys, rewrite summaries, and applicability notes that describe generic preconditions and information-flow effects.

Avoid:

- Competition names, dataset names, leaderboard-specific phrasing, or exact problem-description bait.
- Stuffing aliases/descriptions to make deterministic matching pass for a single validation case.
- Adding target-encoding language to unrelated operations such as TTA.

## Latest Edits

Provider expansion assets were updated in sibling repos:

- `/Users/conrad/personal/sciona-atoms-ml/data/expansions/ml_model_selection.json`
  - Generalized metadata for CV strategy, k-fold ensembling, tree ensemble blending, pseudo-labeling, tree early stopping, metric-optimized thresholding, retrieval reranking, RFE, and smoothed target encoding.
  - Added generic trigger/context vocabulary for entity/time/geographic-safe CV, boosted-tree blends, fold-local target encoding, QWK/MAP@K thresholding, and similarity/edit-distance reranking.

- `/Users/conrad/personal/sciona-atoms-dl/data/expansions/neural_network.json`
  - Generalized metadata for mixed precision, progressive/high-resolution image training, label-preserving training augmentation, domain-specific fine-tuning, TTA, and stochastic depth.

Follow-up candidate-operation pass added runtime-backed reusable operations:

- `/Users/conrad/personal/sciona-atoms-ml/data/expansions/ml_model_selection.json`
  - Added entity embedding encoding, rank-correlation objective, and retrieval candidate-generation operation contracts.

- `/Users/conrad/personal/sciona-atoms-dl/data/expansions/neural_network.json`
  - Added coordinate regression head operation contract.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/ml_model_selection.py`
  - Added runtime rule builders and diagnostics for entity embeddings, rank-correlation objectives, and candidate generation.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/neural_network.py`
  - Added runtime rule builder and diagnostic for coordinate regression heads.

- `/Users/conrad/personal/sciona-matcher/tests/test_ml_model_selection_expansion_assets.py`
- `/Users/conrad/personal/sciona-matcher/tests/test_neural_network_mined_expansion_assets.py`
  - Added asset/runtime coverage for the new reusable operations.

LightGBM large-leaf follow-up pass added one runtime-backed hyperparameter refinement:

- `/Users/conrad/personal/sciona-atoms-ml/data/expansions/ml_model_selection.json`
  - Added LightGBM large-leaf configuration operation contract.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/ml_model_selection.py`
  - Added runtime rule builder and diagnostic for validation-controlled LightGBM large-leaf configuration.

- `/Users/conrad/personal/sciona-matcher/tests/test_ml_model_selection_expansion_assets.py`
  - Added asset/runtime coverage for the LightGBM configuration operation.

Large-backbone scale-attention follow-up pass added one runtime-backed neural architecture refinement:

- `/Users/conrad/personal/sciona-atoms-dl/data/expansions/neural_network.json`
  - Added large-backbone scale-attention operation contract.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/neural_network.py`
  - Added runtime rule builder and diagnostic for high-capacity vision backbones with scale-aware attention.

- `/Users/conrad/personal/sciona-matcher/tests/test_neural_network_mined_expansion_assets.py`
  - Added asset/runtime coverage for the large-backbone scale-attention operation.

Lightweight CNN regression follow-up pass added one runtime-backed neural architecture refinement:

- `/Users/conrad/personal/sciona-atoms-dl/data/expansions/neural_network.json`
  - Added lightweight CNN regression head operation contract.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/neural_network.py`
  - Added runtime rule builder and diagnostic for compact CNN + pooling regression or task heads.

- `/Users/conrad/personal/sciona-matcher/tests/test_neural_network_mined_expansion_assets.py`
  - Added asset/runtime coverage for the lightweight CNN regression operation.

Parallel path optimization follow-up pass added one runtime-backed graph/search refinement:

- `/Users/conrad/personal/sciona-atoms-cs/data/expansions/graph_optimization.json`
  - Added parallel path optimization and merging operation contract.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/graph_optimization.py`
  - Added runtime rule builder and diagnostic for optimizing independent path candidates in parallel before final path extraction.

- `/Users/conrad/personal/sciona-matcher/tests/test_graph_optimization_expansion_assets.py`
  - Added provider asset and runtime coverage for the parallel path optimization operation.

Spatio-temporal U-Net attention follow-up pass added one runtime-backed neural architecture refinement:

- `/Users/conrad/personal/sciona-atoms-dl/data/expansions/neural_network.json`
  - Added 3D U-Net-style spatio-temporal attention operation contract.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/neural_network.py`
  - Added runtime rule builder and diagnostic for volumetric, slice-stack, frame-stack, or video-like tensors that need spatio-temporal attention.

- `/Users/conrad/personal/sciona-matcher/tests/test_neural_network_mined_expansion_assets.py`
  - Added provider asset and runtime coverage for the spatio-temporal U-Net attention operation.

MCTS/backtracking search follow-up pass added one runtime-backed planning/search refinement:

- `/Users/conrad/personal/sciona-atoms/data/expansions/agent_simulation_search_planning.json`
  - Added Monte Carlo Tree Search and backtracking search operation contract.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/agent_simulation_search_planning.py`
  - Added runtime rule builder and diagnostic for candidate-state tree search, MCTS, and backtracking inside planning loops.

- `/Users/conrad/personal/sciona-matcher/tests/test_agent_simulation_search_planning_expansion_assets.py`
  - Added retrieval and runtime coverage for the MCTS/backtracking search operation.

Flow-aware residual attention follow-up pass added one runtime-backed neural architecture refinement:

- `/Users/conrad/personal/sciona-atoms-dl/data/expansions/neural_network.json`
  - Added flow-aware residual attention operation contract.

- `/Users/conrad/personal/sciona-matcher/sciona/principal/expansion_rules/neural_network.py`
  - Added runtime rule builder and diagnostic for residual attention guided by motion, wind, flow, or displacement fields.

- `/Users/conrad/personal/sciona-matcher/tests/test_neural_network_mined_expansion_assets.py`
  - Added provider asset and runtime coverage for the flow-aware residual attention operation.

## Validation Artifacts

Focused tests:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=. pytest -q \
  tests/test_ml_model_selection_expansion_assets.py \
  tests/test_neural_network_mined_expansion_assets.py \
  tests/test_expansion_retrieval.py \
  tests/test_expansion_gap_mining.py \
  tests/test_validation_followup.py
```

Result: `26 passed`.

After the candidate-operation, LightGBM large-leaf, large-backbone scale-attention, lightweight CNN regression, parallel path optimization, spatio-temporal U-Net attention, MCTS/backtracking search, flow-aware residual attention, and retrieval-stability passes, the expanded focused suite result is `40 passed`.

Latest full deterministic validation after the retrieval-stability fix:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=. python scripts/validate_kaggle_batch.py \
  --corpus /Users/conrad/personal/sciona-atoms/research/validation_corpus.json \
  --start 0 --end 307 \
  --output /tmp/sciona_validation_full_20260507_flow_residual_attention_retrieval_fix_v1.json \
  --expansion-rounds 2
```

Latest full validation summary:

- Strict: `103 competitive`, `115 partial`, `89 divergent`
- Trick availability reference: `103 competitive`, `105 partial`, `9 partial+trick_available`, `1 partial+high_risk_trick_suppressed`, `66 divergent`, `18 divergent+trick_available`, `5 divergent+high_risk_trick_suppressed`
- Rescued by expansion/refinement: `109`
- Note: the flow-aware residual attention operation initially shifted `isic-2019` from competitive to partial by pushing `insert_balanced_sampling_before_training` out of the top-40 global operation pool before sequence grouping. `sciona/principal/expansion_retrieval.py` now carries all indexed matching operations into family grouping, restoring `isic-2019` to competitive while preserving `0` reusable-operation candidates.

Latest follow-up report:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=. python scripts/review_validation_followups.py \
  /tmp/sciona_validation_full_20260507_flow_residual_attention_retrieval_fix_v1.json \
  --output /tmp/sciona_validation_followup_20260507_flow_residual_attention_retrieval_fix_v1.json \
  --min-support 2 \
  --similarity-threshold 0.34 \
  --max-clusters 80
```

Follow-up summary:

- `89` remaining divergent
- `33` trick review tickets
- `80` divergent gap clusters
- `0` candidate reusable-operation clusters
- `80` existing-operation clusters

## Remaining Candidate Clusters

No reusable-operation candidate clusters remain at `--min-support 2 --similarity-threshold 0.34 --max-clusters 80`.

Recommendation: stop metadata-only enrichment here. For future clusters, decide whether each is:

- A genuinely reusable expansion/refinement operation with runtime support.
- A new base CDG/topology.
- A trick catalog entry exposed to the architect but not counted as a strict match.
- Too competition-specific to encode at this stage.

## Latest Edit Notes

The retrieval-stability pass touched these matcher files:

- `docs/EXPANSION_REFINEMENT_RESUME.md`
- `sciona/principal/expansion_retrieval.py`
- `tests/test_expansion_delta_planner.py`

The fix prevents global pre-group truncation from dropping a useful lower-scored operation before family sequence selection. Regression coverage mirrors the `isic-2019` validation query and asserts the restored `apply_dl_backbone_substitution` + `insert_balanced_sampling_before_training` pack.

## Current Repo State Notes

It still has unrelated untracked local artifacts:

- `docs/symbolic_math.pdf`
- `validation_results_3.json`
- `validation_results_4.json`
- `validation_results_6.json`

Do not stage the unrelated artifacts unless explicitly requested.

`/Users/conrad/personal/sciona-atoms-ml` may have unrelated local coordinate-descent work. Do not stage or edit it for expansion follow-up passes.

The flow-aware residual attention pass touched this provider asset:

- `/Users/conrad/personal/sciona-atoms-dl/data/expansions/neural_network.json`

`/Users/conrad/personal/sciona-atoms-cs` is clean after the parallel path optimization pass.

`/Users/conrad/personal/sciona-atoms` is clean after the MCTS/backtracking search pass.

## Suggested Next Step

Stop the candidate-cluster loop for this validation configuration. The `isic-2019` strict regression has been fixed; the next useful review is trick-review tickets, not another reusable-operation candidate.
