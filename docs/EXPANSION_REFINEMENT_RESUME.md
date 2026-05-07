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

After the candidate-operation, LightGBM large-leaf, large-backbone scale-attention, lightweight CNN regression, and parallel path optimization passes, the expanded focused suite result is `29 passed`.

Latest full deterministic validation after the parallel path optimization pass:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=. python scripts/validate_kaggle_batch.py \
  --corpus /Users/conrad/personal/sciona-atoms/research/validation_corpus.json \
  --start 0 --end 307 \
  --output /tmp/sciona_validation_full_20260507_parallel_path_optimization_v1.json \
  --expansion-rounds 2
```

Latest full validation summary:

- Strict: `103 competitive`, `114 partial`, `90 divergent`
- Trick availability reference: `103 competitive`, `104 partial`, `9 partial+trick_available`, `1 partial+high_risk_trick_suppressed`, `67 divergent`, `18 divergent+trick_available`, `5 divergent+high_risk_trick_suppressed`
- Rescued by expansion/refinement: `108`

Latest follow-up report:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=. python scripts/review_validation_followups.py \
  /tmp/sciona_validation_full_20260507_parallel_path_optimization_v1.json \
  --output /tmp/sciona_validation_followup_20260507_parallel_path_optimization_v1.json \
  --min-support 2 \
  --similarity-threshold 0.34 \
  --max-clusters 80
```

Follow-up summary:

- `90` remaining divergent
- `33` trick review tickets
- `80` divergent gap clusters
- `3` candidate reusable-operation clusters
- `77` existing-operation clusters

## Remaining Candidate Clusters

These look less like safe metadata-only cleanup and more like new operation/CDG decisions or trick catalog items:

- `gap_cluster_267`: 3D-UNet spatio-temporal attention
- `gap_cluster_290`: MCTS/backtracking search
- `gap_cluster_298`: residual/wind-flow attention

Recommendation: stop metadata-only enrichment here. For each remaining cluster, decide whether it is:

- A genuinely reusable expansion/refinement operation with runtime support.
- A new base CDG/topology.
- A trick catalog entry exposed to the architect but not counted as a strict match.
- Too competition-specific to encode at this stage.

## Current Repo State Notes

The parallel path optimization pass touched these matcher files:

- `docs/EXPANSION_REFINEMENT_RESUME.md`
- `sciona/principal/expansion_rules/graph_optimization.py`
- `tests/test_graph_optimization_expansion_assets.py`

It still has unrelated untracked local artifacts:

- `docs/symbolic_math.pdf`
- `validation_results_3.json`
- `validation_results_4.json`
- `validation_results_6.json`

Do not stage the unrelated artifacts unless explicitly requested.

`/Users/conrad/personal/sciona-atoms-ml` may have unrelated local coordinate-descent work. Do not stage or edit it for expansion follow-up passes.

`/Users/conrad/personal/sciona-atoms-dl` is clean after the lightweight CNN regression pass.

The parallel path optimization pass touched this provider asset:

- `data/expansions/graph_optimization.json`

## Suggested Next Step

Start the next technical pass from the 3 remaining candidate clusters. The safest next decisions are probably whether MCTS/backtracking should be a trick catalog entry or a reusable search-operation expansion, whether residual/wind-flow attention is broad enough for a runtime-backed attention refinement, and whether 3D-UNet spatio-temporal attention is broad enough for a runtime-backed architecture operation.
