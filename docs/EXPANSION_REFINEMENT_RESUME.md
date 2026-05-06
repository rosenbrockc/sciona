# Expansion/Refinement Work Resume

Last updated: 2026-05-06

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

No files in `sciona-matcher` were changed by the guarded metadata pass itself.

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

Latest full deterministic validation:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=. python scripts/validate_kaggle_batch.py \
  --corpus /Users/conrad/personal/sciona-atoms/research/validation_corpus.json \
  --start 0 --end 307 \
  --output /tmp/sciona_validation_full_20260506_guarded_expansion_metadata_v3.json \
  --expansion-rounds 2
```

Latest full validation summary:

- Strict: `102 competitive`, `111 partial`, `94 divergent`
- Trick availability reference: `102 competitive`, `102 partial`, `8 partial+trick_available`, `1 partial+high_risk_trick_suppressed`, `71 divergent`, `19 divergent+trick_available`, `4 divergent+high_risk_trick_suppressed`
- Rescued by expansion/refinement: `103`

Latest follow-up report:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=. python scripts/review_validation_followups.py \
  /tmp/sciona_validation_full_20260506_guarded_expansion_metadata_v3.json \
  --output /tmp/sciona_validation_followup_20260506_guarded_expansion_metadata_v3.json \
  --min-support 2 \
  --similarity-threshold 0.34 \
  --max-clusters 80
```

Follow-up summary:

- `94` remaining divergent
- `32` trick review tickets
- `80` divergent gap clusters
- `11` candidate reusable-operation clusters
- `69` existing-operation clusters

## Remaining Candidate Clusters

These look less like safe metadata-only cleanup and more like new operation/CDG decisions or trick catalog items:

- `gap_cluster_052`: 3D coordinate / CDF / MLP regression heads
- `gap_cluster_086`: LightGBM large-leaf configuration
- `gap_cluster_132`: Kendall Tau optimization
- `gap_cluster_153`: ConvNeXt/EfficientNet large-backbone scale attention
- `gap_cluster_185`: entity embeddings
- `gap_cluster_196`: shallow CNN / CNN regressor
- `gap_cluster_225`: CNN or graph-based candidate generation
- `gap_cluster_262`: parallel path optimization / path merging
- `gap_cluster_280`: 3D-UNet spatio-temporal attention
- `gap_cluster_303`: MCTS/backtracking search
- `gap_cluster_311`: residual/wind-flow attention

Recommendation: stop metadata-only enrichment here. For each remaining cluster, decide whether it is:

- A genuinely reusable expansion/refinement operation with runtime support.
- A new base CDG/topology.
- A trick catalog entry exposed to the architect but not counted as a strict match.
- Too competition-specific to encode at this stage.

## Current Repo State Notes

`/Users/conrad/personal/sciona-matcher` has untracked work from the validation-followup phase:

- `sciona/architect/validation_followup.py`
- `scripts/review_validation_followups.py`
- `tests/test_validation_followup.py`

It also has unrelated untracked local artifacts:

- `docs/symbolic_math.pdf`
- `validation_results_3.json`
- `validation_results_4.json`
- `validation_results_6.json`

Do not stage the unrelated artifacts unless explicitly requested.

`/Users/conrad/personal/sciona-atoms-ml` had unrelated pre-existing local changes before this pass. The guarded metadata pass only touched:

- `data/expansions/ml_model_selection.json`

`/Users/conrad/personal/sciona-atoms-dl` only has:

- `data/expansions/neural_network.json`

## Suggested Next Step

Review and commit the provider asset edits separately from matcher follow-up tooling:

1. In `sciona-atoms-ml`, commit only `data/expansions/ml_model_selection.json`.
2. In `sciona-atoms-dl`, commit only `data/expansions/neural_network.json`.
3. In `sciona-matcher`, commit the validation follow-up tooling plus this resume file if desired, but leave unrelated untracked artifacts alone.
4. Start the next technical pass from the 11 remaining candidate clusters, beginning with the ones that clearly map to reusable operations: entity embeddings, Kendall Tau objective, 3D coordinate/regression heads, and candidate generation.
