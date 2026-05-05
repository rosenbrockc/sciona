# Plan: Validation V2 — Semantic Technique Scoring, CDG Expansion, and Iterative Refinement

## Context

Validation V1 showed that LLM reranking correctly selects the right template
family (36% template match, 64% novel), but the competitive rate is bounded
by **keyword-based technique coverage measurement**. "EfficientNet-B0" in the
winning solution doesn't match "efficientnet_backbone" in the template,
even though the LLM knows they're the same thing.

Additionally, the current validation assumes a single-shot template match —
it doesn't test the architect's ability to **expand or refine** a base
template to better fit the specific competition, which is how the system
is designed to work in practice.

## Three workstreams

### A. Semantic Technique Coverage Scoring

**Problem**: Technique coverage uses keyword overlap between the winning
solution's `key_techniques` list and the matched template's stage text.
"5-fold stratified GroupKFold" doesn't match a stage describing
"cross_validation".

**Solution**: Use the LLM to evaluate technique coverage semantically
as part of the reranking step. Instead of keyword matching each technique,
ask the LLM: "Does this template cover this technique?"

**Implementation**:

Extend the reranker prompt to include technique evaluation:

```
For the best-matching template, also evaluate technique coverage.
For each technique from the winning solution, determine if the
template has a stage that implements it (even if using different
vocabulary).

"technique_coverage": [
  {
    "technique": "EfficientNet-B0 backbone",
    "covered": true,
    "matching_stage": "model_training",
    "reasoning": "Template has efficientnet_backbone architecture atom"
  },
  {
    "technique": "Mosaic augmentation",
    "covered": false,
    "matching_stage": null,
    "reasoning": "No mosaic/multi-image augmentation stage in template"
  }
]
```

This replaces the heuristic `evaluate_template_coverage()` function with
LLM-based evaluation in the `--rerank` path.

**Expected impact**: Competitive rate should jump from 8 (3%) to 40-60
(13-20%) because the LLM can recognize semantic equivalences that keywords
miss.

**File changes**:
- `sciona/architect/template_reranker.py` — extend prompt + response schema
- `scripts/validate_kaggle_batch.py` — use LLM technique coverage when available

### B. CDG Expansion/Refinement Templates

**Problem**: The 307 new competition prompts in the validation corpus each
have a winning solution summary. Many of these are variations of our
existing 125 CDGs — same problem family but with different preprocessing,
model choices, or post-processing. The current system treats them as
either "matches existing template" or "needs completely novel CDG."

**Solution**: Introduce **expansion CDGs** that reference a base template
and describe how to modify it for a specific competition variant. This is
lighter than creating a full new CDG — it says "start with X, then add/
swap/remove these stages."

**Schema for expansion CDGs**:

```json
{
  "asset_id": "expansion.kaggle.isic_2024_from_melanoma_1st",
  "base_template": "melanoma_1st",
  "expansion_type": "refinement",
  "modifications": [
    {
      "action": "swap_stage",
      "target_stage": "image_backbone_training",
      "replacement": {
        "stage_id": "convnext_efficientnet_dual_backbone",
        "description": "Train ConvNeXt-Tiny and EfficientNetV2-S in parallel",
        "concept_type": "neural_network"
      }
    },
    {
      "action": "add_stage",
      "after_stage": "image_backbone_training",
      "new_stage": {
        "stage_id": "lightgbm_metadata_branch",
        "description": "Train LightGBM on lesion/patient tabular metadata",
        "concept_type": "neural_network"
      }
    },
    {
      "action": "swap_stage",
      "target_stage": "rank_average_ensemble",
      "replacement": {
        "stage_id": "partial_auc_weighted_ensemble",
        "description": "Ensemble weighted by partial AUC performance",
        "concept_type": "analysis"
      }
    }
  ],
  "rationale": "ISIC 2024 adds 3D-TBP metadata and uses pAUC instead of AUC, requiring tabular branch and metric-specific weighting"
}
```

**Audit process**: For each of the 307 new competitions, classify as:
1. **Exact match** — an existing CDG covers it (the 8 competitive matches)
2. **Expansion** — start from an existing CDG, swap/add/remove 1-3 stages
3. **Novel** — fundamentally different, needs a new CDG

**Expected distribution**: ~30% exact/expansion, ~70% novel (based on the
64% "novel recommended" from LLM reranking).

**File changes**:
- `sciona-atoms/data/expansion_cdgs/` — new directory for expansion CDGs
- `sciona/architect/solution_index.py` — index expansion CDGs alongside base templates
- Validation script — evaluate expansions as part of template matching

### C. Iterative Refinement in Validation

**Problem**: The current validation is single-shot: match prompt → evaluate.
In practice, the architect proposes a CDG, the user (or an agent) reviews
it, and the architect refines based on feedback. The validation should
test this loop.

**Solution**: Add a multi-round validation mode:

1. **Round 1**: Architect proposes a CDG from template matching
2. **Round 2**: Evaluation agent identifies gaps (missing techniques,
   wrong model family, missing post-processing)
3. **Round 3**: Architect refines the CDG based on gap feedback
4. **Final**: Evaluate the refined CDG

This tests the architect's ability to converge on a competitive solution
through iteration, not just single-shot template matching.

**Implementation**:

```python
async def validate_with_refinement(
    prompt: str,
    solution_summary: str,
    key_techniques: list[str],
    sciona: Sciona,
    max_rounds: int = 2,
) -> ValidationResult:
    # Round 1: Initial proposal
    result = await sciona.propose(problem=prompt)
    evaluation = evaluate_coverage(result, key_techniques)

    for round_num in range(max_rounds):
        if evaluation.assessment == "competitive":
            break

        # Generate refinement feedback
        feedback = await generate_refinement_feedback(
            prompt, result, evaluation, solution_summary, sciona._llm
        )

        # Round N+1: Refined proposal
        refined_prompt = f"{prompt}\n\nRefinement guidance: {feedback}"
        result = await sciona.propose(problem=refined_prompt)
        evaluation = evaluate_coverage(result, key_techniques)

    return ValidationResult(
        final_assessment=evaluation.assessment,
        rounds_used=round_num + 1,
        initial_assessment=first_evaluation.assessment,
        improvement=evaluation.technique_coverage - first_evaluation.technique_coverage,
    )
```

**Expected impact**: Many "divergent" results should improve to "partial"
or "competitive" after 1-2 refinement rounds, since the feedback loop
tells the architect exactly what's missing.

**File changes**:
- `scripts/validate_kaggle_batch.py` — add `--refine` mode with max rounds
- `sciona/architect/template_reranker.py` — add refinement feedback generator
- `sciona/sdk.py` — no changes needed (propose() already works iteratively)

## Execution order

### Phase 1: Semantic technique scoring (highest impact, easiest)

Extend the reranker to evaluate technique coverage. Re-run validation.
Target: competitive rate from 3% to 15-20%.

### Phase 2: CDG audit for expansions

Audit 307 competitions to classify as exact/expansion/novel. Create
expansion CDGs for the ~50-80 that are close variants of existing templates.
This grows the effective template library from 125 to ~200 without
creating full new CDGs.

### Phase 3: Iterative refinement validation

Add multi-round validation. Test the architect's ability to converge.
This is the most realistic test of the system's actual capability.

## Files to create/modify

| File | Action | Phase |
|------|--------|-------|
| `sciona/architect/template_reranker.py` | Extend prompt with technique coverage | 1 |
| `scripts/validate_kaggle_batch.py` | Use LLM technique coverage, add --refine | 1, 3 |
| `sciona-atoms/data/expansion_cdgs/` | Create directory + expansion CDGs | 2 |
| `sciona/architect/solution_index.py` | Index expansion CDGs | 2 |
| `docs/EXPANSION_CDG_SCHEMA.md` | Document expansion CDG format | 2 |
