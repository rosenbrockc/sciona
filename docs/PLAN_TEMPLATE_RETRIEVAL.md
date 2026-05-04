# Implementation Plan: Dejargonize → Top-N → LLM Rerank Template Retrieval

## Context

The Kaggle validation showed 81% "divergent" results — the keyword matcher
finds the right template *family* but technique-level vocabulary doesn't
overlap. "EfficientNetV2-S" in the prompt doesn't match "efficientnet_backbone"
in the template. Grounding is 100%, so the atoms are there. The bottleneck
is template selection.

### What exists

- **TemplateRetriever** (`architect/template_retriever.py`) — 3-phase Memgraph
  cascade + alignment scoring. Works within-decomposition on CDG subgraphs,
  NOT on natural-language prompts.
- **MacroArtifactRetriever** (`services/artifact_retrieval.py`) — pre-decomposition
  goal→artifact matching via token overlap. Works on text but has no
  dejargonization or LLM reranking.
- **Dejargonized descriptions** — backfill script populates `atom_descriptions`
  table with plain-language descriptions. CDG templates have
  `dejargonized_summary` fields but they're populated during CDG authoring,
  not at retrieval time.
- **Solution CDG templates** (125 in `data/solution_cdgs/`) — have `summary`,
  `dejargonized_summary`, `applicability.use_when`, `applicability.key_insight`,
  `family`, `paradigm` fields.

### What's missing

1. **Prompt dejargonizer** — normalize inbound user prompt to canonical ML vocabulary
2. **Solution CDG template index** — expose 125 templates to the retrieval path
3. **LLM reranking** — semantic comparison of prompt against template applicability
4. **Integration** — wire into the `select_strategy` node or `SingleAgentPlanner`

## Architecture

```
User prompt
    │
    ▼
┌──────────────────┐
│ Dejargonize      │  Normalize ML jargon to canonical terms
│ (deterministic   │  "EfficientNetV2-S" → "cnn image backbone"
│  or lightweight  │  "5-fold stratified GroupKFold" → "stratified group cross validation"
│  LLM)            │
└──────┬───────────┘
       │ dejargonized_prompt
       ▼
┌──────────────────┐
│ Phase 1: Top-N   │  Keyword + embedding match against
│ Candidate        │  template dejargonized_summary + use_when
│ Retrieval        │
└──────┬───────────┘
       │ top_n_candidates (N=10)
       ▼
┌──────────────────┐
│ Phase 2: LLM     │  Compare prompt against each candidate's
│ Rerank + Select  │  full applicability block
│                  │  (use_when, do_not_use_when, key_insight,
│                  │   failure_modes, scaling_notes)
└──────┬───────────┘
       │ selected_template (or "compose_novel")
       ▼
┌──────────────────┐
│ Phase 3: Ground  │  Bind template stages to atoms
│ + Adapt          │  (already works — 100% grounding)
└──────────────────┘
```

## Implementation

### Step 1: Prompt Dejargonizer

**File**: `sciona/architect/dejargonizer.py`

Two modes:
- **Heuristic mode** (fast, no LLM): regex-based synonym replacement
- **LLM mode** (better, 1 API call): ask LLM to rewrite prompt in canonical terms

```python
# Heuristic synonym table
_JARGON_MAP: dict[str, str] = {
    # Architecture names → canonical
    "efficientnet": "cnn image backbone",
    "efficientnetv2": "cnn image backbone",
    "resnet": "cnn image backbone",
    "resnext": "cnn image backbone",
    "densenet": "cnn image backbone",
    "vgg": "cnn image backbone",
    "inception": "cnn image backbone",
    "swin": "vision transformer backbone",
    "vit": "vision transformer backbone",
    "deit": "vision transformer backbone",
    "bert": "text transformer encoder",
    "roberta": "text transformer encoder",
    "deberta": "text transformer encoder",
    "xlm": "multilingual text transformer",
    "gpt": "autoregressive language model",
    "llama": "autoregressive language model",
    "whisper": "speech recognition model",
    "yolo": "single stage object detector",
    "yolov5": "single stage object detector",
    "faster rcnn": "two stage object detector",
    "mask rcnn": "instance segmentation detector",
    "unet": "encoder decoder segmentation",
    "deeplabv3": "encoder decoder segmentation",
    "lightgbm": "gradient boosting",
    "xgboost": "gradient boosting",
    "catboost": "gradient boosting",

    # Technique names → canonical
    "cutmix": "image augmentation mixing",
    "mixup": "image augmentation mixing",
    "cutout": "image augmentation masking",
    "gridmask": "image augmentation masking",
    "specaugment": "spectrogram augmentation masking",
    "tta": "test time augmentation",
    "swa": "stochastic weight averaging",
    "ohem": "hard example mining",
    "focal loss": "class imbalance loss",
    "dice loss": "segmentation overlap loss",
    "lovasz": "segmentation overlap loss",
    "arcface": "angular margin metric learning",
    "triplet loss": "metric learning loss",
    "cosine similarity": "embedding distance metric",
    "faiss": "approximate nearest neighbor index",

    # Training patterns → canonical
    "groupkfold": "group cross validation",
    "stratifiedkfold": "stratified cross validation",
    "pseudo labeling": "self training",
    "knowledge distillation": "teacher student training",
    "label smoothing": "soft target regularization",
    "warmup": "learning rate warmup schedule",
    "cosine annealing": "learning rate cosine schedule",
    "onecyclelr": "learning rate one cycle schedule",
}


def dejargonize_heuristic(prompt: str) -> str:
    """Replace known jargon with canonical terms."""
    result = prompt.lower()
    for jargon, canonical in sorted(
        _JARGON_MAP.items(), key=lambda x: -len(x[0])
    ):
        result = result.replace(jargon.lower(), canonical)
    return result


async def dejargonize_llm(prompt: str, llm: LLMClient) -> str:
    """Use LLM to normalize prompt vocabulary."""
    system = (
        "Rewrite the following machine learning problem description using "
        "generic, canonical terminology. Replace specific model names with "
        "their category (e.g., 'EfficientNet-B4' → 'CNN image backbone'), "
        "specific technique names with their function (e.g., 'CutMix' → "
        "'image augmentation with region mixing'), and framework-specific "
        "jargon with plain descriptions. Keep the problem structure and "
        "data description intact. Output only the rewritten text."
    )
    response = await llm.complete(system=system, user=prompt)
    return response.text
```

### Step 2: Solution CDG Template Index

**File**: `sciona/architect/solution_index.py`

Build a searchable index from the 125 solution CDG templates:

```python
@dataclass
class SolutionTemplate:
    """A solution CDG template indexed for retrieval."""
    name: str                          # e.g., "melanoma_1st"
    family: str                        # e.g., "medical_image_tabular"
    paradigm: str                      # e.g., "classification"
    summary: str                       # Technical summary
    dejargonized_summary: str          # Plain-language summary
    use_when: list[str]                # Applicability conditions
    do_not_use_when: list[str]         # Contraindications
    key_insight: str                   # Load-bearing idea
    critical_stages: list[str]
    swappable_stages: list[str]
    failure_modes: list[str]
    scaling_notes: str
    stage_names: list[str]             # All stage IDs
    stage_descriptions: str            # Concatenated stage descriptions
    grounding_rate: float              # From bindings
    cdg_path: Path                     # Path to JSON file


class SolutionTemplateIndex:
    """Searchable index of solution CDG templates."""

    def __init__(self, templates: list[SolutionTemplate]) -> None:
        self._templates = {t.name: t for t in templates}
        self._catalog: PrimitiveCatalog | None = None

    @classmethod
    def from_directory(cls, cdg_dir: Path) -> SolutionTemplateIndex:
        """Load all solution CDGs + bindings from a directory."""
        ...

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        family_filter: str | None = None,
    ) -> list[tuple[SolutionTemplate, float]]:
        """Keyword search against dejargonized summaries + use_when.

        Returns top-k (template, score) pairs.
        """
        query_tokens = _tokenize(query)
        scored = []
        for template in self._templates.values():
            # Build searchable text from dejargonized fields
            search_text = " ".join([
                template.dejargonized_summary,
                " ".join(template.use_when),
                template.family,
                template.paradigm,
                template.stage_descriptions,
            ])
            template_tokens = _tokenize(search_text)

            # TF-IDF-like scoring
            overlap = query_tokens & template_tokens
            score = sum(
                self._idf.get(t, 1.0) for t in overlap
            )

            # Family bonus
            if family_filter and template.family == family_filter:
                score += 5.0

            if score > 0:
                scored.append((template, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]
```

### Step 3: LLM Reranker

**File**: `sciona/architect/template_reranker.py`

```python
async def rerank_templates(
    prompt: str,
    candidates: list[SolutionTemplate],
    llm: LLMClient,
    max_candidates: int = 5,
) -> list[tuple[SolutionTemplate, float, str]]:
    """Use LLM to rerank candidate templates by semantic fit.

    Returns list of (template, confidence, reasoning) sorted by fit.
    """
    # Truncate to max_candidates for cost control
    candidates = candidates[:max_candidates]

    system = TEMPLATE_RERANK_SYSTEM
    user = _format_rerank_prompt(prompt, candidates)
    response = await llm.complete(system=system, user=user)
    rankings = _parse_rerank_response(response.text)
    return rankings


TEMPLATE_RERANK_SYSTEM = """You are an expert ML architect comparing a problem
description against candidate solution templates.

For each candidate, evaluate:
1. Does the problem type match? (classification vs regression vs detection etc.)
2. Does the data modality match? (tabular vs image vs text vs time_series etc.)
3. Are the key challenges similar? (class imbalance, noisy labels, domain shift etc.)
4. Would the template's critical stages apply to this problem?
5. Are there contraindications (do_not_use_when) that disqualify this template?

Output JSON:
{
  "rankings": [
    {
      "template": "<name>",
      "score": 0.0-1.0,
      "reasoning": "1-2 sentences"
    }
  ],
  "best_match": "<name or 'none'>",
  "should_compose_novel": true/false,
  "novel_reasoning": "Why no template fits (if applicable)"
}
"""
```

### Step 4: Integration into Architect Pipeline

Wire the 3-phase retrieval into the existing flow. Two integration points:

#### Option A: Pre-decomposition (in SingleAgentPlanner)

Add solution template matching before macro artifact matching:

```python
# In SingleAgentPlanner.run():
async def run(self, goal: str) -> PlannerRunResult:
    # NEW: Phase 0 — Solution template matching
    dejargonized = dejargonize_heuristic(goal)
    candidates = self._solution_index.search(dejargonized, k=10)
    if candidates:
        ranked = await rerank_templates(goal, candidates, self._llm)
        if ranked and ranked[0][1] >= 0.7:
            # Use this template as the CDG
            template_cdg = load_cdg(ranked[0][0].cdg_path)
            return self._template_match_result(goal, template_cdg, ranked[0])

    # Existing flow: macro match → direct match → decomposition
    ...
```

#### Option B: Within select_strategy (in DecompositionAgent)

Add template search after conjugate detection, before LLM paradigm selection:

```python
# In select_strategy():
async def select_strategy(state, config):
    goal = state["goal"]

    # Existing: conjugate pair detection
    conjugate = _detect_conjugate_pair(goal)
    if conjugate:
        return {"paradigm": "conjugate", ...}

    # NEW: Solution template matching
    dejargonized = dejargonize_heuristic(goal)
    candidates = solution_index.search(dejargonized, k=10)
    if candidates:
        ranked = await rerank_templates(goal, candidates, llm)
        if ranked and ranked[0][1] >= 0.7:
            return {"paradigm": "template_match",
                    "template": ranked[0][0],
                    "confidence": ranked[0][1]}

    # Existing: LLM paradigm selection
    ...
```

**Recommendation: Option A** — the planner is the right place because it
can short-circuit the entire decomposition pipeline when a strong template
match exists. This is analogous to how macro artifacts already work.

### Step 5: Validate and Iterate

Re-run the 307-competition validation with the new pipeline:

```python
# In validate_kaggle_batch.py, replace keyword matching with:
dejargonized = dejargonize_heuristic(prompt)
candidates = solution_index.search(dejargonized, k=10)
# For offline validation, use heuristic dejargonization only (no LLM)
# For online validation, add LLM reranking
```

Expected improvement: technique coverage should jump from 24% to 50-70%
because dejargonized vocabulary aligns ("cnn image backbone" matches
"cnn backbone" in template descriptions).

## Files to create/modify

| File | Action |
|------|--------|
| `sciona/architect/dejargonizer.py` | **Create** — heuristic + LLM dejargonization |
| `sciona/architect/solution_index.py` | **Create** — SolutionTemplateIndex |
| `sciona/architect/template_reranker.py` | **Create** — LLM reranking |
| `sciona/services/planner_service.py` | **Modify** — add Phase 0 template matching |
| `sciona/sdk.py` | **Modify** — expose `propose()` with template matching |
| `scripts/validate_kaggle_batch.py` | **Modify** — use dejargonized search |
| `tests/test_dejargonizer.py` | **Create** — unit tests for synonym table |
| `tests/test_solution_index.py` | **Create** — search quality tests |

## Rollout

### Phase 1: Heuristic dejargonizer + solution index (no LLM)
- Build jargon synonym table (~100 entries)
- Build SolutionTemplateIndex from 125 CDGs
- Re-run validation → measure technique coverage improvement
- Target: 40-50% technique coverage (up from 24%)

### Phase 2: LLM reranker
- Add template reranking prompt
- Wire into SingleAgentPlanner
- Re-run validation with LLM reranking
- Target: 60-70% competitive+partial (up from 19%)

### Phase 3: Full integration
- Dejargonize inbound prompts in propose()
- Template match before decomposition
- Fall through to novel CDG composition if no template fits
- Creative divergence analysis for novel proposals
