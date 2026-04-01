# Baseline Analysis Family — Implementation Plan

## Overview

The BASELINE_ANALYSIS family models multi-scale temporal event detection
pipelines inspired by the `happyml/baseline` package.  These pipelines:

1. Acquire time-series data
2. Fan out to N independent **components** (each a signal processing chain)
3. Each component applies a **sliding window** (MAP combinator) over a
   **step pipeline** (mask → resample → scale → process → fit)
4. Component outputs are combined (product, convolution, coherence, weighted)
5. The combined result is regionized into discrete events

**Key structural challenges:**
- **FitStack state machine** (ONSET→CENTER→OFFSET) — opaque node
- **MAP-over-windows** — requires the MAP combinator (see MAP_COMBINATOR_PLAN.md)
- **Dynamic component fan-out** — N components determined at instantiation, not in the skeleton

**Dependency:** This plan requires the MAP combinator to be implemented first.

---

## Architecture

### New ConceptType

```python
BASELINE_ANALYSIS = "baseline_analysis"
```

### Skeleton topology

The skeleton represents a single-component pipeline.  Multi-component
fan-out is handled at CDG instantiation time by creating N copies of the
component subgraph, each feeding into a shared Combine node.

```
Acquire Data ──→ Component Pipeline ──→ Combine ──→ Regionize

Component Pipeline expands to:
┌───────────────────────────────────────────────────────┐
│  MAP Root (MAP_OVER, window_size=W, hop_size=H)       │
│    ├── Window Slicer                                   │
│    └── Body:                                           │
│         Mask → Resample → Scale → Process → Fit        │
│  Output Transform → Pad → Normalize                    │
└───────────────────────────────────────────────────────┘
```

### Template nodes (skeleton — single component)

```
Acquire Data          (BASELINE_ANALYSIS)
Preprocess            (BASELINE_ANALYSIS)   — mask, resample, scale
Windowed Analysis     (MAP_OVER)            — MAP combinator root
Fit                   (BASELINE_ANALYSIS)   — opaque FitStack node
Output Transform      (BASELINE_ANALYSIS)   — nonzero, clip, function
Normalize             (BASELINE_ANALYSIS)   — max, constant, quantile
Combine               (BASELINE_ANALYSIS)   — product, convolution, etc.
Regionize             (BASELINE_ANALYSIS)   — threshold → discrete regions
```

Edges: `Acquire → Preprocess → Windowed Analysis → Fit → Output Transform → Normalize → Combine → Regionize`

---

## Step 1: Add ConceptType.BASELINE_ANALYSIS to models.py

**File:** `sciona/architect/models.py`

Insert after `MAP_OVER = "map_over"`:

```python
BASELINE_ANALYSIS = "baseline_analysis"
```

**Tests to update:**
- `tests/test_architect_models.py` `test_expected_members` — add `"baseline_analysis"`

---

## Step 2: Skeleton template

**File:** `sciona/architect/skeletons.py`

Add `_build_baseline_analysis()` builder:

```python
def _build_baseline_analysis() -> SkeletonGraph:
    """Multi-scale temporal baseline analysis pipeline."""
    acquire = _node(
        "Acquire Data",
        "Load or receive input time-series data",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="source", type_desc="HPYBaselineTimeSeries")],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )
    preprocess = _node(
        "Preprocess",
        "Apply mask, resample, and scale steps to raw signal",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="prepared", type_desc="np.ndarray")],
    )
    windowed_analysis = _node(
        "Windowed Analysis",
        "Apply MAP-over-windows to run body pipeline on each window",
        ConceptType.MAP_OVER,
        inputs=[IOSpec(name="prepared", type_desc="np.ndarray")],
        outputs=[IOSpec(name="window_results", type_desc="list[any]")],
    )
    windowed_analysis = windowed_analysis.model_copy(
        update={"map_window_size": 1024, "map_hop_size": 512}
    )
    fit = _node(
        "Fit",
        "FitStack state machine: ONSET→CENTER→OFFSET detection",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="window_results", type_desc="list[any]")],
        outputs=[IOSpec(name="fit_result", type_desc="HPYBaselineFitResult")],
    )
    fit = fit.model_copy(
        update={
            "is_opaque": True,
            "matched_primitive": "baseline_fit_stack",
        }
    )
    output_transform = _node(
        "Output Transform",
        "Apply nonzero, clip-shift, or function transform to fit output",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="fit_result", type_desc="HPYBaselineFitResult")],
        outputs=[IOSpec(name="transformed", type_desc="np.ndarray")],
    )
    normalize = _node(
        "Normalize",
        "Normalize output via max, constant, or quantile method",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="transformed", type_desc="np.ndarray")],
        outputs=[IOSpec(name="normalized", type_desc="np.ndarray")],
    )
    combine = _node(
        "Combine",
        "Combine multiple component outputs (product, convolution, coherence)",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="normalized", type_desc="np.ndarray")],
        outputs=[IOSpec(name="combined", type_desc="np.ndarray")],
    )
    regionize = _node(
        "Regionize",
        "Threshold combined signal into discrete event regions",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="combined", type_desc="np.ndarray")],
        outputs=[IOSpec(name="regions", type_desc="list[tuple[int,int]]")],
    )

    edges = [
        _edge(acquire, preprocess, "signal", "signal", "np.ndarray"),
        _edge(preprocess, windowed_analysis, "prepared", "prepared", "np.ndarray"),
        _edge(windowed_analysis, fit, "window_results", "window_results", "list[any]"),
        _edge(fit, output_transform, "fit_result", "fit_result", "HPYBaselineFitResult"),
        _edge(output_transform, normalize, "transformed", "transformed", "np.ndarray"),
        _edge(normalize, combine, "normalized", "normalized", "np.ndarray"),
        _edge(combine, regionize, "combined", "combined", "np.ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.BASELINE_ANALYSIS,
        name="Baseline Analysis",
        description=(
            "Multi-scale temporal event detection: acquire signal, preprocess, "
            "apply windowed analysis via MAP combinator, fit state machine, "
            "transform, normalize, combine components, and regionize."
        ),
        template_nodes=[
            acquire, preprocess, windowed_analysis, fit,
            output_transform, normalize, combine, regionize,
        ],
        template_edges=edges,
        variants=[
            "physiological_baseline",
            "multi_component_detection",
            "temporal_event_extraction",
            "sliding_window_fit",
        ],
    )
```

Register: `ConceptType.BASELINE_ANALYSIS: _build_baseline_analysis()`

**Tests to update:**
- `tests/test_skeletons.py` `test_all_templates_present` — add `ConceptType.BASELINE_ANALYSIS`
- `tests/test_skeletons.py` `_ALLOWED_HETEROGENEOUS` — add:
  ```python
  ConceptType.BASELINE_ANALYSIS: {
      ConceptType.BASELINE_ANALYSIS,
      ConceptType.MAP_OVER,
  },
  ```
- `tests/test_dsp_integration.py` `test_total_skeleton_count` — 30 → 31

---

## Step 3: FitStack primitive registration

**File:** `sciona/expansion_atoms/runtime_baseline_analysis.py` — create

### Runtime atoms

| Function | Inputs | Returns | Purpose |
|---|---|---|---|
| `check_onset_coverage(fit_results, signal_length)` | list, int | `(onset_density: float, has_sufficient_onsets: bool)` | Onset count / signal length; threshold 1e-4 |
| `detect_padding_saturation(padded, original_length)` | ndarray, int | `(padding_overlap_fraction: float, is_saturated: bool)` | Fraction of output that is padding; threshold 0.5 |
| `monitor_normalization_clipping(normalized)` | ndarray | `(clipped_fraction: float, is_clipped: bool)` | Fraction of values at 1.0; threshold 0.1 |
| `validate_component_balance(component_outputs)` | list[ndarray] | `(component_entropy: float, is_balanced: bool)` | Entropy of component contributions; threshold 0.5 (low entropy = one component dominates) |

---

## Step 4: Registry

**File:** `sciona/expansion_atoms/baseline_analysis_registry.py` — create

Standard registry pattern: one `AtomDeclaration` per runtime function.

---

## Step 5: FitStack as AlgorithmicPrimitive

Register `baseline_fit_stack` in the catalog so it can be matched by
`matched_primitive`.  Tunable parameters mapped to `PrimitiveParamSpec`:

| Param | Kind | Default | Range | Semantic role |
|---|---|---|---|---|
| `onset_threshold` | float | 0.5 | [0.0, 1.0] | Detection sensitivity |
| `center_hold_samples` | int | 10 | [1, 1000] | Minimum event duration |
| `offset_decay_rate` | float | 0.1 | [0.001, 1.0] | Offset exponential decay |
| `min_event_gap` | int | 5 | [1, 500] | Minimum gap between events |

These map directly to `HPYBaselineFitRulesOptions` in the original codebase.

---

## Step 6: Expansion rules

**File:** `sciona/principal/expansion_rules/baseline_analysis.py` — create

### Node name constants

```python
_ACQUIRE_DATA = "Acquire Data"
_PREPROCESS = "Preprocess"
_WINDOWED_ANALYSIS = "Windowed Analysis"
_FIT = "Fit"
_OUTPUT_TRANSFORM = "Output Transform"
_NORMALIZE = "Normalize"
_COMBINE = "Combine"
_REGIONIZE = "Regionize"
```

### Rules

| Rule name | LHS pattern | Insertion | Priority |
|---|---|---|---|
| `insert_onset_coverage_check_after_fit` | `[Fit] → [sink]` | Interpose `check_onset_coverage` | 3 |
| `insert_padding_saturation_after_transform` | `[Output Transform] → [sink]` | Interpose `detect_padding_saturation` | 2 |
| `insert_normalization_clipping_after_normalize` | `[Normalize] → [sink]` | Interpose `monitor_normalization_clipping` | 2 |
| `insert_component_balance_after_combine` | `[Combine] → [sink]` | Interpose `validate_component_balance` | 1 |

### Diagnostics

| Function | Intermediate key | Threshold | Triggers rule |
|---|---|---|---|
| `_diagnose_onset_coverage` | `onset_density` | < 1e-4 | onset coverage rule |
| `_diagnose_padding_saturation` | `padding_overlap_fraction` | > 0.5 | padding saturation rule |
| `_diagnose_normalization_clipping` | `clipped_fraction` | > 0.1 | normalization clipping rule |
| `_diagnose_component_balance` | `component_entropy` | < 0.5 | component balance rule |

### Class

`BaselineAnalysisExpansionRuleSet` (name="baseline_analysis", domain="baseline_analysis")

---

## Step 7: Register in `__init__.py`

**File:** `sciona/principal/expansion_rules/__init__.py`

Add import and instance of `BaselineAnalysisExpansionRuleSet`.

---

## Step 8: Tests

**File:** `tests/test_expansion_baseline_analysis.py` — create

### Test CDG

```
Source → Acquire Data(BASELINE_ANALYSIS) → Preprocess(BASELINE_ANALYSIS) →
Windowed Analysis(MAP_OVER) → Fit(BASELINE_ANALYSIS) →
Output Transform(BASELINE_ANALYSIS) → Normalize(BASELINE_ANALYSIS) →
Combine(BASELINE_ANALYSIS) → Regionize(BASELINE_ANALYSIS) → Output
```

### Test categories

1. **Runtime atom tests** — each of the 4 atoms with passing/failing thresholds
2. **Rule application tests** — each rule fires and interposes correctly
3. **Diagnostic tests** — each diagnostic triggers the correct rule
4. **Integration tests** — full CDG with all rules applied
5. **FitStack opacity** — verify `is_opaque=True` prevents decomposition
6. **MAP_OVER integration** — verify windowed analysis node has correct
   `map_window_size`/`map_hop_size` after instantiation

---

## Step 9: Multi-component instantiation helper (optional)

**File:** `sciona/architect/skeletons.py`

Add `instantiate_baseline_multi_component(skeleton, goal, n_components)` helper
that duplicates the Preprocess→...→Normalize chain N times and wires all
into a shared Combine node.  This is a convenience function; the core
skeleton is single-component.

---

## Files Modified/Created Summary

| File | Action |
|---|---|
| `sciona/architect/models.py` | **Modify** — add `BASELINE_ANALYSIS` to ConceptType |
| `sciona/architect/skeletons.py` | **Modify** — add `_build_baseline_analysis()`, register, optional multi-component helper |
| `sciona/expansion_atoms/runtime_baseline_analysis.py` | **Create** — 4 runtime atoms |
| `sciona/expansion_atoms/baseline_analysis_registry.py` | **Create** — atom declarations |
| `sciona/principal/expansion_rules/baseline_analysis.py` | **Create** — 4 rules, 4 diagnostics, RuleSet |
| `sciona/principal/expansion_rules/__init__.py` | **Modify** — register BaselineAnalysisExpansionRuleSet |
| `tests/test_expansion_baseline_analysis.py` | **Create** — comprehensive test suite |
| `tests/test_architect_models.py` | **Modify** — add `"baseline_analysis"` to expected members |
| `tests/test_skeletons.py` | **Modify** — add BASELINE_ANALYSIS to expected templates + allowed heterogeneous |
| `tests/test_dsp_integration.py` | **Modify** — update skeleton count 30 → 31 |

**Total: 3 new files, 7 modified files**

---

## Implementation Order

1. **MAP combinator first** (see `MAP_COMBINATOR_PLAN.md`) — gating dependency
2. Then BASELINE_ANALYSIS in this order:
   - Step 1: ConceptType enum
   - Step 2: Skeleton template
   - Step 3–4: Runtime atoms + registry
   - Step 5: FitStack primitive (catalog entry)
   - Step 6–7: Expansion rules + registration
   - Step 8: Tests
   - Step 9: Multi-component helper (optional)

---

## Verification

```bash
# After MAP combinator:
python -m pytest tests/test_map_combinator.py -v

# After baseline analysis:
python -m pytest tests/test_expansion_baseline_analysis.py -v

# Full regression:
python -m pytest tests/ -x --tb=short \
    --ignore=tests/test_profile_varset.py \
    --ignore=tests/test_rapid_mode.py \
    --ignore=tests/test_receipt.py \
    --ignore=tests/test_e2e_principal_hodges.py
```

---

## Future Extensions

- **Optimizer integration**: Grid search over FitStack `PrimitiveParamSpec`
  parameters via the existing `optimize` machinery
- **Auto-component discovery**: LLM-driven decomposition that determines
  optimal N and per-component step pipeline configuration
- **Cross-component state sharing**: Coherence combiner requires aligned
  windows across components — may need a synchronization edge type
