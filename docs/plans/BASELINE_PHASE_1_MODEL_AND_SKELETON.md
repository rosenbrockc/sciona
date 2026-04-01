# Baseline Analysis — Phase 1: Model & Skeleton

## Goal

Establish the BASELINE_ANALYSIS data model and skeleton template. After this phase,
`ConceptType.BASELINE_ANALYSIS` exists, the 8-node skeleton template is registered,
and all model/skeleton tests pass.

No runtime atoms, expansion rules, or assembler changes in this phase.

---

## Prerequisites

- MAP combinator Phase 1 complete (ConceptType.MAP_OVER exists, `map_window_size`/
  `map_hop_size` fields on AlgorithmicNode) — the baseline skeleton contains a
  MAP_OVER node.

---

## Changes

### 1. `sciona/architect/models.py`

**Add enum member.** Insert after `MAP_OVER = "map_over"`:

```python
BASELINE_ANALYSIS = "baseline_analysis"
```

### 2. `sciona/architect/skeletons.py`

**Add builder function.** The baseline skeleton has 8 nodes and 7 edges:

```
Acquire Data → Preprocess → Windowed Analysis → Fit →
Output Transform → Normalize → Combine → Regionize
```

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

**Register in `SKELETON_TEMPLATES`:**

```python
ConceptType.BASELINE_ANALYSIS: _build_baseline_analysis(),
```

### 3. Test updates

**`tests/test_architect_models.py`** — `test_expected_members`:
Add `"baseline_analysis"` to the `expected` set.

**`tests/test_skeletons.py`** — `test_all_templates_present`:
Add `ConceptType.BASELINE_ANALYSIS` to the `expected` set.

**`tests/test_skeletons.py`** — `_ALLOWED_HETEROGENEOUS`:
Add entry (the skeleton mixes BASELINE_ANALYSIS and MAP_OVER nodes):
```python
ConceptType.BASELINE_ANALYSIS: {
    ConceptType.BASELINE_ANALYSIS,
    ConceptType.MAP_OVER,
},
```

**`tests/test_dsp_integration.py`** — `test_total_skeleton_count`:
Update count from 30 → 31 (assumes MAP Phase 1 already bumped it to 30).

---

## Verification

```bash
python -m pytest tests/test_architect_models.py tests/test_skeletons.py tests/test_dsp_integration.py -v
python -m pytest tests/ -x --tb=short \
    --ignore=tests/test_profile_varset.py \
    --ignore=tests/test_rapid_mode.py \
    --ignore=tests/test_receipt.py \
    --ignore=tests/test_e2e_principal_hodges.py
```

---

## Files Summary

| File | Action |
|---|---|
| `sciona/architect/models.py` | **Modify** — add `BASELINE_ANALYSIS` to ConceptType |
| `sciona/architect/skeletons.py` | **Modify** — add `_build_baseline_analysis()`, register |
| `tests/test_architect_models.py` | **Modify** — add `"baseline_analysis"` to expected members |
| `tests/test_skeletons.py` | **Modify** — add BASELINE_ANALYSIS to expected + allowed heterogeneous |
| `tests/test_dsp_integration.py` | **Modify** — update skeleton count 30 → 31 |

**Total: 0 new files, 5 modified files**
