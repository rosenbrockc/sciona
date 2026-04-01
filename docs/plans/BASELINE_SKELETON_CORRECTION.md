# Baseline Skeleton Correction — happyml Fidelity Fix

## Context

The audit in `docs/BASELINE_SKELETON_AUDIT.md` identified 4 structural errors
in the baseline analysis skeleton vs. how happyml actually executes. This plan
fixes them.

**Current (wrong) topology — 8 nodes, 7 edges:**
```
Acquire Data → Preprocess → Windowed Analysis(MAP) → Fit(opaque) →
Output Transform → Normalize → Combine → Regionize
```

**Corrected topology — 12 nodes, 10 edges:**
```
Acquire Data
  ↓
Windowed Analysis (MAP_OVER root, children=[Mask,Resample,Scale,Per-Window Fit,Output Transform])
  ├── [body] Mask
  ├── [body] Resample
  ├── [body] Scale
  ├── [body] Per-Window Fit
  └── [body] Output Transform
  ↓
Qualify Events (opaque, matched_primitive="baseline_fit_stack")
  ↓
Pad
  ↓
Normalize
  ↓
Combine
  ↓
Regionize
```

**Issues addressed:**
1. Step pipeline (Mask/Resample/Scale) moves INSIDE the MAP body (was before MAP)
2. Output Transform moves INSIDE the MAP body (was after MAP)
3. New "Pad" node added between Qualify Events and Normalize
4. Fit split into "Per-Window Fit" (MAP body) + "Qualify Events" (post-MAP, opaque)
5. "Preprocess" node removed (replaced by individual Mask/Resample/Scale body nodes)

---

## Step 1: Rewrite `_build_baseline_analysis()` in `skeletons.py`

**File:** `sciona/architect/skeletons.py` (lines 1948–2057)

Replace the entire function body with:

```python
def _build_baseline_analysis() -> SkeletonGraph:
    """Multi-scale temporal baseline analysis pipeline.

    Topology mirrors happyml HPYBaselineComponent execution:

    Per-window (MAP body):
        Mask → Resample → Scale → Per-Window Fit → Output Transform

    Post-window (top-level):
        Qualify Events → Pad → Normalize → Combine → Regionize
    """
    # --- Top-level nodes ---
    acquire = _node(
        "Acquire Data",
        "Load or receive input time-series data",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="source", type_desc="HPYBaselineTimeSeries")],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )

    # --- MAP body nodes (per-window step pipeline) ---
    mask = _node(
        "Mask",
        "Apply zeroing/masking to window data",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="window", type_desc="np.ndarray")],
        outputs=[IOSpec(name="masked", type_desc="np.ndarray")],
    )
    resample = _node(
        "Resample",
        "Resample/aggregate signal to anchor sample rate",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="masked", type_desc="np.ndarray")],
        outputs=[IOSpec(name="resampled", type_desc="np.ndarray")],
    )
    scale = _node(
        "Scale",
        "Normalize signal magnitude within window",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="resampled", type_desc="np.ndarray")],
        outputs=[IOSpec(name="scaled", type_desc="np.ndarray")],
    )
    per_window_fit = _node(
        "Per-Window Fit",
        "Run non-linear curve fitting on window (feeds FitStack per window)",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="scaled", type_desc="np.ndarray")],
        outputs=[IOSpec(name="fit_internals", type_desc="HPYBaselineFitStackInternals")],
    )
    output_transform = _node(
        "Output Transform",
        "Convert step results to onset times/values per window (nonzero, clip-shift, function, copy)",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="fit_internals", type_desc="HPYBaselineFitStackInternals")],
        outputs=[IOSpec(name="onsets", type_desc="np.ndarray")],
    )

    # MAP_OVER root — children are the body nodes above
    windowed_analysis = _node(
        "Windowed Analysis",
        "Sliding window iteration over input signal; body runs per window",
        ConceptType.MAP_OVER,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="accumulated_onsets", type_desc="list[np.ndarray]")],
    )
    windowed_analysis = windowed_analysis.model_copy(
        update={
            "map_window_size": 1024,
            "map_hop_size": 512,
            "children": [
                mask.node_id,
                resample.node_id,
                scale.node_id,
                per_window_fit.node_id,
                output_transform.node_id,
            ],
        }
    )

    # --- Post-MAP top-level nodes ---
    qualify_events = _node(
        "Qualify Events",
        "FitStack.qualify(): process accumulated fits through ONSET→CENTER→OFFSET state machine",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="accumulated_onsets", type_desc="list[np.ndarray]")],
        outputs=[IOSpec(name="probability", type_desc="np.ndarray")],
    )
    qualify_events = qualify_events.model_copy(
        update={
            "is_opaque": True,
            "matched_primitive": "baseline_fit_stack",
        }
    )
    pad = _node(
        "Pad",
        "Apply left/right padding around onsets to build probability vector (constant, exponential, linear, gaussian)",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="probability", type_desc="np.ndarray")],
        outputs=[IOSpec(name="padded", type_desc="np.ndarray")],
    )
    normalize = _node(
        "Normalize",
        "Normalize probability to [0, 1] per component (max, constant, or quantile)",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="padded", type_desc="np.ndarray")],
        outputs=[IOSpec(name="normalized", type_desc="np.ndarray")],
    )
    combine = _node(
        "Combine",
        "Combine multiple component outputs (product, convolution, coherence, weighted)",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="normalized", type_desc="np.ndarray")],
        outputs=[IOSpec(name="combined", type_desc="np.ndarray")],
    )
    regionize = _node(
        "Regionize",
        "Threshold combined probability into discrete event regions",
        ConceptType.BASELINE_ANALYSIS,
        inputs=[IOSpec(name="combined", type_desc="np.ndarray")],
        outputs=[IOSpec(name="regions", type_desc="list[tuple[int,int]]")],
    )

    # --- Edges ---
    # MAP body edges (internal to per-window iteration)
    body_edges = [
        _edge(mask, resample, "masked", "masked", "np.ndarray"),
        _edge(resample, scale, "resampled", "resampled", "np.ndarray"),
        _edge(scale, per_window_fit, "scaled", "scaled", "np.ndarray"),
        _edge(per_window_fit, output_transform, "fit_internals", "fit_internals", "HPYBaselineFitStackInternals"),
    ]
    # Top-level edges
    top_edges = [
        _edge(acquire, windowed_analysis, "signal", "signal", "np.ndarray"),
        _edge(windowed_analysis, qualify_events, "accumulated_onsets", "accumulated_onsets", "list[np.ndarray]"),
        _edge(qualify_events, pad, "probability", "probability", "np.ndarray"),
        _edge(pad, normalize, "padded", "padded", "np.ndarray"),
        _edge(normalize, combine, "normalized", "normalized", "np.ndarray"),
        _edge(combine, regionize, "combined", "combined", "np.ndarray"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.BASELINE_ANALYSIS,
        name="Baseline Analysis",
        description=(
            "Multi-scale temporal event detection: acquire signal, "
            "apply per-window step pipeline (mask, resample, scale, fit, transform) "
            "via MAP combinator, qualify events through ONSET→CENTER→OFFSET state machine, "
            "pad, normalize per component, combine, and regionize."
        ),
        template_nodes=[
            acquire,
            windowed_analysis,
            mask,
            resample,
            scale,
            per_window_fit,
            output_transform,
            qualify_events,
            pad,
            normalize,
            combine,
            regionize,
        ],
        template_edges=body_edges + top_edges,
        variants=[
            "physiological_baseline",
            "multi_component_detection",
            "temporal_event_extraction",
            "sliding_window_fit",
        ],
    )
```

**Key structural points:**
- 12 template_nodes (was 8)
- 10 template_edges (was 7): 4 body + 6 top-level
- MAP root's `children` field lists the 5 body node IDs
- `Qualify Events` is opaque with `matched_primitive="baseline_fit_stack"` (moved from old "Fit" node)
- New `Pad` node between Qualify Events and Normalize

---

## Step 2: Rewrite `instantiate_baseline_multi_component()` in `skeletons.py`

**File:** `sciona/architect/skeletons.py` (lines 2234–2368)

The multi-component helper duplicates the per-component pipeline N times with
shared Acquire/Combine/Regionize. The node names and chain composition change.

**Shared nodes (3):** Acquire Data, Combine, Regionize

**Per-component chain (9 nodes):** Windowed Analysis, Mask, Resample, Scale,
Per-Window Fit, Output Transform, Qualify Events, Pad, Normalize

Update:
- `required_names` list: replace old names with new ones
- `chain_names` list: `["Windowed Analysis", "Mask", "Resample", "Scale", "Per-Window Fit", "Output Transform", "Qualify Events", "Pad", "Normalize"]`
- The edge mapping logic is the same pattern — iterate template edges, map shared
  names to shared nodes, chain names to per-component copies

**Expected counts for N components:**
- Nodes: 3 + 9*N (N=1: 12, N=2: 21, N=3: 30)
- Edges: 10*N + 1 (N=1: 11... wait, single component should match instantiate_skeleton which gives 10 edges. Let me recalculate.)

Actually for the multi-component case the `n_components == 1` path delegates to
`instantiate_skeleton()` so its counts don't need to follow this formula. For N >= 2:
- Per component: 4 body edges + 4 top-level edges (Acquire→MAP, MAP→Qualify, Qualify→Pad, Pad→Normalize) + 1 edge (Normalize→Combine) = 9 edges
- Shared: 1 edge (Combine→Regionize)
- Total: 9*N + 1 (N=2: 19, N=3: 28)

Wait, let me re-count. Per component, the template edges that get replicated:
- Body: Mask→Resample, Resample→Scale, Scale→Per-Window Fit, Per-Window Fit→Output Transform = 4
- Top: Acquire→Windowed Analysis, Windowed Analysis→Qualify Events, Qualify Events→Pad, Pad→Normalize, Normalize→Combine = 5
- Skip: Combine→Regionize (shared, added once)
- Total per component: 9
- Plus 1 Combine→Regionize
- N=2: 19, N=3: 28

---

## Step 3: Update expansion rules in `baseline_analysis.py`

**File:** `sciona/principal/expansion_rules/baseline_analysis.py`

### 3a. Update node name constants

Replace:
```python
_FIT = "Fit"
_OUTPUT_TRANSFORM = "Output Transform"
_NORMALIZE = "Normalize"
_COMBINE = "Combine"
_REGIONIZE = "Regionize"

_FIT_EDGE = "fit->transform"
_TRANSFORM_EDGE = "transform->normalize"
_NORMALIZE_EDGE = "normalize->combine"
```

With:
```python
_QUALIFY_EVENTS = "Qualify Events"
_PAD = "Pad"
_NORMALIZE = "Normalize"
_COMBINE = "Combine"
_REGIONIZE = "Regionize"

_QUALIFY_EDGE = "qualify->pad"
_PAD_EDGE = "pad->normalize"
_NORMALIZE_EDGE = "normalize->combine"
```

### 3b. Update docstring

Replace the topology diagram in the module docstring:
```python
"""Expansion rules for the Baseline Analysis family.

Baseline analysis skeleton post-MAP topology:

    Qualify Events -> Pad -> Normalize -> Combine -> Regionize

(The MAP body [Mask -> Resample -> Scale -> Per-Window Fit -> Output Transform]
runs per-window and is not targeted by expansion rules.)

Expansion insertion points:
  - After Qualify Events: onset coverage check
  - After Pad: padding saturation detection
  - After Normalize: normalization clipping monitoring
  - After Combine: component balance validation
"""
```

### 3c. Update `_fit_node()` helper

Rename to `_qualify_events_node()`:
```python
def _qualify_events_node(node_id: str = "qualify") -> AlgorithmicNode:
    return _node(
        node_id,
        _QUALIFY_EVENTS,
        ConceptType.BASELINE_ANALYSIS,
        matched_primitive="baseline_fit_stack",
    )
```

### 3d. Rewrite all 4 rule builders

Each rule's LHS/RHS patterns must use the new node names. The conceptual
mapping:

| Old rule name | Old LHS | New LHS |
|---|---|---|
| `insert_onset_coverage_check_after_fit` | Fit → Output Transform | Qualify Events → Pad |
| `insert_padding_saturation_after_transform` | Fit → Output Transform → Normalize | Qualify Events → Pad → Normalize |
| `insert_normalization_clipping_after_normalize` | Fit → OT → Normalize → Combine | QE → Pad → Normalize → Combine |
| `insert_component_balance_after_combine` | Fit → OT → Norm → Combine → Regionize | QE → Pad → Norm → Combine → Regionize |

**Rule name updates:**
- `insert_onset_coverage_check_after_fit` → `insert_onset_coverage_check_after_qualify`
- `insert_padding_saturation_after_transform` → `insert_padding_saturation_after_pad`
- `insert_normalization_clipping_after_normalize` → unchanged
- `insert_component_balance_after_combine` → unchanged

**Example rewrite for rule 1:**
```python
def _build_insert_onset_coverage_check() -> RewriteRule:
    qualify = _qualify_events_node()
    sink = _baseline_node("sink", _PAD)
    lhs = CDGExport(nodes=[qualify, sink], edges=[_edge("qualify", "sink")])
    interface = CDGExport(nodes=[qualify, sink], edges=[])

    onset = _node(
        "onset",
        "Check Onset Coverage",
        ConceptType.BASELINE_ANALYSIS,
        matched_primitive="check_onset_coverage",
        inputs=[
            IOSpec(name="fit_results", type_desc="list"),
            IOSpec(name="signal_length", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="onset_density", type_desc="float"),
            IOSpec(name="has_sufficient_onsets", type_desc="bool"),
        ],
        description="Check onset detection density relative to signal length.",
        type_signature="list, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[qualify, onset, sink],
        edges=[_edge("qualify", "onset"), _edge("onset", "sink")],
    )

    return RewriteRule(
        name="insert_onset_coverage_check_after_qualify",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"qualify": "qualify", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"qualify": "qualify", "sink": "sink"}, edge_map={}),
        priority=3,
    )
```

Apply the same pattern to rules 2–4, substituting the new node names and
edge constants throughout. Each rule follows the same structural pattern
as the current implementation — only node names/IDs change.

### 3e. Update diagnostic rule_name references

The diagnostics reference rule names. Update:
- `_diagnose_onset_coverage`: change `rule_name` to `"insert_onset_coverage_check_after_qualify"`
- `_diagnose_padding_saturation`: change `rule_name` to `"insert_padding_saturation_after_pad"`
- Others unchanged

---

## Step 4: Update tests

### 4a. `tests/test_expansion_baseline_analysis.py`

**Update `_baseline_cdg()` helper** (lines 63–104) to match new topology:
```python
def _baseline_cdg() -> CDGExport:
    return _cdg(
        [
            _node("src", "Source"),
            _node("acquire", "Acquire Data", ConceptType.BASELINE_ANALYSIS),
            _node(
                "windowed",
                "Windowed Analysis",
                ConceptType.MAP_OVER,
                map_window_size=1024,
                map_hop_size=512,
            ),
            _node("mask", "Mask", ConceptType.BASELINE_ANALYSIS),
            _node("resample", "Resample", ConceptType.BASELINE_ANALYSIS),
            _node("scale", "Scale", ConceptType.BASELINE_ANALYSIS),
            _node("pwfit", "Per-Window Fit", ConceptType.BASELINE_ANALYSIS),
            _node("transform", "Output Transform", ConceptType.BASELINE_ANALYSIS),
            _node(
                "qualify",
                "Qualify Events",
                ConceptType.BASELINE_ANALYSIS,
                primitive="baseline_fit_stack",
                is_opaque=True,
            ),
            _node("pad", "Pad", ConceptType.BASELINE_ANALYSIS),
            _node("normalize", "Normalize", ConceptType.BASELINE_ANALYSIS),
            _node("combine", "Combine", ConceptType.BASELINE_ANALYSIS),
            _node("regionize", "Regionize", ConceptType.BASELINE_ANALYSIS),
            _node("out", "Output"),
        ],
        [
            _edge("src", "acquire"),
            # Body edges
            _edge("mask", "resample"),
            _edge("resample", "scale"),
            _edge("scale", "pwfit"),
            _edge("pwfit", "transform"),
            # Top-level edges
            _edge("acquire", "windowed"),
            _edge("windowed", "qualify"),
            _edge("qualify", "pad"),
            _edge("pad", "normalize"),
            _edge("normalize", "combine"),
            _edge("combine", "regionize"),
            _edge("regionize", "out"),
        ],
    )
```

**Update `test_rule_names`** — change expected rule names:
```python
assert names == {
    "insert_onset_coverage_check_after_qualify",
    "insert_padding_saturation_after_pad",
    "insert_normalization_clipping_after_normalize",
    "insert_component_balance_after_combine",
}
```

**Update rule application test method names and key lookups:**
- `test_onset_rule_applies`: use key `"insert_onset_coverage_check_after_qualify"`
- `test_padding_rule_applies`: use key `"insert_padding_saturation_after_pad"`
- Others unchanged

**Update diagnostic test rule_name assertions:**
- `test_onset_coverage_diagnostic`: expect `"insert_onset_coverage_check_after_qualify"`
- `test_padding_saturation_diagnostic`: expect `"insert_padding_saturation_after_pad"`
- Others unchanged

**Update `TestBaselineAnalysisSkeleton`:**
- `test_fit_node_has_opaque_primitive`: look for node named `"Qualify Events"` instead of `"Fit"`
- `test_instantiated_nodes_preserve_metadata`: look for `"Qualify Events"` instead of `"Fit"`
- Add test for `Pad` node existence
- Add test for MAP body children: verify `Windowed Analysis` node has `children` containing 5 node IDs
  matching Mask, Resample, Scale, Per-Window Fit, Output Transform

### 4b. `tests/test_baseline_multi_component.py`

**Update expected counts:**
- `test_two_components_have_expected_counts`: 21 nodes, 19 edges (was 13, 13)
- `test_three_components_have_expected_count`: 30 nodes, 28 edges (was 18, 19)

**Update `test_component_naming_and_shared_nodes`:**
Replace expected per-component names:
```python
for idx in range(1, 4):
    suffix = f" (Component {idx})"
    assert f"Windowed Analysis{suffix}" in names
    assert f"Mask{suffix}" in names
    assert f"Resample{suffix}" in names
    assert f"Scale{suffix}" in names
    assert f"Per-Window Fit{suffix}" in names
    assert f"Output Transform{suffix}" in names
    assert f"Qualify Events{suffix}" in names
    assert f"Pad{suffix}" in names
    assert f"Normalize{suffix}" in names
```

(Remove the old `"Preprocess"` and `"Fit"` assertions.)

**Update `test_component_metadata_is_preserved`:**
- Look for `"Qualify Events (Component 1)"` instead of `"Fit (Component 1)"`
- Assert `qualify.is_opaque is True` and `qualify.matched_primitive == "baseline_fit_stack"`

### 4c. `tests/test_skeletons.py`

**No change needed to `_ALLOWED_HETEROGENEOUS`** — the allowed set
`{BASELINE_ANALYSIS, MAP_OVER}` still applies.

**The parametrized well-formedness tests** (`test_has_nodes_and_edges`,
`test_no_dangling_edges`, `test_unique_node_ids`, `test_nodes_match_paradigm`,
`test_all_nodes_pending`, `test_has_description_and_name`) will automatically
validate the new skeleton — no manual changes needed.

### 4d. `tests/test_dsp_integration.py`

**Skeleton count unchanged** — still 31.

---

## Step 5: Update `runtime_baseline_analysis.py` atoms (NO CHANGE)

The 4 runtime atoms (`check_onset_coverage`, `detect_padding_saturation`,
`monitor_normalization_clipping`, `validate_component_balance`) remain correct.
Their logic is independent of the skeleton topology.

---

## Step 6: Update `baseline_analysis_registry.py` (NO CHANGE)

The registry declarations remain correct — they reference the runtime functions
which are unchanged.

---

## Verification

```bash
# Run baseline-specific tests
python -m pytest tests/test_expansion_baseline_analysis.py tests/test_baseline_multi_component.py -v

# Run skeleton well-formedness tests (will catch dangling edges, missing nodes, etc.)
python -m pytest tests/test_skeletons.py -v

# Run full model + integration tests
python -m pytest tests/test_architect_models.py tests/test_dsp_integration.py -v

# Full regression
python -m pytest tests/ -x --tb=short \
    --ignore=tests/test_profile_varset.py \
    --ignore=tests/test_rapid_mode.py \
    --ignore=tests/test_receipt.py \
    --ignore=tests/test_e2e_principal_hodges.py \
    --ignore=tests/test_ingest_biosppy_ecg.py \
    --ignore=tests/test_ingest_stateful.py
```

---

## Files Summary

| File | Action |
|---|---|
| `sciona/architect/skeletons.py` | **Modify** — rewrite `_build_baseline_analysis()` (12 nodes, 10 edges) and `instantiate_baseline_multi_component()` (new chain names) |
| `sciona/principal/expansion_rules/baseline_analysis.py` | **Modify** — update node constants, rule LHS/RHS patterns, rule names, diagnostic references |
| `tests/test_expansion_baseline_analysis.py` | **Modify** — update test CDG, rule names, skeleton assertions |
| `tests/test_baseline_multi_component.py` | **Modify** — update expected counts, node names |

**Total: 0 new files, 4 modified files**

No changes to: `models.py`, `runtime_baseline_analysis.py`, `baseline_analysis_registry.py`,
`test_skeletons.py` (parametrized tests auto-validate), `test_dsp_integration.py` (count unchanged),
`expansion_rules/__init__.py` (class name unchanged).
