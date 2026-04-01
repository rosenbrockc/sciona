# Baseline Skeleton Audit — happyml Fidelity Check

## Summary

The current `_build_baseline_analysis()` skeleton has **5 structural
mismatches** against the actual happyml `HPYBaselineComponent` execution
flow. The data-flow topology is incorrect — several nodes are on the wrong
side of the MAP boundary, and one critical step (Padding) is missing entirely.

---

## Actual happyml Execution Flow

Source: `happyml/baseline/core.py` lines 815–862, 545–562, 776–799, 565–573

### Per-window loop (`HPYBaselineComponent.__call__`)

```python
for data in buffer:             # sliding window iteration
    component(data)             # calls __call__ per window
```

Inside `__call__`:

```
1. Step Pipeline (lines 827-841):
   for step in self.options.steps:
       internals = step(component, data, stack=stack)
       ldata[step.name] = internals.after    # feed forward to next step

   Step types: MASK → RESAMPLE → SCALE → FUNCTION → EXP_RISE → VARIANCE → FIT

2. Output Transform (lines 843-858):
   onsets, times = self.options.output(signal, t)
   self.times.append(times)
   self.values.append(onsets)
```

### Post-window (`HPYBaselineComponent.proba` / `finish_stack`)

```
3. Qualify Events (lines 545-562):
   for fit_step in steps:
       stack.qualify()           # ONSET→CENTER→OFFSET state machine
       proba += stack.proba      # accumulate probability
   proba = normalizer(proba)     # normalize HERE, per-component

4. Padding (lines 786-795) — only if no fit stacks:
   left_padding.apply(onsets, t, anchor)
   right_padding.apply(onsets, t, anchor)
   proba = normalizer(proba)     # normalize HERE too

5. Regionize (lines 565-573):
   regions = regioner.apply(proba, anchor, mask)
```

### Cross-component (`HPYBaselineAnalyzer.combine`)

```
6. Combine (lines 1835-1839):
   combined = component.options.combine(analyzer)
```

---

## Current Skeleton vs. Reality

```
CURRENT SKELETON:                    ACTUAL FLOW:

Acquire Data                         Acquire Data
  ↓                                    ↓
Preprocess (mask+resample+scale)     ┌─ MAP_OVER (per window): ──────────┐
  ↓                                  │   Step Pipeline                   │
Windowed Analysis (MAP_OVER)         │     (mask→resample→scale→...→fit) │
  ↓                                  │   Output Transform                │
Fit (opaque)                         │     (step result → onsets)        │
  ↓                                  └────────────────────────────────────┘
Output Transform                       ↓
  ↓                                  Qualify Events (FitStack.qualify)
Normalize                              ↓
  ↓                                  Pad (left/right → probability)
Combine                                ↓
  ↓                                  Normalize (per-component)
Regionize                              ↓
                                     Combine (across components)
                                       ↓
                                     Regionize (probability → regions)
```

---

## Issue 1: Step Pipeline is INSIDE the MAP, not before it

**Current:** `Preprocess → Windowed Analysis(MAP)`
**Actual:** `MAP { Step Pipeline per window }`

The step pipeline (mask, resample, scale, and optionally function, exp_rise,
variance, fit) runs on each window's data inside `__call__`. The skeleton
incorrectly places "Preprocess" before the MAP node, implying it runs once
on the full signal before windowing.

**Fix:** Move step pipeline nodes inside the MAP body. The MAP body should
contain: `Mask → Resample → Scale → Process → Per-Window Fit`.

---

## Issue 2: Output Transform is INSIDE the MAP, not after Fit

**Current:** `Fit → Output Transform` (both after MAP)
**Actual:** `MAP { ... → Output Transform }` (inside the per-window loop)

Lines 843–858 show output transform runs per-window inside `__call__`,
converting step results to onset times/values. It accumulates across windows.
The skeleton places it after the post-window Fit node.

**Fix:** Output Transform should be the last node in the MAP body.

---

## Issue 3: Missing Padding node

**Current:** No Padding node exists.
**Actual:** `proba` property (lines 786–795) applies left/right padding.

Padding is a distinct, configurable step with 4 strategies (constant,
exponential, linear, gaussian) and independent left/right configuration.
It converts discrete onset times into a continuous probability vector.
This is one of the most tunable parts of the pipeline.

**Fix:** Add a "Pad" node between Qualify Events and Normalize.

---

## Issue 4: Fit has two distinct phases

**Current:** Single "Fit" node (opaque) after MAP.
**Actual:** Per-window `FitStack.__call__()` + post-window `FitStack.qualify()`.

The per-window fit runs curve fitting on each window (inside the MAP body).
The post-window qualify processes accumulated fits through the
ONSET→CENTER→OFFSET state machine to produce qualified events and a
probability vector.

**Fix:** Split into:
- "Per-Window Fit" inside the MAP body (runs curve fits per window)
- "Qualify Events" after the MAP (FitStack.qualify — the opaque stateful node)

---

## Issue 5: Normalize is per-component, not post-combine

**Current:** `Normalize` sits between `Output Transform` and `Combine`.
**Actual:** Normalize runs inside `finish_stack()` (line 562) and `proba`
(line 799) — it's per-component, before combine.

This one is actually correct in position (before Combine) but the skeleton's
data flow implies it receives combined output. The fix is just to ensure the
description says "per-component normalization."

**Status:** Position is correct; description could be improved.

---

## Proposed Corrected Skeleton

```
Acquire Data (BASELINE_ANALYSIS)
  ↓
MAP_OVER Root (map_window_size, map_hop_size)
  ├── [body] Mask (BASELINE_ANALYSIS)
  ├── [body] Resample (BASELINE_ANALYSIS)
  ├── [body] Scale (BASELINE_ANALYSIS)
  ├── [body] Per-Window Fit (BASELINE_ANALYSIS)
  └── [body] Output Transform (BASELINE_ANALYSIS)
  ↓
Qualify Events (BASELINE_ANALYSIS, is_opaque=True, matched_primitive="baseline_fit_stack")
  ↓
Pad (BASELINE_ANALYSIS)
  ↓
Normalize (BASELINE_ANALYSIS)
  ↓
Combine (BASELINE_ANALYSIS)
  ↓
Regionize (BASELINE_ANALYSIS)
```

**Node count:** 11 (1 acquire + 1 MAP root + 5 MAP body + 4 post-MAP)
**Edge count:** 10
**MAP body edges:** Mask → Resample → Scale → Per-Window Fit → Output Transform (4)
**Top-level edges:** Acquire → MAP Root, MAP Root → Qualify Events, Qualify → Pad, Pad → Normalize, Normalize → Combine, Combine → Regionize (6)

### Key properties:
- MAP Root: `map_window_size > 0`, `map_hop_size > 0`, `children = [mask, resample, scale, fit, output_transform]`
- Per-Window Fit: regular BASELINE_ANALYSIS node (runs per window, feeds FitStack)
- Qualify Events: `is_opaque = True`, `matched_primitive = "baseline_fit_stack"` (the stateful post-window processing)
- Pad: new node for left/right padding configuration

### Variants (unchanged):
- `physiological_baseline`
- `multi_component_detection`
- `temporal_event_extraction`
- `sliding_window_fit`

---

## Impact on Other Files

| File | Change needed |
|---|---|
| `sciona/architect/skeletons.py` | Rewrite `_build_baseline_analysis()` |
| `sciona/architect/skeletons.py` | Update `instantiate_baseline_multi_component()` — shared/chain node names change |
| `sciona/principal/expansion_rules/baseline_analysis.py` | Update node name constants and rule LHS patterns |
| `sciona/expansion_atoms/runtime_baseline_analysis.py` | No change (atoms are correct) |
| `tests/test_expansion_baseline_analysis.py` | Update test CDG topology |
| `tests/test_baseline_multi_component.py` | Update expected node/edge counts and names |
| `tests/test_skeletons.py` | Update `_ALLOWED_HETEROGENEOUS` if node types change |
| `tests/test_dsp_integration.py` | Skeleton count unchanged (still 31) |
