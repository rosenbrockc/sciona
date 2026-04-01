# Baseline Analysis — Phase 3: Expansion Rules, Registration & Tests

## Goal

Create the 4 DPO expansion rules + 4 diagnostics for the baseline analysis
family, register the `BaselineAnalysisExpansionRuleSet` in the default rule
sets, and write the comprehensive test suite. After this phase, the baseline
analysis family is fully functional end-to-end.

---

## Prerequisites

- Baseline Phase 1 complete (ConceptType.BASELINE_ANALYSIS, skeleton registered)
- Baseline Phase 2 complete (runtime atoms importable, registry created)

---

## Changes

### 1. `sciona/principal/expansion_rules/baseline_analysis.py` — Create

Follow the pattern in `sciona/principal/expansion_rules/clustering.py`.

**Module docstring:**

```python
"""Expansion rules for the Baseline Analysis family.

Baseline analysis skeleton topology (8 nodes, linear pipeline):

    Acquire Data -> Preprocess -> Windowed Analysis -> Fit ->
    Output Transform -> Normalize -> Combine -> Regionize

Expansion insertion points:
  - After Fit: onset coverage check
  - After Output Transform: padding saturation detection
  - After Normalize: normalization clipping monitoring
  - After Combine: component balance validation
"""
```

**Imports** (mirror clustering.py):

```python
from __future__ import annotations

import logging

import numpy as np

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
)
```

**Constants:**

```python
_DOMAIN = "baseline_analysis"

_ACQUIRE_DATA = "Acquire Data"
_PREPROCESS = "Preprocess"
_WINDOWED_ANALYSIS = "Windowed Analysis"
_FIT = "Fit"
_OUTPUT_TRANSFORM = "Output Transform"
_NORMALIZE = "Normalize"
_COMBINE = "Combine"
_REGIONIZE = "Regionize"
```

**Helper `_node()` and `_edge()`:** Copy from clustering.py (identical pattern).

**4 Rules:**

| Rule name | LHS pattern | Interposed atom | Priority |
|---|---|---|---|
| `insert_onset_coverage_check_after_fit` | `[Fit] → [sink]` | `check_onset_coverage` | 3 |
| `insert_padding_saturation_after_transform` | `[Output Transform] → [sink]` | `detect_padding_saturation` | 2 |
| `insert_normalization_clipping_after_normalize` | `[Normalize] → [sink]` | `monitor_normalization_clipping` | 2 |
| `insert_component_balance_after_combine` | `[Combine] → [sink]` | `validate_component_balance` | 1 |

Each rule follows the DPO pattern:
- L graph: source node → sink node (matched by name)
- K graph: source node, sink node (preserved)
- R graph: source node → interposed atom → sink node
- Morphisms: L←K→R

**4 Diagnostics:**

```python
def _diagnose_onset_coverage(
    intermediates: dict, _ctx: ExpansionContext
) -> ExpansionDiagnostic | None:
    density = intermediates.get("onset_density")
    if density is not None and density < 1e-4:
        return ExpansionDiagnostic(
            rule_name="insert_onset_coverage_check_after_fit",
            reason=f"Onset density {density:.2e} below 1e-4 threshold",
            severity="warning",
        )
    return None


def _diagnose_padding_saturation(
    intermediates: dict, _ctx: ExpansionContext
) -> ExpansionDiagnostic | None:
    fraction = intermediates.get("padding_overlap_fraction")
    if fraction is not None and fraction > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_padding_saturation_after_transform",
            reason=f"Padding fraction {fraction:.2%} exceeds 50% threshold",
            severity="warning",
        )
    return None


def _diagnose_normalization_clipping(
    intermediates: dict, _ctx: ExpansionContext
) -> ExpansionDiagnostic | None:
    clipped = intermediates.get("clipped_fraction")
    if clipped is not None and clipped > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_normalization_clipping_after_normalize",
            reason=f"Clipped fraction {clipped:.2%} exceeds 10% threshold",
            severity="warning",
        )
    return None


def _diagnose_component_balance(
    intermediates: dict, _ctx: ExpansionContext
) -> ExpansionDiagnostic | None:
    entropy = intermediates.get("component_entropy")
    if entropy is not None and entropy < 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_component_balance_after_combine",
            reason=f"Component entropy {entropy:.3f} below 0.5 threshold",
            severity="warning",
        )
    return None
```

**RuleSet class:**

```python
class BaselineAnalysisExpansionRuleSet:
    """Expansion rule set for the baseline analysis family."""

    name = "baseline_analysis"
    domain = "baseline_analysis"

    def __init__(self) -> None:
        self._rules = [
            # Build all 4 rules here using _build_*() helpers
        ]
        self._diagnostics = [
            _diagnose_onset_coverage,
            _diagnose_padding_saturation,
            _diagnose_normalization_clipping,
            _diagnose_component_balance,
        ]

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

    def diagnose(
        self, intermediates: dict, ctx: ExpansionContext
    ) -> list[ExpansionDiagnostic]:
        results = []
        for diag_fn in self._diagnostics:
            result = diag_fn(intermediates, ctx)
            if result is not None:
                results.append(result)
        return results
```

### 2. `sciona/principal/expansion_rules/__init__.py` — Register

Add import and instance:

```python
from sciona.principal.expansion_rules.baseline_analysis import (
    BaselineAnalysisExpansionRuleSet,
)
```

Add to the returned list:

```python
BaselineAnalysisExpansionRuleSet(),
```

### 3. `tests/test_expansion_baseline_analysis.py` — Create (or extend from Phase 2)

Follow the pattern in `tests/test_expansion_clustering.py`.

**Test CDG structure:**

```
Source(CUSTOM) →
Acquire Data(BASELINE_ANALYSIS) →
Preprocess(BASELINE_ANALYSIS) →
Windowed Analysis(MAP_OVER) →
Fit(BASELINE_ANALYSIS) →
Output Transform(BASELINE_ANALYSIS) →
Normalize(BASELINE_ANALYSIS) →
Combine(BASELINE_ANALYSIS) →
Regionize(BASELINE_ANALYSIS) →
Output(CUSTOM)
```

**Test categories:**

1. **Runtime atom tests** (if not already covered in Phase 2):
   - Each of the 4 atoms with passing and failing threshold inputs
   - Edge cases: empty arrays, zero-length signals, single component

2. **Rule construction tests:**
   - `BaselineAnalysisExpansionRuleSet` has `name == "baseline_analysis"`
   - `rules()` returns exactly 4 rules
   - Each rule has the expected name

3. **Rule application tests** (one per rule):
   - Build the test CDG
   - Apply each rule individually
   - Verify the interposed node exists in the rewritten graph
   - Verify edge connectivity is maintained

4. **Diagnostic tests** (one per diagnostic):
   - `_diagnose_onset_coverage`: fires when `onset_density < 1e-4`
   - `_diagnose_padding_saturation`: fires when `padding_overlap_fraction > 0.5`
   - `_diagnose_normalization_clipping`: fires when `clipped_fraction > 0.1`
   - `_diagnose_component_balance`: fires when `component_entropy < 0.5`
   - Each returns `None` when threshold not breached

5. **Integration test:**
   - Build full CDG with all baseline analysis nodes
   - Run full `diagnose()` with all-bad intermediates → 4 diagnostics returned
   - Run full `diagnose()` with all-good intermediates → 0 diagnostics returned

6. **Skeleton-specific tests:**
   - Fit node has `is_opaque == True`
   - Fit node has `matched_primitive == "baseline_fit_stack"`
   - Windowed Analysis node has `map_window_size == 1024` and `map_hop_size == 512`

---

## Verification

```bash
python -m pytest tests/test_expansion_baseline_analysis.py -v

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
| `sciona/principal/expansion_rules/baseline_analysis.py` | **Create** — 4 rules, 4 diagnostics, RuleSet class |
| `sciona/principal/expansion_rules/__init__.py` | **Modify** — import + register BaselineAnalysisExpansionRuleSet |
| `tests/test_expansion_baseline_analysis.py` | **Create or extend** — comprehensive test suite |

**Total: 1–2 new files, 1 modified file**

---

## Reference

- Expansion rules pattern: `sciona/principal/expansion_rules/clustering.py`
- Test pattern: `tests/test_expansion_clustering.py`
- Graph rewriter: `sciona/architect/graph_rewriter.py` (Morphism, RewriteRule)
- ExpansionRuleSet protocol: `sciona/principal/expansion.py` (ExpansionContext, ExpansionDiagnostic)
