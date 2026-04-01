# Baseline Analysis — Phase 2: Runtime Atoms, Registry & FitStack Primitive

## Goal

Create the 4 runtime diagnostic atoms, their registry, and the FitStack
`AlgorithmicPrimitive` catalog entry with tunable parameters. After this phase,
all atoms are importable and tested, and the FitStack primitive is registered.

---

## Prerequisites

- Baseline Phase 1 complete (ConceptType.BASELINE_ANALYSIS exists, skeleton registered)

---

## Changes

### 1. `sciona/expansion_atoms/runtime_baseline_analysis.py` — Create

4 pure functions following the pattern in `runtime_clustering.py`:

```python
"""Runtime atoms for Baseline Analysis expansion rules.

Provides deterministic, pure functions for baseline analysis pipeline
diagnostics:

  - Onset coverage analysis (onset density relative to signal length)
  - Padding saturation detection (fraction of output that is padding)
  - Normalization clipping monitoring (fraction of values at ceiling)
  - Component balance validation (entropy of component contributions)
"""

from __future__ import annotations

import numpy as np


def check_onset_coverage(
    fit_results: list,
    signal_length: int,
) -> tuple[float, bool]:
    """Check that fit step produces sufficient onset detections.

    Args:
        fit_results: List of fit result objects (onset markers).
        signal_length: Total length of the input signal in samples.

    Returns:
        (onset_density, has_sufficient_onsets) where has_sufficient_onsets
        is True if onset_density >= 1e-4.
    """
    if signal_length <= 0:
        return 0.0, False
    density = len(fit_results) / signal_length
    return density, density >= 1e-4


def detect_padding_saturation(
    padded: np.ndarray,
    original_length: int,
) -> tuple[float, bool]:
    """Detect excessive padding in output signal.

    Args:
        padded: 1D array of padded output signal.
        original_length: Length of the original (unpadded) signal.

    Returns:
        (padding_overlap_fraction, is_saturated) where is_saturated is
        True if padding_overlap_fraction > 0.5.
    """
    padded = np.asarray(padded, dtype=np.float64).ravel()
    if padded.size == 0 or original_length <= 0:
        return 0.0, False
    padding_len = max(0, padded.size - original_length)
    fraction = padding_len / padded.size
    return fraction, fraction > 0.5


def monitor_normalization_clipping(
    normalized: np.ndarray,
) -> tuple[float, bool]:
    """Monitor fraction of normalized values clipped at 1.0.

    Args:
        normalized: 1D array of normalized values (expected in [0, 1]).

    Returns:
        (clipped_fraction, is_clipped) where is_clipped is True if
        clipped_fraction > 0.1.
    """
    arr = np.asarray(normalized, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0, False
    clipped = float(np.mean(np.isclose(arr, 1.0)))
    return clipped, clipped > 0.1


def validate_component_balance(
    component_outputs: list[np.ndarray],
) -> tuple[float, bool]:
    """Validate that component contributions are balanced.

    Computes the entropy of energy contributions across components.
    Low entropy means one component dominates the combination.

    Args:
        component_outputs: List of 1D arrays, one per component.

    Returns:
        (component_entropy, is_balanced) where is_balanced is True if
        component_entropy >= 0.5.
    """
    if len(component_outputs) <= 1:
        return 1.0, True

    energies = []
    for output in component_outputs:
        arr = np.asarray(output, dtype=np.float64).ravel()
        energies.append(float(np.sum(arr ** 2)) if arr.size > 0 else 0.0)

    total = sum(energies)
    if total == 0.0:
        return 1.0, True

    probs = [e / total for e in energies]
    # Normalized entropy (0 to 1)
    n = len(probs)
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    max_entropy = np.log2(n) if n > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
    return float(normalized_entropy), normalized_entropy >= 0.5
```

### 2. `sciona/expansion_atoms/baseline_analysis_registry.py` — Create

Follow the pattern in `clustering_registry.py`:

```python
"""Registry for baseline analysis primitives and expansion atoms."""

from __future__ import annotations

BASELINE_ANALYSIS_DECLARATIONS = {
    "check_onset_coverage": (
        "sciona.expansion_atoms.runtime_baseline_analysis.check_onset_coverage",
        "list, int -> tuple[float, bool]",
        "Check onset detection density relative to signal length.",
    ),
    "detect_padding_saturation": (
        "sciona.expansion_atoms.runtime_baseline_analysis.detect_padding_saturation",
        "ndarray, int -> tuple[float, bool]",
        "Detect excessive padding fraction in output signal.",
    ),
    "monitor_normalization_clipping": (
        "sciona.expansion_atoms.runtime_baseline_analysis.monitor_normalization_clipping",
        "ndarray -> tuple[float, bool]",
        "Monitor fraction of normalized values clipped at ceiling.",
    ),
    "validate_component_balance": (
        "sciona.expansion_atoms.runtime_baseline_analysis.validate_component_balance",
        "list[ndarray] -> tuple[float, bool]",
        "Validate energy balance across component contributions.",
    ),
}
```

### 3. FitStack primitive catalog entry

Find where `AlgorithmicPrimitive` entries are registered in the catalog
(likely `sciona/hunter/` or a primitives catalog file). Add a
`baseline_fit_stack` primitive with tunable parameters:

```python
AlgorithmicPrimitive(
    name="baseline_fit_stack",
    source="happyml-baseline",
    category=ConceptType.BASELINE_ANALYSIS,
    description=(
        "FitStack state machine for temporal event detection. "
        "Implements ONSET→CENTER→OFFSET automaton that persists "
        "across sliding windows."
    ),
    inputs=[
        IOSpec(name="window_results", type_desc="list[any]"),
    ],
    outputs=[
        IOSpec(name="fit_result", type_desc="HPYBaselineFitResult"),
    ],
    type_signature="list[any] -> HPYBaselineFitResult",
    tunable_params=[
        PrimitiveParamSpec(
            name="onset_threshold",
            kind="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            semantic_role="Detection sensitivity",
            range_source="happyml-baseline",
        ),
        PrimitiveParamSpec(
            name="center_hold_samples",
            kind="int",
            default=10,
            min_value=1,
            max_value=1000,
            semantic_role="Minimum event duration",
            range_source="happyml-baseline",
        ),
        PrimitiveParamSpec(
            name="offset_decay_rate",
            kind="float",
            default=0.1,
            min_value=0.001,
            max_value=1.0,
            log_scale=True,
            semantic_role="Offset exponential decay rate",
            range_source="happyml-baseline",
        ),
        PrimitiveParamSpec(
            name="min_event_gap",
            kind="int",
            default=5,
            min_value=1,
            max_value=500,
            semantic_role="Minimum gap between detected events",
            range_source="happyml-baseline",
        ),
    ],
    param_status=ParamStatus.APPROVED,
)
```

If no existing catalog mechanism is found, create the entry in
`sciona/expansion_atoms/baseline_analysis_registry.py` as an additional
export alongside the declarations dict.

### 4. Tests — atom + registry tests

Add to `tests/test_expansion_baseline_analysis.py` (or create a
focused test file `tests/test_baseline_atoms.py`):

**Runtime atom tests** (each function tested with passing and failing inputs):

- `check_onset_coverage`: empty list → low density; many onsets → sufficient
- `detect_padding_saturation`: short padding → not saturated; long padding → saturated
- `monitor_normalization_clipping`: varied values → not clipped; mostly 1.0 → clipped
- `validate_component_balance`: equal energy → balanced; one dominant → unbalanced
- Edge cases: empty arrays, zero-length signals, single component

**Registry tests:**
- All 4 declarations present in `BASELINE_ANALYSIS_DECLARATIONS`
- Each FQDN is importable

---

## Verification

```bash
python -m pytest tests/test_expansion_baseline_analysis.py -v -k "atom or registry"
# Or if using a separate file:
python -m pytest tests/test_baseline_atoms.py -v

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
| `sciona/expansion_atoms/runtime_baseline_analysis.py` | **Create** — 4 runtime atom functions |
| `sciona/expansion_atoms/baseline_analysis_registry.py` | **Create** — atom declarations + FitStack primitive |
| `tests/test_expansion_baseline_analysis.py` | **Create** — atom + registry tests (or `tests/test_baseline_atoms.py`) |

**Total: 3 new files, 0 modified files**

---

## Reference

- Runtime atom pattern: `sciona/expansion_atoms/runtime_clustering.py`
- Registry pattern: `sciona/expansion_atoms/clustering_registry.py`
- PrimitiveParamSpec: `sciona/architect/models.py` lines 154–184
- AlgorithmicPrimitive: `sciona/architect/models.py` lines 187–203
