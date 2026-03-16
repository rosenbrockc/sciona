# Uncertainty Modeling For Atom Profiling

## Current State

The precision-gradient path in
[ageom/synthesizer/ghost_sim.py](ageom/synthesizer/ghost_sim.py)
uses heuristic scalar "atom error factors" to estimate how much each atom
expands input uncertainty when propagating error intervals through a CDG.

Current behavior:
- `_ATOM_ERROR_FACTORS`: a hardcoded dict of 18 DSP atoms → scalar multipliers
- `_infer_atom_error_factor()`: looks up the dict, falls back to keyword
  matching (`filter` → 1.2, `detect` → 1.35, etc.), then defaults to `1.0`
- `_compute_precision_gradients()`: walks the CDG in topological order,
  multiplies each node's input `ErrorInterval` by the factor, stores
  output width − input width as the precision gradient
- Two consumers read the result:
  - `CreditAssigner._gradient_precision()` in
    [ageom/principal/backprop.py](ageom/principal/backprop.py)
    normalizes gradients to percentages for optimization ranking
  - `OptunaManager.check_early_prune()` in
    [ageom/principal/hpo.py](ageom/principal/hpo.py)
    rejects trials with `inf`/`nan` gradients

### Coverage gap

There are 377+ atoms across 130+ domains. Only 18 have explicit factors.
The keyword fallback covers maybe 30–40 more. The remaining ~310 atoms
silently receive `1.0`, which claims zero error expansion — almost certainly
wrong for matrix inversions, iterative solvers, Monte Carlo atoms, etc.

The current heuristic cannot distinguish "measured to be 1.0" from "never
estimated." This must be fixed before the factor table is useful at scale.

---

## Recommended Data Model

Replace the single-scalar heuristic with a structured uncertainty estimate
per atom:

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class AtomUncertaintyEstimate:
    """Per-atom uncertainty propagation estimate."""

    mode: Literal["analytic", "empirical", "heuristic", "unknown"]
    """How this estimate was obtained."""

    scalar_factor: float | None = None
    """Multiplicative error expansion factor (output_spread / input_spread).
    None means no estimate is available — distinct from 1.0 which means
    'measured and found to be error-neutral.'"""

    confidence: float = 0.0
    """0.0–1.0 indicating trust in this estimate.
    heuristic → 0.1–0.3, empirical → 0.5–0.9, analytic → 0.9–1.0."""

    n_trials: int = 0
    """Number of perturbation trials (empirical mode only)."""

    epsilon: float = 0.0
    """Input perturbation scale used (empirical mode only)."""

    input_regime: str = ""
    """Description of the input distribution tested, e.g. 'gaussian_1d_n1024'."""

    notes: str = ""
    """Free-text provenance or caveats."""
```

Key design decisions:
- `scalar_factor = None` means "unknown" — the propagation loop treats this
  as `1.0` but the profiler can flag it as an uncalibrated node
- `confidence` lets the credit assigner weight calibrated nodes higher
- `input_regime` records what was tested; different regimes may yield
  different factors (a filter's error expansion depends on signal bandwidth)

---

## Implementation Phases

### Phase 1 — Extract uncertainty backend interface (ageo-matcher)

**Goal:** Pure refactor. Identical `precision_gradients` output for all
existing CDGs. No new math.

**What changes:**

1. Add `ageom/synthesizer/uncertainty.py` with:

```python
from typing import Protocol

class UncertaintyBackend(Protocol):
    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        """Return the uncertainty estimate for an atom by short name."""
        ...

class HeuristicBackend:
    """Wraps the existing _ATOM_ERROR_FACTORS dict + keyword fallback."""

    def __init__(self, factors: dict[str, float] | None = None):
        self._factors = factors or _ATOM_ERROR_FACTORS

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        factor = self._lookup(atom_name)
        if factor is None:
            return AtomUncertaintyEstimate(mode="unknown")
        return AtomUncertaintyEstimate(
            mode="heuristic",
            scalar_factor=factor,
            confidence=0.2,
        )

    def _lookup(self, atom_name: str) -> float | None:
        if atom_name in self._factors:
            return self._factors[atom_name]
        # keyword fallback — same logic as current _infer_atom_error_factor
        lowered = atom_name.lower()
        for tokens, factor in _KEYWORD_FALLBACKS:
            if any(t in lowered for t in tokens):
                return factor
        return None  # unknown, not 1.0
```

2. Replace `_infer_atom_error_factor()` calls in
   `_compute_precision_gradients()` with `backend.estimate()`. The
   propagation loop itself does not change — it still multiplies
   `ErrorInterval` bounds by `scalar_factor`, defaulting to `1.0` when
   `scalar_factor is None`.

3. `run_ghost_simulation()` accepts an optional `UncertaintyBackend`
   parameter, defaulting to `HeuristicBackend()`.

**What does NOT change:**
- `_compute_precision_gradients()` algorithm (topological interval propagation)
- `GhostSimReport.precision_gradients` schema
- Consumer code in `backprop.py` and `hpo.py`
- Any code in `ageo-atoms`

**Verification:**
- Round-trip test: for every CDG in the test fixtures, assert that
  `precision_gradients` output with `HeuristicBackend` exactly matches the
  output from the current hardcoded path
- Existing `test_principal.py` tests continue to pass unchanged

**Files touched (ageo-matcher only):**
- `ageom/synthesizer/uncertainty.py` (new)
- `ageom/synthesizer/ghost_sim.py` (extract `_ATOM_ERROR_FACTORS` and
  `_infer_atom_error_factor` into `uncertainty.py`, accept backend param)
- `tests/test_ghost_sim.py` (add round-trip equivalence test)

---

### Phase 2 — Add uncertainty metadata to atoms (ageo-atoms)

**Goal:** Give each atom a place to declare its empirically measured or
analytically derived error factor, so the matcher can read it at catalog
sync time instead of relying on a hardcoded dict.

**What changes:**

1. Add an optional `error_factor` field to CDG `AlgorithmicNode` IOSpec or
   to a new sibling file `uncertainty.json` per atom directory:

```json
{
  "atom": "dm_can_brute_force",
  "estimates": [
    {
      "mode": "empirical",
      "scalar_factor": 1.12,
      "confidence": 0.75,
      "n_trials": 500,
      "epsilon": 1e-6,
      "input_regime": "gaussian_1d_n256"
    }
  ]
}
```

2. Extend `scripts/audit.py` to validate `uncertainty.json` files when
   present (schema check, `scalar_factor > 0`, `confidence` in [0,1]).

3. Extend the matcher's `source_catalog.py` to read `uncertainty.json`
   during catalog sync and populate `AtomUncertaintyEstimate` on each
   `AlgorithmicPrimitive`.

4. Add a new backend in `uncertainty.py`:

```python
class CatalogBackend:
    """Reads estimates from the synced atom catalog."""

    def __init__(self, catalog: dict[str, AtomUncertaintyEstimate]):
        self._catalog = catalog

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        return self._catalog.get(
            atom_name,
            AtomUncertaintyEstimate(mode="unknown"),
        )
```

5. Compose backends with a `ChainBackend` that tries `CatalogBackend`
   first, falls back to `HeuristicBackend`:

```python
class ChainBackend:
    def __init__(self, *backends: UncertaintyBackend):
        self._backends = backends

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        for backend in self._backends:
            est = backend.estimate(atom_name)
            if est.mode != "unknown":
                return est
        return AtomUncertaintyEstimate(mode="unknown")
```

**Files touched:**
- `ageo-atoms`: `scripts/audit.py` (schema validation), atom directories
  get `uncertainty.json` (initially empty or absent — opt-in)
- `ageo-matcher`: `ageom/synthesizer/uncertainty.py` (add `CatalogBackend`,
  `ChainBackend`), `ageom/architect/source_catalog.py` (read
  `uncertainty.json`)

---

### Phase 3 — Empirical perturbation harness (ageo-atoms)

**Goal:** Build a test harness that measures actual error expansion factors
for atoms with `np.ndarray → np.ndarray` signatures by running them on
perturbed inputs. This produces real numbers for `uncertainty.json`.

**What changes:**

1. Add `scripts/measure_uncertainty.py`:

```python
"""Measure empirical error expansion factors for atoms.

For each atom f with signature (np.ndarray) -> np.ndarray:
  1. Generate a representative input x_0 (seeded, deterministic)
  2. Compute y_0 = f(x_0)
  3. For each epsilon in [1e-8, 1e-6, 1e-4, 1e-2]:
     a. Generate N perturbed inputs: x_i = x_0 + N(0, epsilon * I)
     b. Compute y_i = f(x_i) for each
     c. Measure output spread: std(y_i) / std(x_i - x_0)
     d. Record as the scalar_factor at that epsilon
  4. Write results to the atom's uncertainty.json
"""
```

Design decisions:

- **Input generation:** Use a seeded `np.random.Generator` per atom. The
  base input `x_0` should be "typical" for the domain — not pathological.
  For atoms with shape constraints (2D, square, etc.), generate inputs that
  satisfy preconditions. Reuse witness metadata to infer expected shapes.

- **Perturbation model:** Additive Gaussian noise scaled by epsilon.
  `x_i = x_0 + eps * rng.standard_normal(x_0.shape)`. This is the simplest
  model and sufficient for a first pass.

- **Scalar summary:** Use the ratio of output-to-input standard deviations
  averaged across array elements:

  ```
  factor = mean(std(Y, axis=0)) / mean(std(X_perturbed, axis=0))
  ```

  where `Y` is the (N, *output_shape) matrix of outputs and `X_perturbed`
  is the (N, *input_shape) matrix of perturbed inputs.

- **Multi-epsilon sweep:** Testing at multiple epsilon values reveals
  whether the atom is locally linear (factor is stable across epsilon) or
  nonlinear/discontinuous (factor varies wildly). Store results at each
  epsilon; the primary estimate uses `epsilon=1e-6` (small enough for
  linearization to hold for smooth atoms, large enough to measure for
  noisy ones).

- **Failure handling:** Some atoms will crash on perturbed inputs that
  violate preconditions (e.g., non-positive-definite matrices). The harness
  catches `icontract.ViolationError` and filters those trials. If >50% of
  trials fail, flag the atom as "perturbation-hostile" and skip it.

- **N trials:** Default 500. Configurable per atom if variance is high.

2. Add `scripts/generate_base_inputs.py`:

A registry of representative inputs per atom, keyed by atom name:

```python
BASE_INPUTS: dict[str, Callable[[], np.ndarray]] = {
    "dm_can_brute_force": lambda: rng.standard_normal(256),
    "spline_bandpass_correction": lambda: rng.standard_normal(128) + 10.0,
    "axial_attention": lambda: rng.standard_normal((8, 8)),
    "dijkstra_path_planning": lambda: np.abs(rng.standard_normal((5, 5))),
    # ... one per atom
}
```

For atoms not in this registry, attempt to auto-generate an input by
reading the atom's preconditions and witness signature:
- `ndim >= 2` → generate 2D
- `isinstance(data, np.ndarray)` + `isfinite` → standard normal
- Shape constraints from witness `AbstractArray.shape` → match shape

3. Add `tests/test_uncertainty.py`:

Smoke tests that verify the harness runs without error on a handful of
representative atoms and produces factors in a sane range (0.1–100.0).
Does NOT assert specific factor values — those are empirical measurements,
not correctness properties.

**Atom categories for Phase 3 (prioritized):**

| Category | Example atoms | Expected behavior | Count |
|----------|--------------|-------------------|-------|
| Linear transforms | `fft`, `ifft`, `dct` | factor ~1.0–2.0, stable across epsilon | ~10 |
| Filters | `butter`, `sosfilt`, `lfilter` | factor 1.0–3.0, mostly stable | ~15 |
| Detection/segmentation | `kazemi_peak_detection`, `christov_segmenter` | factor varies with epsilon (nonlinear) | ~20 |
| Linear algebra | `kalman predict/update`, `lu_factor`, `solve` | factor depends on condition number | ~12 |
| Monte Carlo / stochastic | `monte_carlo_anti`, `rng_skip` | high variance, needs many trials | ~8 |
| Iterative solvers | `loopy_bp`, `greedy_mapping` | may be chaotic; perturbation-hostile | ~5 |
| Identity-like | `graph_time_scale_management`, `apply_offsets` | factor ~1.0 | ~10 |

Start with the first two categories (linear transforms and filters) since
they're well-understood, then expand to detection/segmentation where the
current heuristics are weakest.

**Files touched (ageo-atoms only):**
- `scripts/measure_uncertainty.py` (new)
- `scripts/generate_base_inputs.py` (new)
- `tests/test_uncertainty.py` (new)
- `ageoa/*/uncertainty.json` (generated output — committed per domain)

---

### Phase 4 — Populate factors for all ndarray atoms (ageo-atoms)

**Goal:** Run the Phase 3 harness across all ~57 atom files with
`np.ndarray → np.ndarray` signatures. Commit measured `uncertainty.json`
files. This replaces the hardcoded 18-entry dict with real data.

**What changes:**

1. Run `scripts/measure_uncertainty.py --all` to generate `uncertainty.json`
   for every eligible atom.

2. Manual review of outliers:
   - Factors > 10.0 → likely a bug in the atom or a pathological base input
   - Factors < 0.1 → atom is contractive; double-check it's not collapsing
     signal to zero
   - Factors that vary >10x across epsilon levels → atom is highly nonlinear;
     flag `mode: "empirical"` with a note about regime sensitivity

3. For atoms with `object` signatures (not `np.ndarray`), the harness
   can't run automatically. These keep `mode: "heuristic"` or `"unknown"`
   until their signatures are tightened (a separate effort).

4. Update `scripts/audit.py` to warn (not fail) when an `np.ndarray` atom
   has no `uncertainty.json`.

**Exit criteria:**
- Every atom with `np.ndarray → np.ndarray` signature has a measured
  `uncertainty.json` with `mode: "empirical"`
- `scripts/audit.py` reports 0 warnings for missing uncertainty data on
  ndarray atoms
- `../ageo-matcher/sync_catalog.sh` reads the new files and populates
  `CatalogBackend`

---

### Phase 5 — Analytic Gaussian propagation for smooth atoms (ageo-matcher)

**Goal:** For atoms where Phase 4 showed stable factors across epsilon
levels (confirming local linearity), compute the Jacobian numerically and
use `Sigma_out = J @ Sigma_in @ J.T` for structured covariance propagation.

**What changes:**

1. Add `AnalyticBackend` in `uncertainty.py`:

```python
class AnalyticBackend:
    """Numerical Jacobian via finite differences."""

    def estimate(self, atom_name: str) -> AtomUncertaintyEstimate:
        impl = get_atom_impl(atom_name)
        x0 = get_base_input(atom_name)
        J = _numerical_jacobian(impl, x0, eps=1e-7)
        # Scalar summary: spectral norm of J
        factor = float(np.linalg.norm(J, ord=2))
        return AtomUncertaintyEstimate(
            mode="analytic",
            scalar_factor=factor,
            confidence=0.95,
        )
```

2. The scalar factor from the Jacobian's spectral norm (largest singular
   value) is an upper bound on local error amplification. This is
   mathematically rigorous for small perturbations.

3. Use this only for atoms where Phase 4 confirmed epsilon-stability.
   For nonlinear atoms, the empirical estimate remains primary.

4. Optionally propagate full covariance matrices through linear sub-chains
   of the CDG, reducing to scalar only at nonlinear boundaries. This is an
   optimization for chains of filters/transforms and does not need to be
   implemented in the first pass.

**Files touched:**
- `ageo-matcher`: `ageom/synthesizer/uncertainty.py` (add `AnalyticBackend`)
- `ageo-atoms`: `uncertainty.json` files updated with `mode: "analytic"`
  entries alongside the empirical ones

---

### Phase 6 — Use richer uncertainty in profiling (ageo-matcher)

**Goal:** Feed the calibrated uncertainty data back into profiling and
credit assignment so the Principal makes better optimization decisions.

**What changes:**

1. `CreditAssigner._gradient_precision()` now weights gradients by
   `confidence`:
   ```python
   scored[nid] = abs(pg) * estimate.confidence
   ```
   Nodes with `mode: "unknown"` get low confidence and therefore less
   optimization pressure, avoiding false bottleneck identification.

2. `GhostSimReport` gains a new field:
   ```python
   uncalibrated_nodes: list[str] = field(default_factory=list)
   """Nodes where no measured uncertainty data was available."""
   ```
   The synthesizer populates this from `mode == "unknown"` estimates.

3. The Principal can surface uncalibrated nodes as a secondary
   recommendation: "Consider measuring uncertainty for these atoms to
   improve precision profiling accuracy."

4. `OptunaManager` can use confidence-weighted precision gradients for
   smarter trial pruning — low-confidence high-gradient nodes should not
   trigger aggressive pruning.

---

## Quick Reference

| Phase | Repo | Key deliverable | Depends on |
|-------|------|----------------|------------|
| 1 | ageo-matcher | `UncertaintyBackend` protocol + `HeuristicBackend` | — |
| 2 | both | `uncertainty.json` schema + `CatalogBackend` + `ChainBackend` | Phase 1 |
| 3 | ageo-atoms | `measure_uncertainty.py` harness | Phase 2 |
| 4 | ageo-atoms | Measured factors for all ndarray atoms | Phase 3 |
| 5 | ageo-matcher | `AnalyticBackend` (Jacobian) for smooth atoms | Phase 4 |
| 6 | ageo-matcher | Confidence-weighted credit assignment | Phase 1 |

Phases 5 and 6 can proceed in parallel once Phase 4 is done.

---

## Immediate Next Step

Implement Phase 1. It is a pure refactor of `ghost_sim.py` with no behavior
change and no cross-repo dependencies. Verify with round-trip tests against
existing CDG fixtures. Then proceed to Phase 2+3 together — the schema and
the harness can be designed as a unit.
