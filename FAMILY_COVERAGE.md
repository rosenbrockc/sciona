# Skeleton Family Coverage Plan

## Current State

19 skeleton topologies are defined in `sciona/architect/skeletons.py`, each with a
matching `ExpansionRuleSet` (21 rule sets total, since some skeletons share a
`ConceptType` key via `NAMED_SKELETONS`).  Every skeleton has:

- Runtime atoms in `sciona/expansion_atoms/runtime_*.py`
- Registry in `sciona/expansion_atoms/*_registry.py`
- DPO expansion rules + diagnostics in `sciona/principal/expansion_rules/*.py`
- Tests in `tests/test_expansion_*.py`

The `ConceptType` enum defines 47 values.  17 are registered as paradigm keys in
`SKELETON_TEMPLATES`, 2 more exist only in `NAMED_SKELETONS` (kalman_filter,
signal_detect_measure).  The remaining values fall into three categories:

1. **Node-level types** used *inside* existing skeletons (not standalone paradigms):
   PROBABILISTIC_ORACLE, CONJUGATE_UPDATE, DATA_EXTRACTION, MCMC_PROPOSAL,
   SMC_REWEIGHT, ORACLE_GRADIENT, LOG_PROB, PRIOR_DISTRIBUTION,
   LIKELIHOOD_EVALUATION
2. **Orchestration/utility types** with no algorithmic topology:
   STATE_INIT, DATA_ASSEMBLY, CONDITIONAL_ROUTING, VISUALIZATION,
   OBSERVABILITY, CUSTOM, EXTERNAL_TOOL
3. **Paradigm-level types that need skeletons**: the families listed below.

---

## Proposed New Families

### Phase 1 — Linear Algebra & Optimization

Foundational numerical methods that underpin many existing skeletons (Kalman filter
uses matrix decomposition, VI/ADVI uses gradient-based optimization, graph signal
processing uses eigendecomposition).  Adding these unlocks cross-domain expansion
between numerical core and the statistical/signal families that depend on them.

**Family 1: Linear Algebra** (`ALGEBRA`)

Covers matrix decomposition, linear system solving, and eigenvalue problems.
Already referenced by `strategy_classifier.py` (Cholesky, eigenvalue, linear
system) and `catalog.py` (3 algebra primitives).

Skeleton topology (3 nodes, linear):
```
Factorize → Solve/Transform → Validate
```
Variants: `lu_decomposition`, `qr_decomposition`, `cholesky`, `svd`,
`eigendecomposition`

Runtime atoms (4):
- `check_matrix_conditioning` — condition number analysis, ill-conditioning detection
- `validate_decomposition_accuracy` — ||A - LU|| / ||A|| residual check
- `detect_rank_deficiency` — numerical rank estimation vs expected rank
- `monitor_iterative_convergence` — for iterative solvers (CG, GMRES), track
  residual norm decay

Diagnostics trigger on: condition_number > 1e12, residual_error > 1e-8,
effective_rank < expected_rank, convergence_stall (residual ratio > 0.99).

New `ConceptType` values needed: none (`ALGEBRA` exists).

**Family 2: Continuous Optimization** (new `ConceptType`: `OPTIMIZATION`)

Covers gradient-based and derivative-free optimization for continuous objectives.
Distinct from GREEDY (combinatorial) and VI_ELBO (variational inference-specific).
Referenced by `deterministic_decompose.py` via `("loss", "gradient", "backprop")`.

Skeleton topology (4 nodes, linear with feedback):
```
Initialize → Compute Gradient → Update Parameters → Check Convergence
```
Variants: `gradient_descent`, `newton_method`, `lbfgs`, `conjugate_gradient`,
`nelder_mead`

Runtime atoms (4):
- `detect_vanishing_gradient` — gradient norm collapse detection
- `analyze_loss_landscape` — local curvature estimation (Hessian spectral analysis)
- `check_constraint_violation` — feasibility gap for constrained problems
- `monitor_convergence_rate` — actual vs expected convergence order

Diagnostics trigger on: gradient_norm < 1e-15 (vanishing), condition_number > 1e10
(ill-conditioned landscape), constraint_gap > tolerance, convergence_order < 0.5 *
expected.

New `ConceptType` values needed: `OPTIMIZATION` (add to enum).

**Family 3: Combinatorial Optimization** (`COMBINATORICS`)

Covers exact and approximate methods for discrete optimization.  Already in the
`ConceptType` enum and referenced by `ingest_coq100.py`.

Skeleton topology (4 nodes, tree-structured):
```
Bound → Branch → Prune → Select
```
Variants: `branch_and_bound`, `constraint_propagation`, `sat_solver`,
`integer_programming`

Runtime atoms (4):
- `analyze_branching_factor` — effective branching factor vs theoretical
- `monitor_bound_tightness` — gap between upper and lower bounds over time
- `detect_symmetry` — identify symmetry in the search space for breaking
- `check_pruning_effectiveness` — fraction of subtrees pruned vs explored

Diagnostics trigger on: branching_factor > 10 (exponential blowup),
bound_gap_ratio > 0.5 (loose bounds), symmetry_fraction > 0.3, pruning_rate < 0.1.

New `ConceptType` values needed: none (`COMBINATORICS` exists).

---

### Phase 2 — Neural & Learning Methods

Machine learning pipelines are referenced by the ingester (`NEURAL_NETWORK` in
`deterministic_decompose.py` and `chunker.py`) but have no skeleton.  These
families share a common train/infer duality and backpropagation-based diagnostics.

**Family 4: Neural Network** (`NEURAL_NETWORK`)

Skeleton topology (4 nodes, linear with optional feedback):
```
Forward Pass → Loss Computation → Backward Pass → Parameter Update
```
Variants: `feedforward`, `convolutional`, `recurrent`, `transformer_attention`

Runtime atoms (4):
- `detect_gradient_explosion` — max gradient norm exceeds threshold
- `analyze_activation_statistics` — dead neuron detection (ReLU), saturation
- `monitor_loss_convergence` — loss curve plateau/oscillation detection
- `check_weight_distribution` — weight norm drift, fan-in/fan-out scaling

**Family 5: Clustering** (new `ConceptType`: `CLUSTERING`)

Skeleton topology (3 nodes, iterative):
```
Initialize Centers → Assign Points → Update Centers
```
Variants: `k_means`, `dbscan`, `hierarchical`, `spectral_clustering`,
`gaussian_mixture`

Runtime atoms (4):
- `analyze_cluster_balance` — size distribution across clusters
- `monitor_assignment_stability` — fraction of points changing clusters
- `detect_empty_clusters` — degenerate clusters with no members
- `validate_separation` — inter-cluster vs intra-cluster distance ratio

**Family 6: Dimensionality Reduction** (new `ConceptType`: `DIMENSIONALITY_REDUCTION`)

Skeleton topology (3 nodes, linear):
```
Center/Scale → Project → Validate Reconstruction
```
Variants: `pca`, `kernel_pca`, `tsne`, `umap`, `autoencoder`

Runtime atoms (4):
- `analyze_explained_variance` — cumulative variance ratio per component
- `detect_crowding` — neighbor preservation quality (trustworthiness metric)
- `check_reconstruction_error` — ||X - X_reconstructed|| / ||X||
- `validate_orthogonality` — component orthogonality for linear methods

---

### Phase 3 — Numerical Integration & Simulation

Time-stepping and integration methods that are structurally similar (they share an
"advance state → check error → adapt step" pattern) but differ in their error
analysis.  Important for scientific computing pipelines.

**Family 7: ODE Solvers** (new `ConceptType`: `ODE_SOLVER`)

Skeleton topology (4 nodes, adaptive loop):
```
Evaluate Derivative → Advance State → Estimate Error → Adapt Step Size
```
Variants: `euler`, `runge_kutta_4`, `dormand_prince`, `bdf`, `adams_bashforth`

Runtime atoms (4):
- `monitor_step_rejection_rate` — fraction of steps rejected by error control
- `detect_stiffness` — eigenvalue ratio of Jacobian (stiff system indicator)
- `check_energy_conservation` — drift in conserved quantities (Hamiltonian systems)
- `validate_order_of_accuracy` — empirical convergence order vs theoretical

**Family 8: Quadrature / Numerical Integration** (new `ConceptType`: `QUADRATURE`)

Skeleton topology (3 nodes, adaptive):
```
Sample Points → Evaluate Integrand → Estimate Error/Refine
```
Variants: `trapezoidal`, `simpsons`, `gauss_legendre`, `monte_carlo_integration`,
`adaptive_quadrature`

Runtime atoms (4):
- `analyze_integrand_smoothness` — derivative magnitude estimation
- `detect_singularity` — integrand blowup near evaluation points
- `monitor_convergence_rate` — error vs number of evaluations
- `check_domain_coverage` — distribution of sample points (gaps/clustering)

**Family 9: Random Sampling & Sketching** (new `ConceptType`: `RANDOMIZED`)

Skeleton topology (3 nodes, linear):
```
Generate Samples → Sketch/Hash → Estimate
```
Variants: `reservoir_sampling`, `count_min_sketch`, `locality_sensitive_hashing`,
`random_projection`, `importance_sampling`

Runtime atoms (4):
- `validate_hash_independence` — collision rate vs theoretical expectation
- `analyze_sketch_accuracy` — point query error distribution
- `monitor_sample_coverage` — coverage of the population space
- `check_concentration_bound` — empirical vs theoretical error bounds

---

### Phase 4 — Information Theory & Compression

These families share an entropy/coding-theoretic foundation and are structurally
adjacent (compression uses entropy estimation; channel coding uses error detection).

**Family 10: Information Theory** (new `ConceptType`: `INFORMATION_THEORY`)

Skeleton topology (3 nodes, linear):
```
Estimate Distribution → Compute Entropy/Divergence → Validate Bounds
```
Variants: `entropy_estimation`, `kl_divergence`, `mutual_information`,
`rate_distortion`

Runtime atoms (4):
- `check_distribution_support` — zero-probability bins causing log(0)
- `analyze_sample_sufficiency` — sample size vs distribution complexity
- `detect_numerical_underflow` — log-probability underflow in computation
- `validate_information_inequality` — data processing inequality violations

**Family 11: Compression** (new `ConceptType`: `COMPRESSION`)

Skeleton topology (3 nodes, pipeline):
```
Model Source → Encode → Decode/Verify
```
Variants: `huffman_coding`, `arithmetic_coding`, `lempel_ziv`, `dictionary_coding`

Runtime atoms (4):
- `analyze_compression_ratio` — achieved vs theoretical (entropy) limit
- `validate_lossless_roundtrip` — decoded == original
- `detect_dictionary_bloat` — dictionary size growth rate
- `monitor_encoding_throughput` — symbols per operation

#### Phase 4 Implementation Plan

Implement sequentially: **Information Theory** → **Compression**.

##### Step 1: Add the new `ConceptType` values

Modify:
- `sciona/architect/models.py`
- `tests/test_architect_models.py`

Add:
- `INFORMATION_THEORY = "information_theory"`
- `COMPRESSION = "compression"`

Update the expected enum set in `tests/test_architect_models.py`.

##### Step 2: Information Theory family

Modify:
- `sciona/architect/skeletons.py`
- `tests/test_skeletons.py`
- `tests/test_dsp_integration.py`

Create:
- `sciona/expansion_atoms/runtime_information_theory.py`
- `sciona/expansion_atoms/information_theory_registry.py`
- `sciona/principal/expansion_rules/information_theory.py`
- `tests/test_expansion_information_theory.py`

Skeleton topology (3 nodes, linear):
```
Estimate Distribution → Compute Entropy/Divergence → Validate Bounds
```

Suggested node names:
- `Estimate Distribution`
- `Compute Entropy/Divergence`
- `Validate Bounds`

Variants:
- `entropy_estimation`
- `kl_divergence`
- `mutual_information`
- `rate_distortion`

Runtime atoms:
- `check_distribution_support(probabilities) -> (zero_mass_fraction, has_full_support)`
- `analyze_sample_sufficiency(sample_count, support_size) -> (samples_per_symbol, is_sufficient)`
- `detect_numerical_underflow(log_probabilities) -> (underflow_fraction, is_stable)`
- `validate_information_inequality(lhs_values, rhs_values) -> (max_violation, inequality_holds)`

Diagnostic thresholds:
- support failures when `zero_mass_fraction > 0.0`
- sample insufficiency when `samples_per_symbol < 5.0`
- numerical underflow when `underflow_fraction > 0.05`
- inequality violation when `max_violation > 1e-9`

Rule insertion pattern:
- before `Compute Entropy/Divergence`: support check and sample sufficiency
- after `Compute Entropy/Divergence`: numerical underflow detection
- after `Validate Bounds`: information inequality validation

Rule set:
- class name: `InformationTheoryExpansionRuleSet`
- `name = "information_theory"`
- `domain = "information_theory"`

##### Step 3: Compression family

Modify:
- `sciona/architect/skeletons.py`
- `tests/test_skeletons.py`
- `tests/test_dsp_integration.py`

Create:
- `sciona/expansion_atoms/runtime_compression.py`
- `sciona/expansion_atoms/compression_registry.py`
- `sciona/principal/expansion_rules/compression.py`
- `tests/test_expansion_compression.py`

Skeleton topology (3 nodes, pipeline):
```
Model Source → Encode → Decode/Verify
```

Suggested node names:
- `Model Source`
- `Encode`
- `Decode/Verify`

Variants:
- `huffman_coding`
- `arithmetic_coding`
- `lempel_ziv`
- `dictionary_coding`

Runtime atoms:
- `analyze_compression_ratio(original_bits, compressed_bits, entropy_bound) -> (ratio_gap, is_efficient)`
- `validate_lossless_roundtrip(original, decoded) -> (mismatch_fraction, is_lossless)`
- `detect_dictionary_bloat(dictionary_sizes) -> (growth_rate, is_bounded)`
- `monitor_encoding_throughput(symbol_counts, runtimes_ms) -> (symbols_per_ms, is_fast_enough)`

Diagnostic thresholds:
- inefficient compression when `ratio_gap > 0.2`
- lossy roundtrip when `mismatch_fraction > 0.0`
- dictionary bloat when `growth_rate > 2.0`
- slow throughput when `symbols_per_ms < 1e3`

Rule insertion pattern:
- before `Encode`: compression ratio analysis
- after `Encode`: dictionary bloat detection
- after `Decode/Verify`: lossless roundtrip validation
- after `Decode/Verify`: throughput monitoring

Rule set:
- class name: `CompressionExpansionRuleSet`
- `name = "compression"`
- `domain = "compression"`

##### Step 4: Register the new families

Modify:
- `sciona/principal/expansion_rules/__init__.py`

Add lazy imports and append:
- `InformationTheoryExpansionRuleSet()`
- `CompressionExpansionRuleSet()`

##### Step 5: Update shared regression counts

Modify:
- `tests/test_skeletons.py`
- `tests/test_dsp_integration.py`

Update:
- expected `ConceptType` skeleton keys to include
  `ConceptType.INFORMATION_THEORY` and `ConceptType.COMPRESSION`
- total skeleton count by `+2` from the pre-Phase-4 baseline

##### Step 6: Verification

Run targeted tests first:

```bash
pytest -q tests/test_architect_models.py tests/test_skeletons.py tests/test_expansion_information_theory.py tests/test_expansion_compression.py
```

Then run the broader family regression slice:

```bash
pytest -q tests/test_expansion_optimization.py tests/test_expansion_linear_algebra.py tests/test_expansion_information_theory.py tests/test_expansion_compression.py
```

##### Reference pattern files

Use these as the implementation templates:
- `sciona/principal/expansion_rules/linear_algebra.py`
- `sciona/principal/expansion_rules/optimization.py`
- `sciona/expansion_atoms/runtime_linear_algebra.py`
- `sciona/expansion_atoms/linear_algebra_registry.py`
- `tests/test_expansion_linear_algebra.py`
- `tests/test_expansion_optimization.py`

---

## Architecture & Constraints

Every phase must adhere to the following.  These are non-negotiable invariants of
the existing system that planner agents must respect.

### File Structure (per family)

Each family produces exactly 4 files plus a modification:

| File | Purpose |
|---|---|
| `sciona/expansion_atoms/runtime_{family}.py` | 4 pure runtime atom functions |
| `sciona/expansion_atoms/{family}_registry.py` | Declaration metadata (documentation only) |
| `sciona/principal/expansion_rules/{family}.py` | 4 DPO rules, 4 diagnostics, 1 RuleSet class |
| `tests/test_expansion_{family}.py` | Unit + integration tests |
| `sciona/principal/expansion_rules/__init__.py` | Add to `default_rule_sets()` |

Phases that introduce new `ConceptType` values also modify:
- `sciona/architect/models.py` — add enum value
- `sciona/architect/skeletons.py` — add `_build_*()` + register in `SKELETON_TEMPLATES`

### Runtime Atoms

- **Pure functions**: no side effects, no I/O, no global state mutation.
- **Deterministic**: use `np.random.RandomState(42)` for any sampling.
- **NumPy only**: input/output types are `np.ndarray`, `float`, `int`, `bool`, `tuple`.
- **Return signature**: always `tuple[<metric>, <flag>]` — a numeric metric and a
  boolean quality indicator.
- **Defensive**: handle empty arrays, single elements, zero denominators gracefully.

### DPO Rewrite Rules

Each rule is a `RewriteRule` with the span L ← K → R:
- **LHS (L)**: 2-node pattern — one typed node + one CUSTOM wildcard, connected by
  one edge.  The typed node matches by `ConceptType` (not `matched_primitive`).
- **Interface (K)**: same 2 nodes, no edges.
- **RHS (R)**: 3 nodes — the 2 original + 1 new interposed node.  The new node has
  `matched_primitive` set to the runtime atom function name.
- **Morphisms**: identity maps on the 2 preserved node IDs, empty edge maps.
- **Priority**: 1 (low) to 3 (high) — higher priority rules fire first when
  multiple diagnostics trigger.

Node matching semantics (in `graph_rewriter.py:_node_matches_pattern`):
1. If pattern has `matched_primitive` → exact string match on target node
2. Else if pattern `concept_type != CUSTOM` → concept type match
3. Else (CUSTOM) → matches anything

### Diagnostics

- Each diagnostic function has signature:
  `(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None`
- Reads from `context.intermediates` (a `dict[str, Any]` populated at runtime).
- Returns `None` when required keys are missing (cross-domain safety).
- Wraps values in `try/except (ValueError, TypeError)` before numeric conversion.
- `ExpansionDiagnostic.severity` is clamped to `[0.0, 1.0]`.
- `ExpansionDiagnostic.rule_name` must exactly match a `RewriteRule.name` in the
  same rule set.
- The `ExpansionEngine` only applies rules whose diagnostic severity ≥ 0.3
  (`activation_threshold`).

### ExpansionRuleSet Class

Must satisfy the `ExpansionRuleSet` protocol:
- Class attributes: `name: str`, `domain: str`
- `__init__` builds `self._rules` (list of 4 `RewriteRule` instances)
- `diagnose(cdg, context) -> list[ExpansionDiagnostic]` iterates diagnostic
  functions, collects non-None results
- `rules() -> list[RewriteRule]` returns a copy of `self._rules`

Registered in `default_rule_sets()` via lazy import inside the function body
(avoids circular imports).

### Skeleton Registration

New skeletons added to `SKELETON_TEMPLATES` (keyed by `ConceptType`) or
`NAMED_SKELETONS` (keyed by variant string).  Use `NAMED_SKELETONS` when a
`ConceptType` already has a skeleton (e.g., if a new family reuses an existing
type).

Skeleton nodes use `_node()` and `_edge()` helpers defined at module top.
Node IDs are prefixed with `tpl_` and use lowercase_with_underscores.
`NodeStatus.PENDING` for template nodes (not `ATOMIC`).

### Tests

Each test file must include:
1. **Runtime atom unit tests** — 3 cases per atom (normal, edge case, empty/zero)
2. **DPO rule application tests** — verify `GraphRewriter().apply_rule()` succeeds
   and the new node's `matched_primitive` appears in the result
3. **Diagnostic trigger/no-trigger tests** — verify threshold boundary behavior
4. **Cross-domain safety** — `diagnose()` returns `[]` with empty `ExpansionContext`
5. **Integration test** — `ExpansionEngine([RuleSet()]).expand()` returns
   `result.expanded == True` when intermediates exceed thresholds

Test CDGs are built with helper functions `_node()`, `_edge()`, `_cdg()` local to
each test file, matching the skeleton topology.

### Regression

After each phase, the full test suite must pass:
```bash
python -m pytest tests/ -x --tb=short
```
The suite must stay under 60 seconds total.

---

## Phase Summary

| Phase | Families | New ConceptTypes | New Skeletons | Est. Files |
|---|---|---|---|---|
| 1 | Linear Algebra, Continuous Optimization, Combinatorial Optimization | 1 (`OPTIMIZATION`) | 3 | 15 + 2 shared |
| 2 | Neural Network, Clustering, Dimensionality Reduction | 2 (`CLUSTERING`, `DIMENSIONALITY_REDUCTION`) | 3 | 15 + 2 shared |
| 3 | ODE Solvers, Quadrature, Random Sampling & Sketching | 3 (`ODE_SOLVER`, `QUADRATURE`, `RANDOMIZED`) | 3 | 15 + 2 shared |
| 4 | Information Theory, Compression | 2 (`INFORMATION_THEORY`, `COMPRESSION`) | 2 | 10 + 2 shared |

Each phase is self-contained: it adds skeletons, expansion rules, and tests
without depending on subsequent phases.  Phase 1 should be done first because
linear algebra and optimization are foundational to the statistical families
already in the project.  Phases 2–4 can be reordered without consequence.
