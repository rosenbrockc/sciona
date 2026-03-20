---
name: MCMC/HMC expansion rules implementation plan
description: Approved implementation plan for MCMC/HMC expansion rules — divergence detection, step size adaptation, mass matrix estimation, convergence diagnostics
type: project
---

## MCMC/HMC Expansion Rules — Implementation Plan (APPROVED)

### HMC Skeleton Topology (6 nodes in sciona/architect/skeletons.py, lines 863-981)

```
Init ──→ Half Step P1 ──→ Full Step Q ──→ Oracle Query ──→ Half Step P2 ──→ Accept
```

Node details:
- `Initialization Subgraph` (MCMC_KERNEL) — outputs: q, p, epsilon, mass_matrix
- `Half Step Momentum Start` (MCMC_KERNEL) — inputs: p, q, epsilon, log_density → p_half
- `Full Step Position` (MCMC_KERNEL) — inputs: q, p_half, epsilon, mass_matrix → q_new
- `Oracle Query` (PROBABILISTIC_ORACLE) — inputs: q_new, log_density → log_prob, grad
- `Half Step Momentum End` (MCMC_KERNEL) — inputs: p_half, grad, epsilon → p_new
- `Acceptance Criterion` (MCMC_KERNEL) — inputs: q, q_new, p, p_new, log_prob → accepted_q, accepted

### New Runtime Atoms (`sciona/runtime_mcmc.py`)

**1. `detect_divergent_transitions(energies_initial, energies_proposed, threshold=1000.0)`**
Detect when |H_proposed - H_initial| exceeds threshold (integrator failure from too-large step size).
- Returns: `tuple[np.ndarray, np.ndarray]` (energy_errors, divergence_mask)

**2. `compute_dual_averaging_step_size(accept_probs, target_accept=0.65, epsilon_0=1.0, gamma=0.05, t0=10, kappa=0.75)`**
Nesterov dual averaging for step size adaptation (Hoffman & Gelman 2014).
- Returns: `float` (adapted_epsilon)

**3. `estimate_mass_matrix(samples, diagonal_only=True)`**
Estimate mass matrix M from warmup samples. Diagonal = per-parameter variance. Dense = covariance.
- Returns: `np.ndarray` (M_estimated)

**4. `compute_convergence_diagnostics(chains)`**
Compute R-hat and ESS across chains. chains shape: (n_chains, n_samples, n_params).
- Returns: `tuple[np.ndarray, np.ndarray]` (rhat_per_param, ess_per_param)

### Registry (`sciona/mcmc_registry.py`)

```python
MCMC_DECLARATIONS = {
    "detect_divergent_transitions": (...),
    "compute_dual_averaging_step_size": (...),
    "estimate_mass_matrix": (...),
    "compute_convergence_diagnostics": (...),
}
```

### DPO Expansion Rules (4 rules in `sciona/principal/expansion_rules/mcmc.py`)

| Rule | LHS Pattern | Insertion | Priority |
|---|---|---|---|
| `insert_divergence_detection_after_accept` | `[Acceptance Criterion] → [*]` | Interpose `detect_divergent_transitions` | 3 |
| `insert_step_size_adaptation_before_leapfrog` | `[*] → [Half Step Momentum Start]` | Interpose `compute_dual_averaging_step_size` | 3 |
| `insert_mass_matrix_estimation_before_leapfrog` | `[*] → [Full Step Position]` | Interpose `estimate_mass_matrix` | 2 |
| `insert_convergence_diagnostics_after_accept` | `[Acceptance Criterion] → [*]` | Interpose `compute_convergence_diagnostics` | 1 |

NOTE: Rules targeting the same edge compose — after rule 1 fires on `Accept → sink`, rule 4 can match `detect_divergent_transitions → sink`. For rules 2 and 3, they target different edges (init→half_step_p1 vs init→full_step_q via mass_matrix).

For LHS matching:
- `Acceptance Criterion` nodes match by name="Acceptance Criterion" and concept_type=MCMC_KERNEL
- `Half Step Momentum Start` matches by name and concept_type=MCMC_KERNEL
- `Full Step Position` matches by name and concept_type=MCMC_KERNEL
- Wildcard `[*]` uses ConceptType.CUSTOM (no matched_primitive)

### Diagnostics (4 functions)

**1. `_diagnose_divergent_transitions`**
- Input: `context.intermediates["energies_initial"]`, `context.intermediates["energies_proposed"]`
- Metric: fraction of |delta_H| > 1000
- Threshold: > 0.0 (any divergence)
- Triggers: `insert_divergence_detection_after_accept`

**2. `_diagnose_acceptance_rate`**
- Input: `context.intermediates["accept_probs"]`
- Metric: mean acceptance probability
- Threshold: outside [0.55, 0.85] range
- Triggers: `insert_step_size_adaptation_before_leapfrog`

**3. `_diagnose_parameter_scale_variance`**
- Input: `context.intermediates["samples"]`
- Metric: max(marginal_std) / min(marginal_std)
- Threshold: > 10
- Triggers: `insert_mass_matrix_estimation_before_leapfrog`

**4. `_diagnose_convergence`**
- Input: `context.intermediates["chains"]` — shape (n_chains, n_samples, n_params)
- Metric: max R-hat across parameters
- Threshold: > 1.01
- Triggers: `insert_convergence_diagnostics_after_accept`

### Files to Create/Modify

| File | Action |
|---|---|
| `sciona/runtime_mcmc.py` | **Create** — 4 runtime atoms |
| `sciona/mcmc_registry.py` | **Create** — declarations |
| `sciona/principal/expansion_rules/mcmc.py` | **Create** — 4 DPO rules, 4 diagnostics, MCMCExpansionRuleSet |
| `sciona/principal/expansion_rules/__init__.py` | **Modify** — add MCMCExpansionRuleSet to default_rule_sets() |
| `tests/test_expansion_mcmc.py` | **Create** — tests |

### Implementation Notes

- Follow the exact same patterns used in `expansion_rules/signal_event_rate.py` and `expansion_rules/sequential_filter.py`
- Use `_node()` and `_edge()` helpers for rule construction
- LHS nodes use `concept_type` matching (MCMC_KERNEL, PROBABILISTIC_ORACLE) since skeleton nodes don't have matched_primitive set
- All diagnostics return None when their required data is missing from context (cross-domain safety)
- R-hat formula: split-R-hat from Vehtari et al. 2021 (rank-normalized, split chains)
- ESS formula: bulk ESS using autocorrelation with Geyer's initial monotone sequence
- Test the HMC CDG helper with nodes matching the skeleton topology
