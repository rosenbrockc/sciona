# Hyperparameter Optimization Plan

This document describes how to add real per-atom hyperparameter optimization to `sciona optimize`.

## Current State

Today, Principal optimizes by:
- changing the evaluation objective
- swapping `matched_primitive` choices (curated variant families + UCB1 ledger bandit)
- re-decomposing the CDG under new constraints (time-travel coordinate descent)

It does **not** currently optimize numeric or symbolic parameters inside atoms.

Examples of parameters that are not yet tunable:
- filter cutoffs
- window sizes
- thresholds
- smoothing strengths
- iteration counts
- tolerance values

Current relevant code:
- [sciona/principal/graph.py](sciona/principal/graph.py)
- [sciona/principal/variant_mutation.py](sciona/principal/variant_mutation.py)
- [sciona/principal/atom_ledger.py](sciona/principal/atom_ledger.py)
- [sciona/principal/hpo.py](sciona/principal/hpo.py)
- [sciona/architect/models.py](sciona/architect/models.py)
- [sciona/architect/catalog.py](sciona/architect/catalog.py)
- [sciona/synthesizer/assembler.py](sciona/synthesizer/assembler.py)
- [sciona/principal/evaluator.py](sciona/principal/evaluator.py)

## Goal

Enable Principal to optimize:
1. structure
2. primitive selection
3. per-node hyperparameters

without mixing benchmark-specific constants into Principal itself.

## Core Design

Add a hyperparameter layer at the primitive metadata level.

Each primitive should be able to declare:
- which parameters are tunable
- which are fixed implementation details
- valid domains and constraints
- defaults
- whether tuning is safe under the primitive’s contract

Principal should then carry a trial-local parameter assignment alongside the CDG, synthesize/export a runnable artifact with those assignments, and evaluate the resulting loss.

## New Data Model

Extend [AlgorithmicPrimitive](/Users/conrad/personal/sciona/sciona/architect/models.py) with a hyperparameter schema.

Recommended additions:
- `tunable_params: list[PrimitiveParamSpec]`
- `param_families: list[str]`
- `param_schema_version: str`

Recommended `PrimitiveParamSpec` fields:
- `name`
- `kind`: `int | float | categorical | bool`
- `default`
- `min_value`
- `max_value`
- `log_scale`
- `choices`
- `step`
- `constraints`
- `semantic_role`
- `safe_to_optimize`

The important boundary is that this metadata belongs to the primitive, not to a benchmark.

Recommended provenance fields:
- `range_source`
- `source_reference`
- `source_confidence`
- `review_notes`

## Parameter Audit

This needs an explicit atom-by-atom audit. Do not expose parameters automatically from function signatures.

Parameter ranges should be grounded in primary sources in this order:
1. the wrapped implementation
2. the original upstream implementation in `third_party/` or the source repo
3. official API documentation / docstrings
4. the cited paper or method documentation when the parameter comes from a named algorithm
5. targeted web search for documented heuristics, defaults, warnings, and operating ranges

If upstream sources do not justify a safe range, the parameter should stay `blocked` rather than receiving a guessed search interval.

For each candidate atom:
1. Read the wrapped implementation.
2. Read the original upstream implementation when it exists.
3. Read official docs and cited papers if the parameter is algorithm-specific.
4. Use targeted web search when the code/docs do not clearly define practical heuristics or safe bounds.
5. Identify parameters that affect algorithm behavior.
6. Separate true hyperparameters from:
   - required data inputs
   - derived values
   - internal invariants
   - values that would break contracts if tuned
7. Define a bounded safe domain only if primary sources justify it.
8. Record provenance for the chosen default and range.
9. Mark whether the parameter should be:
   - exposed to Principal
   - family-only tuned
   - never tuned

Audit output should be stored in primitive metadata, not in ad hoc optimizer code.

## Initial Audit Targets

Start with atoms that already dominate RMSE or latency in real runs.

Priority 1:
- [sciona/runtime_signal_event_rate.py](/Users/conrad/personal/sciona/sciona/runtime_signal_event_rate.py)
  - `filter_signal_for_detection`
  - `detect_peaks_in_signal`
  - `compute_event_rate_smoothed`

Likely candidates:
- bandpass low/high cutoffs
- filter order
- clipping scale
- peak prominence multiplier
- refractory distance multiplier
- smoothing window size

Priority 2:
- other signal-processing families with stable continuous parameters
- atoms already represented as curated scaffold families

Priority 3:
- broader catalog primitives with clear numeric controls

Do **not** start with opaque or highly discrete atoms where parameter changes mostly destroy semantics.

## Principal Changes

Extend Principal state to carry parameter assignments.

Recommended additions to [PrincipalState](/Users/conrad/personal/sciona/sciona/principal/graph.py):
- `node_params: dict[str, dict[str, object]]`
- `best_node_params: dict[str, dict[str, object]]`

Principal update step should support three mutation types:
1. primitive swap
2. structural change
3. parameter perturbation

Parameter mutations should:
- only target the current bottleneck node or a small neighborhood
- respect primitive metadata bounds
- prefer local search around the current best point

## Search Strategy

Use a staged approach.

Phase 1:
- local parameter search only on the top-ranked bottleneck node
- small bounded perturbations around defaults

Phase 2:
- Optuna-backed search for parameter sets on the current fixed structure

Phase 3:
- mixed search over:
  - structure
  - primitive variant
  - hyperparameters

This avoids exploding the search space too early.

## Export / Runtime Changes

Exported artifacts must accept injected parameter assignments.

Recommended approach:
- generate wrapper functions that accept optional param overrides
- keep primitive defaults in the runtime layer
- let exported runners apply trial-specific overrides through a generated config object

This should not require editing raw source strings per trial.

Preferred mechanism:
1. primitive metadata defines tunables
2. synthesizer emits a stable call surface
3. runner injects a `params` payload
4. artifact executes with those params

## Evaluation Requirements

Parameter optimization only matters if the exported runner can be evaluated on the true objective.

That means:
- use reference-aware evaluation specs
- record the exact parameter assignment per trial
- save loss, telemetry, and structure together

Per trial, persist:
- objective loss
- parameter assignment
- structure summary
- primitive assignment summary
- node telemetry

## Complexity / Accuracy Tracking

When tuning parameters, always track:
- RMSE or other task loss
- total runtime
- peak memory
- node-level telemetry
- graph complexity

This should make it possible to answer:
- did accuracy improve from structure or parameters?
- which parameter increased cost?
- is the accuracy gain worth the latency/memory tradeoff?

## Safety Rules

- No benchmark-specific parameter hacks inside Principal.
- No unconstrained tuning of parameters that can violate contracts.
- No automatic exposure of every function argument.
- Prefer interpretable bounded spaces over free-form search.
- Treat categorical algorithm choices as primitive/variant selection, not numeric hyperparameters.
- No guessed search bounds when upstream code/docs/papers do not support them.
- Every approved tunable parameter should carry provenance back to upstream code, docs, or papers.

## Implementation Phases

### Phase 1: Metadata
- Add `PrimitiveParamSpec` and primitive-level tunable parameter metadata.
- Add tests for schema validation and serialization.

### Phase 2: Runtime Wiring
- Add parameter override support to exported Python runners.
- Ensure defaults preserve current behavior.

### Phase 3: Principal State
- Add parameter assignments to Principal state and trial history.
- Persist assignments in `trial_history.json`.

### Phase 4: Search
- Add a local parameter mutation operator.
- Add optional Optuna-backed sampling for tunable params.

### Phase 5: Benchmark Validation
- Run `sciona optimize` on ECG HR and compare:
  - baseline
  - primitive-only optimization
  - primitive + hyperparameter optimization

### Phase 6: Broader Family Rollout
- Extend audited tunable params to other curated scaffold families.

## Recommended First Deliverable

Implement hyperparameter support only for the signal-event-rate family first, but in a general framework.

That means:
- generic metadata model
- generic Principal parameter state
- generic runner injection path
- one audited family filled out end to end

This keeps the architecture general while limiting the first audit surface.

---

## Gap Analysis and Architecture Critique

The following issues were identified through a systematic review of the codebase against the plan above. Reviewed 2026-03-18.

### Critical Gaps

#### 1. No end-to-end parameter flow exists

The plan describes a parameter flow (metadata → Principal → synthesizer → artifact → evaluation → ledger) but every link in this chain is currently missing:

- **Primitives** (`AlgorithmicPrimitive` in `architect/models.py:133`) have no `tunable_params` field.
- **CDG nodes** (`AlgorithmicNode`) have no `parameter_assignments` field.
- **AssemblyUnit** (`synthesizer/models.py:13`) has no `parameter_overrides` field.
- **ExportBundle** (`synthesizer/models.py:84`) has no way to carry parameter assignments.
- **ExecutionSandbox.evaluate()** (`principal/evaluator.py:41`) invokes artifacts as subprocesses with fixed `[python, artifact, dataset]` args — no `--params` mechanism.
- **Trial history** (`principal/graph.py:205`) records structure and loss but no parameter values.
- **AtomLedger** (`principal/atom_ledger.py`) records `(slot, atom, gradient)` but not which parameter configuration was used.

Every one of these must be wired for even a single tunable parameter to flow from Optuna through to evaluation and back.

#### 2. Runtime functions don't accept parameter overrides

The audit targets in `runtime_signal_event_rate.py` have signatures like `filter_signal_for_detection(signal, sampling_rate)`. All tunable values are hardcoded constants:

| Function | Hardcoded value | Line |
|---|---|---|
| `filter_signal_for_detection` | Butterworth order=4, clipping=8σ, low=3.0 Hz, high=25.0 Hz | 53, 56-57, 61 |
| `detect_peaks_in_signal` | prominence=1.5×scale, distance=0.45×rate | 94-95 |
| `compute_event_rate_smoothed` | window=5 | 142 |

Before the Principal can tune anything, these functions must accept optional parameters with defaults that preserve current behavior. This is a prerequisite for Phase 1, not Phase 2 as the plan implies.

**Recommendation**: Add a Phase 0 that externalizes the runtime function parameters first.

#### 3. OptunaManager is a passive pruner, not an active optimizer

`OptunaManager` (`principal/hpo.py:25`) currently provides only:
- `check_early_prune()` — static method for ghost-sim gating
- `param_importances()` — fANOVA analysis (requires completed trials that don't yet exist)
- `report_trial()` — reports loss to Optuna's pruner

It does **not** call `trial.suggest_*()` to suggest parameters. It is **never instantiated** in the Principal graph — `hpo.py` is imported only for `TrialPrunedEarly` and the static `check_early_prune`. Phase 4 ("Add optional Optuna-backed sampling") understates the work: `OptunaManager` needs to be redesigned from a passive pruner into an active trial manager that owns the ask/tell loop.

**Recommendation**: Specify how Optuna's ask/tell interface reconciles with the Principal graph's fixed `seed → forward → evaluate → gradients → time_travel` structure. Options: (a) Optuna wraps the entire graph loop externally, (b) a new `suggest_parameters` node is inserted before `forward`, (c) `time_travel_update` becomes the suggestion point. The plan should pick one.

#### 4. Assembler parameter injection mechanism is unspecified

The Python assembler (`assembler.py:_compose_python`) generates calls like:

```python
filter_result = filter_signal_for_detection(input_signal, sampling_rate)
```

The plan says "generate wrapper functions that accept optional param overrides" but doesn't specify whether this happens at:
- **code generation time** — assembler emits kwargs inline per trial
- **runtime config** — a generated config object is read at function entry
- **subprocess invocation** — params passed as CLI arg, read by a harness

These have very different costs. Inline generation requires re-synthesizing per trial. A runtime config requires all primitives to read from it. A subprocess arg is most decoupled.

**Recommendation**: Use the subprocess arg approach (`--params params.json`). This is least invasive — it doesn't change the assembler or primitive signatures, only adds a thin harness. It also works naturally with `ExecutionSandbox.evaluate()` which already supports extra CLI args via `--eval-spec`. The harness reads the JSON and passes kwargs to each primitive call.

#### 5. Mixed search routing is underspecified

Phase 3 says "mixed search over structure + primitive variant + hyperparameters" but doesn't address:

- **When does the Principal choose parameter perturbation vs. structural mutation?** Currently `route_after_gradients` always routes to `time_travel`. A parameter-only trial would skip time-travel entirely.
- **How does Optuna's search space change when the structure changes?** A new CDG topology has different atoms with different tunables. The Optuna study needs rebuilding per topology.
- **Should parameter tuning happen on a frozen structure?** The plan's staged approach suggests yes (Phase 2 before Phase 3), but the Principal loop has no "freeze structure, tune params" mode.

**Recommendation**: Add an explicit routing decision after `compute_gradients`:
- If the bottleneck node has tunable params and the gradient is below a structural-change threshold → parameter perturbation trial (no topology change).
- If the gradient indicates structural issues → time-travel re-decomposition.
- Scope each Optuna study to a `(topo_hash, primitive_signature)` pair — when the structure changes, start a new study.

#### 6. Trial history doesn't support attribution

The plan asks "did accuracy improve from structure or parameters?" but the current trial history format (`{trial, loss, thread_id, structure}`) has no way to distinguish mutation types. To support attribution, each entry needs:

- `mutation_type: "structural" | "primitive_swap" | "parameter_perturbation"`
- `parameter_assignments: dict[node_id, dict[param_name, value]]`
- `parameter_delta: dict[node_id, dict[param_name, {old, new}]]`

Without this, the complexity/accuracy tracking goals are unachievable.

### Missing Pieces

#### 7. No parameter-aware gradient computation

`CreditAssigner` (`principal/backprop.py`) computes gradients per-node but treats each node as a black box. When a node has tunable parameters, the gradient should distinguish:
- wrong primitive choice → swap atom
- wrong parameters → perturb params
- structural placement → re-decompose

The plan doesn't address parameter-level gradients. At minimum, the system needs a way to compare "same atom, different params" across trials to compute a numerical parameter gradient (finite difference).

**Recommendation**: Extend `AtomObservation` with `parameters: dict[str, Any]` and add `parameter_sensitivity(slot, atom, param_name)` to the AtomLedger — estimate `∂loss/∂param` from historical observations.

#### 8. No parameter constraint propagation

The plan mentions `constraints` in `PrimitiveParamSpec` but doesn't address cross-parameter constraints. For `filter_signal_for_detection`:
- `low_cutoff` must be < `high_cutoff`
- Both must be < `nyquist = sampling_rate / 2`
- `filter_order` interacts with cutoff sharpness

**Recommendation**: Support two constraint types: (1) per-parameter bounds (already in the plan), (2) inter-parameter constraints as a validation callable on the primitive. Optuna trials that violate constraints get pruned immediately.

#### 9. No rollback for parameter regressions

The plan's safety rules say "no unconstrained tuning" but don't address what happens when a parameter change increases loss. The current Principal loop always moves forward — there's no mechanism to revert a parameter change. With structural changes, time-travel provides rollback via Architect checkpoints. Parameters need an equivalent.

**Recommendation**: Track `best_node_params` (already in the plan) and revert to it when a parameter trial increases loss beyond a threshold. This is cheaper than time-travel since it doesn't require re-decomposition.

#### 10. AtomLedger should track parameter configurations

The AtomLedger records `(slot, atom) → gradient_score` but the same atom with different parameters has very different performance. Without parameter tracking, `filter_signal_for_detection(order=2)` and `filter_signal_for_detection(order=8)` are indistinguishable.

**Recommendation**: Extend `AtomObservation` with `params_hash: str` (hash of the parameter dict). `rank_candidates` can then distinguish "atom is generally bad" from "atom with these specific params is bad."

### Suggested Phase Reordering

The current implementation phases have a dependency ordering issue. Revised:

| Phase | What | Why this order |
|-------|------|----------------|
| **0** | Externalize runtime function parameters as `**kwargs` with defaults | Nothing else works without an injection target |
| **1** | `PrimitiveParamSpec` metadata model + audit for signal-event-rate | Defines the search space |
| **2** | Parameter flow: `ExportBundle.parameter_assignments` + `--params` subprocess arg + harness | Enables a parameter to reach the artifact |
| **3** | Principal state: `node_params`, routing decision (param vs. structural), trial history recording | The optimizer can now suggest and record parameters |
| **4** | Optuna integration: `suggest_trial_params()` + study-per-topology scoping | Actual search begins |
| **5** | AtomLedger enrichment: parameter tracking + sensitivity analysis | Feedback loop closes |
| **6** | Benchmark validation on ECG HR | Prove it works |
