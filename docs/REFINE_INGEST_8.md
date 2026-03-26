# REFINE_INGEST Phase 8 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 8 makes the remaining ingestion-contract artifacts canonical-semantic,
not just the wrappers.

Phases 1 through 7 improved extraction, IR, planning, emission, verification,
regression coverage, and non-Python parity. The main remaining gap is that
several contract surfaces still rely heavily on legacy macro-atom shapes or
generic templates:

- ghost witnesses are mostly concept-template driven rather than canonical-IR
  driven
- CDG and sub-graph construction are still derived from legacy atom trees
- match results and declaration metadata are still emitted with generic
  descriptions and Python-centric assumptions
- state-model/docstring/contract text still comes mostly from legacy or
  placeholder fields rather than canonical operation/state semantics

The objective is:

- keep wrappers, witnesses, CDG metadata, and match results aligned to the same
  canonical semantic source
- reduce semantic drift between generated code and generated metadata
- preserve the existing ingestion contract without inventing semantics in any
  artifact type

Key rule:

- once `canonical_ir` and planning exist, legacy `MacroAtomSpec` should no
  longer be the semantic source of truth for witnesses, CDG metadata, or match
  results

## Scope Boundaries

In scope:

- canonical witness generation where IR/planning evidence is sufficient
- canonical CDG node/edge metadata construction
- canonical declaration/match-result metadata
- state-model and contract/docstring tightening where canonical state semantics
  are known
- focused regression coverage for witness/CDG/match fidelity

Out of scope:

- redesigning the ghost simulator itself
- changing proof/verification policy beyond consuming better artifacts
- rewriting the whole declaration/matching stack outside emitter-owned metadata
- large planner or extractor changes unless a small interface hook is needed
- broad natural-language quality tuning beyond semantic fidelity

## Current Code Touchpoints

Primary implementation surface:

- `sciona/ingester/emitter.py`

Potential supporting surfaces:

- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/graph.py`

Tests that should drive phase 8:

- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_message_passing.py`
- `tests/test_bayesian_ingester.py`
- `tests/test_ingest_regression_harness.py`

Indirectly affected output surfaces:

- `generated_witnesses`
- `generated_state_models`
- `cdg`
- `sub_graphs`
- `match_results`

## Current Gaps

The current pipeline now emits more semantically faithful wrappers, but the
surrounding contract artifacts can still drift.

Observed issues:

- `generate_ghost_witnesses(...)` largely ignores canonical IR/planning and
  emits generic pass-through templates based on `MacroAtomSpec` shape and
  concept type
- witness signatures do not consistently reflect exact canonical inputs,
  outputs, query-vs-mutator semantics, or state-slot usage
- `build_cdg_export(...)` and `build_sub_graphs(...)` still derive node
  hierarchy and edge meaning from legacy macro-atom trees, not canonical
  operations/groups
- `build_match_results(...)` still emits generic declaration metadata and uses
  Python-centric assumptions even for non-Python canonical cases
- state-model docstrings and related artifact text are still mostly placeholder
  or legacy-adapter driven

This creates a new kind of drift:

- the wrapper may call the right upstream method
- but the witness, CDG node metadata, or match declaration may still describe a
  weaker or different operation

## Phase 8 Deliverables

### 1. Canonical Witness Generation

Upgrade witness generation so canonical IR and planning can drive witness
signatures and return/state structure when evidence is sufficient.

Required capabilities:

- witness parameter lists derived from canonical wrapper inputs / operation
  bindings rather than only legacy atom inputs
- stateful witnesses reflect exact canonical required-state and post-state
  semantics instead of blanket state pass-through
- query/metadata operations produce witnesses consistent with non-mutating
  behavior
- stateless/non-Python canonical operations can still use conservative generic
  witness bodies, but their signatures and return shapes should match canonical
  outputs
- when canonical evidence is insufficient, witnesses should fail closed or fall
  back conservatively without inventing hidden state or outputs

Important constraint:

- phase 8 is about semantic alignment, not about making ghost witnesses deeply
  behaviorally rich

### 2. Canonical CDG Construction

Upgrade CDG construction so nodes and edges are derived from canonical
operations/planned groups when present.

Required behavior:

- node names and descriptions should come from canonical operation/group
  semantics first
- node status and hierarchy should reflect phase-3 planning decisions rather
  than only legacy child-atom trees
- typed edges should prefer canonical data/state/metadata relationships when
  available
- legacy macro-atom tree emission remains the fallback only when canonical
  planning is absent

This should reduce cases where a CDG looks plausible but does not match the
actual canonical operation decomposition.

### 3. Canonical Match/Declaration Metadata

Upgrade `build_match_results(...)` to use canonical semantics when available.

Required improvements:

- declaration names, type signatures, and docstrings should align with the
  canonical operation/group
- non-Python canonical cases should not be mislabeled as generic Python-shaped
  declarations if source-language information is available
- match metadata should stop describing invented or legacy-only outputs/state

### 4. State-Model and Contract Text Tightening

Where canonical state semantics are already known, tighten emitted text:

- state-model docstrings should mention config vs fitted vs mutable runtime
  state more precisely
- wrapper/witness docs should reflect whether an operation is mutating,
  querying, predicting, or metadata-only
- avoid generic placeholder descriptions when canonical role and output binding
  information already exists

This is still deterministic text shaping, not LLM-authored prose.

## Required Interfaces With Prior Phases

Interface from phase 2:

- canonical IR remains the semantic source of truth for operation bindings,
  state slots, and output sources

Interface from phase 3:

- planning graph is the source of decomposition hierarchy when present

Interface from phase 4:

- wrapper emission already consumes canonical IR/planning; phase 8 should align
  witnesses/CDG/match metadata to that same source

Interface from phase 5:

- semantic mismatches in witnesses or metadata should remain fail-fast if they
  break verification, not enter generic repair

Interface from phase 6:

- regression harness can now assert stronger semantic quality on witness/CDG or
  declaration-facing artifacts

Interface from phase 7:

- non-Python canonical operations should participate in the same contract
  surfaces where supported

## Deterministic vs LLM Responsibilities

Deterministic in phase 8:

- canonical witness signature construction
- canonical CDG node/edge construction
- declaration/match metadata shaping
- docstring/state-model text tightening from known semantics

LLM responsibilities in phase 8:

- none for the core artifact alignment
- no new LLM dependency should be introduced for witness/CDG/contract text

## Data Model Changes

Phase 8 should prefer reusing existing canonical models:

- `IngestIRPlan`
- `OperationSpec`
- `OutputBindingSpec`
- `StateSlotSpec`
- `PlannedOperationGroup`

Small additive model changes are acceptable only if needed to expose:

- canonical display/doc metadata
- source-language/prover metadata for declarations
- explicit canonical node identifiers for CDG/match reuse

Avoid:

- duplicating canonical semantics into a second artifact-specific IR

## Rollout Plan

### Step 0. Lock Artifact-Fidelity Regression Cases

Before changes, confirm tests cover:

- canonical stateless Python wrapper + witness
- canonical stateful wrapper + witness
- canonical non-Python wrapper path
- CDG export shape
- match-result metadata shape

If coverage is missing, add focused tests first.

### Step 1. Add Canonical Witness Helpers

- introduce internal helpers in `emitter.py` that derive witness signatures and
  return/state shape from canonical context
- preserve existing concept-specialized witness bodies where they are still
  semantically valid, but drive their interfaces from canonical evidence first

### Step 2. Route Witness Generation Through Canonical Context

- update `generate_ghost_witnesses(...)` to consume canonical IR/planning when
  available
- keep legacy witness generation only as a fallback for plans without canonical
  context

### Step 3. Rebuild CDG Metadata Over Canonical Planning

- update `build_cdg_export(...)` and related helpers to prefer canonical
  planned-group / operation hierarchy
- keep legacy tree recursion only as fallback

### Step 4. Tighten Match/Declaration Metadata

- update `build_match_results(...)` so declaration metadata is derived from
  canonical operation/group semantics and source-language context
- ensure non-Python canonical cases do not get mislabeled metadata

### Step 5. Tighten State-Model and Artifact Text

- improve docstrings/descriptions where canonical state semantics are already
  known
- keep this deterministic and additive

### Step 6. Expand Regression Coverage

- add emitter tests for canonical witness signatures and return/state shape
- add CDG/match-result tests for canonical node metadata and hierarchy
- extend harness expectations where a curated case can assert stronger contract
  fidelity

## Concrete File Plan

Expected edits:

- `sciona/ingester/emitter.py`
- optionally `sciona/ingester/models.py`
- optionally `sciona/ingester/regression_harness.py`
- tests:
  - `tests/test_ingester_emitter.py`
  - `tests/test_ingest_stateful.py`
  - `tests/test_ingest_regression_harness.py`
  - optionally `tests/test_message_passing.py`
  - optionally `tests/test_bayesian_ingester.py`

Prefer keeping the write scope mostly inside `emitter.py` and emitter-owned
tests.

## Regression Risks

Primary risks:

- canonical witness generation breaks existing ghost simulation expectations
- CDG hierarchy diverges from existing consumers’ assumptions
- canonical metadata tightening overfits to one family and harms others
- non-Python declarations or witnesses pick the wrong source-language/prover
  metadata

Mitigations:

- keep legacy fallback when canonical context is absent
- fail closed instead of inventing witness outputs or state
- preserve message-passing and Bayesian specialized witness behavior unless
  canonical evidence clearly improves their interfaces
- run stateful, emitter, regression-harness, and adjacent specialized tests

## Test and Benchmark Plan

Direct emitter tests:

- canonical stateless witness uses exact canonical inputs/outputs
- canonical stateful witness reflects exact state threading
- canonical non-Python witness/metadata path stays conservative and correct
- CDG export prefers canonical node/group hierarchy
- match results expose canonical type signature/doc metadata

Protected regression slice:

- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_regression_harness.py`
- specialized witness families if touched:
  - `tests/test_message_passing.py`
  - `tests/test_bayesian_ingester.py`

Harness expectation:

- at least one curated case should assert stronger artifact fidelity beyond
  wrapper correctness, such as witness signature or canonical metadata presence

## Acceptance Criteria

Phase 8 is complete when all of the following are true:

- canonical IR/planning drives witness interfaces when present
- CDG hierarchy and typed edges prefer canonical semantics over legacy atom-tree
  guesses
- match-result declaration metadata aligns with canonical operations/groups
- state-model/docstring artifact text is tighter where canonical semantics are
  already known
- no artifact invents outputs, state, or semantics outside canonical evidence
- protected emitter/stateful/harness regressions remain green

## Deferred to Later Work

Not required in phase 8:

- redesigning ghost simulation logic itself
- rewriting the wider declaration/retrieval pipeline
- fully bespoke witness semantics for every concept family
- large natural-language documentation generation improvements

Phase 8 should align the remaining emitted contract artifacts to canonical
semantics, not broaden the system in unrelated directions.
