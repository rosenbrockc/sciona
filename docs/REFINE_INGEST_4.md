# REFINE_INGEST Phase 4 Implementation Plan

## Phase Goal

Phase 4 makes wrapper and state emission deterministic over canonical ingest IR
and phase-3 planning output.

The emitter should become a mostly mechanical pretty-printer over proven facts.
It must stop guessing:

- which upstream method to call
- which arguments to pass
- which state must be injected before the call
- which state must be extracted after the call
- where an emitted output actually comes from

If the emitter still needs to infer those things, the IR/planning boundary is
still leaking semantics.

## Scope Boundaries

In scope:

- deterministic wrapper emission from canonical IR/planned groups
- exact state injection and extraction behavior
- exact output binding behavior for return values, attribute reads, tuples, and
  metadata/query objects
- backward-compatible integration into `emit_ingestion_bundle(...)`
- tests proving upstream parity on representative Python OO cases

Out of scope:

- verification/repair policy changes
- broad witness redesign beyond passing the right types/shapes through
- tree-sitter parity beyond keeping existing flows compatible
- large CDG redesign outside what emission needs

Key rule:

- wrapper source may only reference methods, attrs, state slots, and outputs
  already declared in canonical IR or phase-3 planning output

## Current Code Touchpoints

Primary implementation surface:

- `sciona/ingester/emitter.py`

Current functions that still need semantic upgrade:

- `generate_atom_wrappers(...)`
- `generate_stateful_wrappers(...)`
- `emit_ingestion_bundle(...)`
- `build_cdg_export(...)`
- witness naming/typing helpers only as needed

Supporting models:

- `sciona/ingester/models.py`
  - `IngestIRPlan`
  - `OperationSpec`
  - `MethodBinding`
  - `OutputBindingSpec`
  - `StateSlotSpec`
  - `StateEffectSpec`
  - `PlannedOperationGroup`
  - `ValidatedMacroPlan`

Phase-2/3 context providers:

- `sciona/ingester/chunker.py`
- `sciona/ingester/graph.py`

Tests that should drive phase 4:

- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_config_flatten.py`
- `tests/test_ingest_biosppy_ecg.py`
- `tests/test_ingest_dl_boundary.py`
- `tests/test_ingest_procedural.py`
- `tests/test_bayesian_ingester.py`
- `tests/test_message_passing.py`

## Current Gaps

The current emitter still relies too heavily on legacy `MacroAtomSpec` shapes.

Specific problems in current code:

- `generate_atom_wrappers(...)` still emits `NotImplementedError` stubs for
  ordinary wrappers instead of real IR-driven method calls
- `generate_stateful_wrappers(...)` injects every state field into the object,
  calls methods with all atom inputs, and then extracts all state fields back
  without distinguishing required vs irrelevant state
- output emission still assumes outputs correspond to object attributes in the
  stateful path
- tuple outputs are reconstructed generically rather than from canonical output
  bindings
- metadata/query returns are not emitted using binding kinds
- `emit_ingestion_bundle(...)` chooses stateful vs non-stateful generation based
  on legacy `state_models`, not canonical required state

Phase 4 should make the emitter consume canonical planning/IR first and treat
legacy macro-atoms as compatibility fallbacks only.

## Required Inputs From Prior Phases

Phase 4 should consume:

- phase-2 canonical IR:
  - `IngestIRPlan.operations`
  - `StateSlotSpec`
  - `OutputBindingSpec`
  - `StateEffectSpec`
- phase-3 planning output:
  - `IngestPlanGraph`
  - `PlannedOperationGroup`
  - group/member operation mapping

Minimum assumptions:

- output bindings already distinguish:
  - `return_value`
  - `attribute_read`
  - `tuple_element`
  - `self_return`
  - `metadata_object`
  - `constant`
- state slots already distinguish config/fitted/derived/transient kinds
- phase-3 has already decided keep/decompose/block

## Emission Model

### Canonical Emitter Input

The canonical emitter input should become:

- `ValidatedMacroPlan.plan.canonical_ir`
- optionally guided by `ValidatedMacroPlan.plan.planning_graph`

Recommended approach:

- add an internal “emission view” adapter that lowers canonical IR/planned
  groups into a deterministic wrapper spec
- keep that adapter private to `emitter.py`
- do not force the rest of the codebase to learn emitter internals

### Wrapper-Level Spec

Recommended internal model:

- `WrapperSpec`
  - wrapper function name
  - source method binding(s)
  - parameter list
  - required state slots
  - emitted output bindings
  - state update slots
  - return type
  - source language/class handling

Even if this stays local to `emitter.py`, the structure should be explicit.

## Deterministic Emission Rules

### Rule Family 1: Exact Method Calls

The emitter must derive the call site from `MethodBinding.signature`.

Required behavior:

- only pass direct wrapper inputs that correspond to declared method parameters
- preserve keyword-only parameters when required
- never pass every atom input blindly to every method
- for multi-method grouped operations, preserve deterministic call order from
  planning output

### Rule Family 2: Exact State Injection

State injection must be selective, not blanket.

Required behavior:

- inject only the state slots required by the planned group / operation
- config slots needed by the method must be rehydrated
- fitted state needed for predict/query methods must be rehydrated
- irrelevant derived/transient slots should not be injected just because they
  exist in the legacy state model

Safe fallback during transition:

- if the compatibility adapter can only expose a superset, allow superset
  injection temporarily, but the canonical emission path should track the exact
  required slot set and tests should assert it

### Rule Family 3: Exact State Extraction

State extraction must follow `StateEffectSpec`, not “extract every field”.

Required behavior:

- update only slots written or initialized by the emitted operation/group
- preserve untouched slots from prior state
- allow read-only operations to return unchanged state
- support no-state operations without forcing a fake state model

### Rule Family 4: Exact Output Sourcing

Emit outputs from `OutputBindingSpec.binding_kind`.

Required behavior by binding kind:

- `return_value`
  - capture the real method return value
- `attribute_read`
  - read `obj.<source_attr>` after the call
- `tuple_element`
  - index into the real returned tuple
- `self_return`
  - do not invent a data output; return state continuity only if the wrapper
    contract requires it
- `metadata_object`
  - return the real metadata/query object from the method call
- `constant`
  - emit the constant value only if represented in IR; otherwise fail closed

Explicitly forbidden:

- reading `obj.<output_name>` when the binding kind is `return_value`
- inventing `result` outputs when the operation has no emitted output binding

### Rule Family 5: Multi-Method Group Emission

If phase-3 grouped multiple operations/methods into one planned unit, phase 4
must emit a real ordered wrapper:

- instantiate object once
- inject required pre-state
- call each bound method in order
- collect outputs from the declared bound source
- extract post-state based on accumulated state effects

If the required sequencing cannot be represented faithfully from planning data,
the emitter should fail rather than improvise.

## Compatibility Strategy

Phase 4 should remain additive while improving real emission behavior.

Recommended rollout:

1. Add canonical emitter helpers alongside existing legacy functions.
2. Make `emit_ingestion_bundle(...)` prefer canonical IR/planning when present.
3. Fall back to legacy macro-atom emission only when canonical IR is missing.
4. Keep CDG and witness generation stable unless canonical info improves them.

Important rule:

- legacy `MacroAtomSpec` should not be treated as the canonical source once
  `canonical_ir` exists

## Required Interfaces With Other Phases

Interface from phase 3:

- planned groups and decisions define which operations are emitted
- grouped operations must expose order and member bindings clearly enough for
  faithful wrapper generation

Interface to phase 5:

- emission failures caused by underspecified semantics should surface directly,
  not be hidden behind repair loops
- bundle/debug artifacts should preserve canonical IR/planning context when
  emission fails

## Deterministic vs LLM Responsibilities

Deterministic:

- wrapper generation
- method argument mapping
- state injection/extraction
- output binding
- return type construction from canonical bindings

LLM:

- none for semantic emission
- existing witness drafting/opaque helper flows may remain, but wrapper
  semantics themselves must not depend on LLM guessing

## Rollout Plan

### Step 0. Lock Representative Emission Fixtures

Ensure tests cover:

- sklearn-style fit/predict/query workflow
- rolling/stateful class
- BioSPPy staged wrapper
- opaque DL boundary
- procedural ingest case
- Bayesian/message-passing cases that must not regress

### Step 1. Add Canonical Emission Helpers

In `emitter.py`, add helpers that derive deterministic wrapper specs from:

- canonical operations
- planned groups
- state slots
- output bindings

Possible helpers:

- `_canonical_wrapper_specs(...)`
- `_emit_ir_wrapper(...)`
- `_emit_ir_stateful_wrapper(...)`
- `_render_output_binding(...)`
- `_render_state_update(...)`

### Step 2. Emit Real Stateless Wrappers

Upgrade `generate_atom_wrappers(...)` for canonical IR-backed plans so it can:

- call the real source method/function where appropriate
- return bound outputs faithfully
- keep the old `NotImplementedError` path only for unsupported legacy cases

### Step 3. Emit Real Stateful Wrappers

Upgrade `generate_stateful_wrappers(...)` so it:

- injects required state slots
- calls exact methods with exact parameters
- captures return values where the binding says to
- extracts only written/updated state slots
- returns unchanged state for read-only query/metadata operations

### Step 4. Route Bundle Emission Through Canonical IR

Update `emit_ingestion_bundle(...)` to:

- prefer canonical emission when `canonical_ir` exists
- keep procedural and opaque paths working
- only use legacy macro-atom emission as fallback

### Step 5. Tighten Failure Semantics

If canonical IR/planning is present but insufficient for faithful emission:

- fail with a clear error
- stage debug artifacts that include canonical IR and planning graph
- do not silently fall back to invented legacy behavior

### Step 6. Expand Tests

- add emitter tests for each output binding kind
- add stateful tests for selective injection/extraction
- add grouped-operation emission tests
- add no-invention regression tests

## Concrete File Plan

Expected edits:

- `sciona/ingester/emitter.py`
  - main phase-4 implementation
- `sciona/ingester/models.py`
  - only if tiny additive fields/helpers are needed
- `sciona/ingester/graph.py`
  - only if better emission-failure artifacts are needed
- tests
  - `tests/test_ingester_emitter.py`
  - `tests/test_ingest_stateful.py`
  - `tests/test_ingest_biosppy_ecg.py`
  - `tests/test_ingest_config_flatten.py`
  - `tests/test_bayesian_ingester.py` if the canonical path affects it

## Regression Risks

Primary risks:

- canonical emission breaks legacy passing cases
- grouped planned operations lack enough detail for exact call emission
- state extraction becomes too narrow and drops necessary persisted fields
- metadata/query wrappers accidentally mutate or rehydrate wrong state

Mitigations:

- keep canonical-vs-legacy emission tests side by side during rollout
- fail closed on underspecified grouped operations
- preserve untouched state fields in `model_copy(update=...)`
- use direct parity assertions against representative upstream-style fixtures

## Test and Benchmark Plan

Direct emitter tests:

- `return_value` output returns the actual call result
- `attribute_read` output reads `obj.<source_attr>`
- `tuple_element` output extracts the correct tuple position
- `metadata_object` output returns the query object directly
- `self_return` does not create a fake data output

Stateful tests:

- required fitted state is injected before predict/query wrappers
- config state required by methods is injected
- only written state slots are updated
- read-only query methods return unchanged state

Grouped-operation tests:

- multi-method planned group emits ordered real method calls
- outputs and state updates come from the declared binding source

Protected integration cases:

- rolling/windowed class
- BioSPPy wrapper
- opaque DL boundary
- procedural ingest
- Bayesian/message-passing smoke coverage

## Acceptance Criteria

Phase 4 is complete when all of the following are true:

- canonical IR/planning drives wrapper emission when present
- emitted wrappers call real upstream methods with exact signatures
- fitted/config state is rehydrated before methods that require it
- emitted outputs come from declared output bindings, not guessed attrs
- metadata/query methods return metadata/query objects without fake mutation
- no wrapper invents outputs or state fields outside canonical evidence
- current protected ingest cases still pass
- emission failures surface clearly when canonical data is insufficient

## Deferred to Later Phases

Phase 5:

- tighten repair semantics around mechanical failures only
- make semantic emission failures surface immediately with useful artifacts

## Recommended Execution Order

1. Lock emission-focused regression fixtures.
2. Add canonical wrapper-spec helpers.
3. Upgrade stateless wrapper emission.
4. Upgrade stateful wrapper emission.
5. Route bundle emission through canonical IR.
6. Tighten failure semantics and expand tests.
