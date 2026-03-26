# REFINE_INGEST Phase 9 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 9 retires the transition architecture and makes canonical ingest
planning first-class throughout the runtime.

Phases 1 through 8 established:

- deterministic semantic facts
- canonical IR
- deterministic planning over that IR
- canonical wrapper emission
- canonical contract artifacts
- non-Python parity

The main remaining structural gap is that the runtime still executes through a
legacy compatibility layer:

- `chunker.py` still repeatedly lowers canonical IR back into
  `MacroAtomSpec` / `StateModelSpec`
- decomposition and enrichment still carry legacy atom trees as primary
  mutable state
- emitter entrypoints still accept legacy plan shapes as the main carrier
- procedural and canonical paths remain partially split

That means the system is semantically improved, but architecturally still
transitional.

The objective is:

- make canonical IR plus planning graph the primary execution format
- reduce legacy macro-atom/state-model structures to export compatibility only
- unify procedural and object-oriented ingestion around the same canonical
  runtime path
- keep the current contract outputs stable while removing semantic duplication

Key rule:

- after phase 9, canonical IR/planning should be the runtime source of truth,
  and legacy macro-atom/state-model views should exist only where an external
  compatibility surface still requires them

## Scope Boundaries

In scope:

- canonical-first runtime plan handling in `chunker.py`
- canonical-first decomposition/enrichment flow
- canonical-first emission entrypoints
- reducing or isolating legacy adapter generation
- unifying procedural plan generation with canonical runtime structures
- regression coverage for canonical-first execution

Out of scope:

- deleting all legacy model classes immediately if external tests still need
  them
- redesigning verification policy again
- broad CLI redesign
- replacing public output contract files or changing bundle filenames
- large extractor redesign beyond what canonical-first runtime integration
  needs

## Current Code Touchpoints

Primary implementation surfaces:

- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/models.py`
- `sciona/ingester/graph.py`

Secondary/supporting surfaces:

- `sciona/ingester/cache.py`
- `sciona/ingester/regression_harness.py`

Tests that should drive phase 9:

- `tests/test_ingester_chunker.py`
- `tests/test_chunker_depth.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_procedural.py`
- `tests/test_ingest_regression_harness.py`
- selected tree-sitter/FFI tests if canonical-first runtime touches them

## Current Gaps

The current runtime still has duplicated semantic representations.

Observed structural issues:

- `_attach_canonical_ir(...)` rebuilds a legacy `ProposedMacroPlan` carrying
  both canonical and adapter-produced legacy fields
- `_adapt_ir_to_legacy_plan(...)` and `_legacy_*` helpers keep recreating
  `macro_atoms`, `state_models`, and `edge_definitions` for downstream use
- decomposition still iterates over `plan.macro_atoms` even when canonical
  planning exists
- procedural ingestion builds a legacy `ValidatedMacroPlan` directly instead of
  producing canonical runtime state first
- emitter APIs still accept legacy macro-atom/state-model arguments as the main
  call shape, with canonical context threaded in as an override

Consequences:

- canonical semantics can still drift from adapter output during later passes
- more code has to keep adapter and canonical representations in sync
- future semantic changes remain harder than necessary because the pipeline
  still mutates the legacy view

## Phase 9 Deliverables

### 1. Canonical Runtime Plan Shape

Phase 9 should define a canonical-first runtime plan shape for the pipeline.

Recommended direction:

- keep `ValidatedMacroPlan` for compatibility if needed
- but make its primary executable content:
  - `canonical_ir`
  - `planning_graph`
  - canonical validation metadata
- treat legacy fields as derived/exported views, not mutable execution state

If needed, introduce a small helper/view layer such as:

- `CanonicalRuntimePlan`
- or canonical accessor helpers that remove direct mutation of `macro_atoms`

The exact name can vary, but the execution path must stop depending on legacy
structures as its working state.

### 2. Canonical-First Chunker and Planner Flow

Upgrade chunker/planner flow so each stage transforms canonical structures
directly.

Required behavior:

- proposal/lowering yields canonical IR first
- decomposition operates on canonical operations/groups
- enrichment/abstracting attaches metadata to canonical operations/groups
- validation checks canonical coverage and invariants directly
- legacy macro-atom trees are generated only when a compatibility surface
  explicitly asks for them

Important constraint:

- phase 9 should not regress deterministic planning rules or reintroduce
  LLM-heavy behavior

### 3. Canonical-First Emission Entry

Upgrade emitter entrypoints so the emitter consumes a canonical runtime plan,
not legacy macro-atoms with canonical overrides.

Required behavior:

- `emit_ingestion_bundle(...)` should prefer a canonical runtime plan interface
- internal wrapper/witness/CDG helpers should derive their working atom/group
  views from canonical operations/groups
- compatibility helpers may still materialize legacy views for niche output
  cases, but they should no longer drive the emitter

### 4. Canonical Procedural Ingestion

The procedural path should join the same runtime architecture.

Required behavior:

- procedural extraction should lower into canonical IR directly
- procedural emission should reuse canonical runtime machinery instead of a
  separate legacy plan builder
- procedural regressions should still remain low-friction and deterministic

### 5. Legacy Adapter Isolation

Do not necessarily delete the adapter in phase 9, but isolate it.

Recommended end state:

- one explicit compatibility export layer
- adapter code no longer spread through chunker/planner/emitter execution paths
- tests can still snapshot or compare legacy exports where useful, but runtime
  behavior is canonical-first

## Required Interfaces With Prior Phases

Interface from phase 2:

- canonical IR remains the only semantic source of truth for operations, state
  slots, outputs, and edges

Interface from phase 3:

- planning graph remains the decomposition source of truth

Interface from phase 4:

- canonical wrapper emission should continue unchanged in behavior while moving
  to canonical-first runtime inputs

Interface from phase 6:

- regression harness should continue to measure canonical behavior without
  needing to inspect adapter internals

Interface from phase 7:

- non-Python canonical cases must continue working through the canonical-first
  runtime

Interface from phase 8:

- witnesses/CDG/match metadata should continue to derive from canonical context
  without depending on legacy adapter state

## Deterministic vs LLM Responsibilities

Deterministic in phase 9:

- runtime plan shaping
- canonical decomposition and enrichment flow
- canonical emission inputs
- procedural-path unification
- compatibility export isolation

LLM responsibilities in phase 9:

- none beyond existing bounded fallback points already present in earlier
  phases
- phase 9 should remove architectural dependence on legacy shapes, not add new
  prompt-based transitions

## Data Model Changes

Expected additive model work:

- small canonical runtime plan helper/model if useful
- explicit flags or helper accessors for compatibility-export generation
- cache payload updates if canonical-first runtime state needs clearer
  serialization

Avoid:

- duplicating canonical semantics into another parallel runtime IR
- breaking existing bundle serialization without a clear migration path

## Rollout Plan

### Step 0. Lock Canonical-First Regression Slice

Before refactoring, confirm tests cover:

- canonical chunker/planner behavior
- canonical emitter behavior
- procedural ingest behavior
- harness runs across representative families

Add any missing focused tests first.

### Step 1. Introduce Canonical Runtime Accessors / Helper Model

- add a small canonical runtime abstraction if needed
- centralize how runtime code accesses operations, groups, state slots, and
  compatibility exports
- avoid large model churn if helper accessors are sufficient

### Step 2. Refactor Chunker to Operate on Canonical Runtime State

- reduce repeated `macro_atoms` mutations
- move proposal/decomposition/enrichment to canonical operations/groups
- keep compatibility export generation at the boundaries only

### Step 3. Refactor Procedural Path Into Canonical Runtime Flow

- replace direct legacy plan construction with canonical lowering
- ensure procedural bundle emission still matches current contract outputs

### Step 4. Refactor Emitter Entry to Canonical Runtime Inputs

- make canonical runtime state the default input
- isolate any remaining adapter materialization to narrow compatibility helpers

### Step 5. Isolate Legacy Export Layer

- move remaining legacy adapter code behind explicit helper functions/modules
- remove adapter dependence from the main control flow

### Step 6. Expand Regression Coverage

- add tests that assert canonical-first execution no longer depends on legacy
  mutation
- keep harness, procedural, stateful, and non-Python slices green

## Concrete File Plan

Expected edits:

- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/models.py`
- `sciona/ingester/graph.py`
- optionally `sciona/ingester/cache.py`
- optionally `sciona/ingester/regression_harness.py`

Tests:

- `tests/test_ingester_chunker.py`
- `tests/test_chunker_depth.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_procedural.py`
- `tests/test_ingest_regression_harness.py`

## Regression Risks

Primary risks:

- canonical-first refactor breaks compatibility assumptions in emitter helpers
- procedural ingest regresses while being unified
- cached bundles or staged artifacts lose required fields
- some tests still implicitly depend on legacy macro-atom mutation order

Mitigations:

- keep legacy export helpers available during rollout
- refactor in additive steps with side-by-side regression coverage
- preserve bundle/file output contract exactly unless a test intentionally
  changes
- verify stateful, procedural, non-Python, and harness slices together

## Test and Benchmark Plan

Direct canonical-runtime tests:

- chunker returns canonical-first validated plans without needing mutable legacy
  atom trees
- emitter accepts canonical runtime state and preserves current wrapper/witness
  outputs
- procedural ingestion uses canonical lowering path

Protected regression slice:

- `tests/test_ingester_chunker.py`
- `tests/test_chunker_depth.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_procedural.py`
- `tests/test_ingest_regression_harness.py`
- selected tree-sitter/FFI tests if touched

Harness expectation:

- representative cases still complete with no drop in canonical artifact
  availability

## Acceptance Criteria

Phase 9 is complete when all of the following are true:

- canonical IR/planning is the runtime source of truth in chunking and
  emission
- procedural ingest runs through the same canonical-first runtime architecture
- legacy macro-atom/state-model adapters are isolated to explicit compatibility
  export paths
- contract outputs remain stable and regression slices stay green
- future semantic changes can target canonical runtime state without needing to
  keep multiple mutable semantic representations in sync

## Deferred to Later Work

Not required in phase 9:

- deleting all legacy classes immediately
- major public API changes for `sciona ingest`
- redesigning verification or repair loops again
- broader productization of benchmark/harness reporting

Phase 9 should finish the architectural transition to canonical-first runtime
execution, not broaden the ingest feature set.
