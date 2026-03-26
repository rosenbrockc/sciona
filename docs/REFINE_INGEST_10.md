# REFINE_INGEST Phase 10 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 10 reduces the remaining legacy compatibility burden now that
canonical-first runtime execution exists.

After phase 9, canonical IR and planning are the runtime source of truth, but
the compatibility layer still exists in several visible forms:

- runtime code still calls compatibility materialization helpers in multiple
  places
- `macro_atoms` and `state_models` are still threaded through many helper
  signatures even when canonical context is primary
- tests still frequently assert legacy-shape details first and canonical
  behavior second

The objective is:

- isolate compatibility export generation behind a narrower boundary
- reduce eager or repeated legacy materialization in chunker/emitter runtime
- move tests and helper APIs toward canonical-first expectations
- preserve current output contract and protected-family behavior

Key rule:

- phase 10 should simplify the architecture without changing the semantic source
  of truth or reintroducing new parallel runtime representations

## Scope Boundaries

In scope:

- reducing repeated calls to compatibility materialization helpers
- tightening canonical-first helper/accessor patterns
- moving adapter logic toward a more explicit export boundary
- updating tests so canonical-first behavior is asserted more directly
- small runtime API cleanups that reduce legacy-first assumptions

Out of scope:

- deleting all legacy classes outright
- redesigning verification/repair behavior
- major CLI or cache redesign
- large extractor changes
- changing public bundle/file outputs

## Current Code Touchpoints

Primary implementation surfaces:

- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`

Potential supporting surfaces:

- `sciona/ingester/graph.py`
- `sciona/ingester/cache.py`

Tests that should drive phase 10:

- `tests/test_ingester_chunker.py`
- `tests/test_chunker_depth.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_procedural.py`
- `tests/test_ingest_regression_harness.py`

## Current Gaps

Phase 9 made runtime execution canonical-first, but the code still carries
signs of transition.

Observed issues:

- compatibility materialization still happens in multiple runtime entrypoints:
  - `materialize_legacy_plan_views(...)`
  - `_runtime_compat_plan(...)`
  - `_runtime_plan(...)`
- chunker stages still frequently operate over `plan.macro_atoms` after
  materializing compatibility exports
- emitter helpers still accept `macro_atoms` / `state_models` as core inputs
  even when canonical context is available
- many tests still build or assert legacy-shaped plans directly
- the adapter boundary exists, but it is still too visible to the rest of the
  runtime

This is not a semantic correctness bug, but it still costs:

- extra code paths to maintain
- more chances for compatibility drift
- higher cognitive overhead for future planners and implementers

## Phase 10 Deliverables

### 1. Narrower Compatibility Boundary

Phase 10 should make compatibility exports more clearly export-only.

Recommended direction:

- keep one explicit helper boundary for generating legacy views
- avoid ad hoc runtime materialization at multiple call sites
- make it obvious which functions require canonical runtime state vs
  compatibility exports

The goal is not to remove every adapter artifact, but to make it easier to see
where the adapter begins and ends.

### 2. Canonical-First Runtime Accessors

Introduce or strengthen canonical-first accessors/helpers so runtime code does
not need to reach for legacy fields by default.

Good targets:

- helper access to runtime operations/groups
- helper access to compatibility exports only when needed
- smaller helper APIs that take a validated canonical runtime plan rather than
  raw `macro_atoms` / `state_models`

### 3. Chunker Simplification

Reduce chunker dependence on compatibility materialization in day-to-day flow.

Priority behaviors:

- proposal / config flattening / state hoisting / abstracting should avoid
  materializing legacy views earlier than necessary
- decomposition should consume canonical runtime state first and only materialize
  legacy children where existing surfaces still require them
- tests should prove canonical-first decomposition still works when legacy views
  are empty or absent

### 4. Emitter Helper Simplification

Continue the emitter shift from legacy-first helper signatures toward
canonical-first helper signatures.

Priority behaviors:

- internal helper inputs should increasingly accept plan-level canonical context
  rather than separate legacy arrays
- compatibility exports may still be used for final source shaping where the
  public contract still expects them, but not as the default semantic input

### 5. Canonical-First Test Expectations

Update tests so they reflect the architecture we now want to preserve.

This means:

- canonical runtime behavior should be the first assertion
- compatibility export behavior should be asserted only where intentionally
  preserved
- regression tests should stay focused on output contract stability, not
  historical adapter internals

## Required Interfaces With Prior Phases

Interface from phase 9:

- canonical IR/planning is already the runtime source of truth
- compatibility exports already exist as explicit materialization helpers

Interface to later work:

- phase 11 should expand the real-world regression corpus against the simplified
  runtime surface, not against transitional adapter-heavy code paths

## Deterministic vs LLM Responsibilities

Deterministic in phase 10:

- runtime simplification
- helper/API cleanup
- compatibility export isolation
- test updates

LLM responsibilities in phase 10:

- none for the core work
- no new prompt dependency should be introduced here

## Data Model Changes

Expected additive work:

- small helper APIs or explicit compatibility-export accessors in
  `models.py`
- optional small type or helper changes that reduce legacy-field dependence

Avoid:

- introducing a second canonical runtime wrapper model unless it clearly removes
  complexity
- changing canonical semantic models themselves unless required by the cleanup

## Rollout Plan

### Step 0. Lock the Regression Slice

Before refactoring, confirm coverage for:

- chunker canonical-first behavior
- emitter canonical-first behavior
- stateful behavior
- procedural behavior
- regression harness slice

### Step 1. Tighten Compatibility Helpers

- centralize or clarify compatibility export access
- remove redundant helper layers where possible

### Step 2. Simplify Chunker Flow

- reduce unnecessary materialization of legacy exports
- keep canonical runtime state primary throughout proposal/decomposition/enrich
  stages

### Step 3. Simplify Emitter Helper Inputs

- move more emitter internals toward plan-level canonical context
- keep compatibility use narrow and explicit

### Step 4. Update Tests

- add or tighten tests proving canonical-first behavior still succeeds when
  legacy exports are empty or stale
- reduce tests that only assert transitional adapter details

## Concrete File Plan

Expected edits:

- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`

Tests:

- `tests/test_ingester_chunker.py`
- `tests/test_chunker_depth.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_procedural.py`
- optionally `tests/test_ingest_regression_harness.py`

## Regression Risks

Primary risks:

- simplification removes compatibility materialization still needed by one path
- tests become too cleanup-oriented and stop protecting actual output behavior
- stateful or procedural paths regress because they still rely on narrow legacy
  exports in a few places

Mitigations:

- preserve contract outputs exactly
- keep canonical-first and compatibility-export tests side by side while
  refactoring
- run stateful, procedural, emitter, and harness slices together

## Test and Benchmark Plan

Direct tests:

- canonical runtime helpers work when `macro_atoms` / `state_models` are empty
- chunker still decomposes and enriches correctly from canonical state
- emitter still emits correct bundles from canonical plans
- procedural plans continue to carry canonical IR and emit correctly

Protected regression slice:

- `tests/test_ingester_chunker.py`
- `tests/test_chunker_depth.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_procedural.py`
- `tests/test_ingest_regression_harness.py`

## Acceptance Criteria

Phase 10 is complete when all of the following are true:

- compatibility export generation is more clearly isolated
- runtime code depends less on eager legacy materialization
- canonical-first behavior is asserted directly in the relevant tests
- contract outputs remain unchanged for protected cases
- the codebase is simpler for future planners to extend without reopening the
  legacy transition architecture

## Deferred to Later Work

Phase 11:

- broaden the regression corpus and add real-world golden snapshots

Phase 12:

- performance, cache, and artifact stability work
