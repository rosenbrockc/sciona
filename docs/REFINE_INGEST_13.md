# REFINE_INGEST Phase 13 Implementation Plan

## Phase Goal

Phase 13 removes more dead or low-value transitional surface area now that the
ingest runtime is canonical-first end-to-end and the operational stabilization
work through phase 12 is complete.

The objective is:

- identify transitional helper paths that no longer add real compatibility
  value
- delete or narrow those paths without changing canonical semantics
- make compatibility exports more obviously export-only
- reduce the amount of runtime and test code that still reasons in legacy-first
  terms

Key rule:

- this is a cleanup and simplification phase, not a semantic redesign phase; it
  must preserve canonical runtime behavior and the current ingest contract

## Scope Boundaries

In scope:

- deleting dead transitional helper functions or call paths
- narrowing compatibility-export plumbing where runtime no longer needs it
- simplifying chunker/emitter helper signatures that still carry legacy-first
  assumptions
- tightening tests so they assert intended canonical-first behavior directly

Out of scope:

- changing canonical extraction, planning, or emission semantics
- changing verification/repair policy
- broad cache or monitor redesign
- removing deliberately preserved public compatibility exports without a clear
  replacement or explicit decision

## Current Code Touchpoints

Primary implementation surfaces:

- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`

Representative tests that should drive the work:

- `tests/test_ingester_chunker.py`
- `tests/test_chunker_depth.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_procedural.py`
- `tests/test_ingest_regression_harness.py`

## Current Transitional Surface

The runtime is already canonical-first, but several transitional surfaces still
exist:

### 1. Compatibility Materialization Helpers In `models.py`

Current helpers include:

- `legacy_macro_atoms_from_ir(...)`
- `legacy_state_models_from_ir(...)`
- `legacy_edges_from_ir(...)`
- `runtime_macro_atoms(...)`
- `runtime_state_models(...)`
- `runtime_edge_definitions(...)`
- `materialize_legacy_plan_views(...)`

Not all of these are necessarily dead, but phase 13 should determine which ones
are still needed as explicit compatibility exports and which ones are only
leftover transition scaffolding.

### 2. Legacy-Plan-First IR Construction And Adaptation In `chunker.py`

The chunker still carries transitional shapes such as:

- `_build_state_slots(..., legacy_plan=...)`
- `_slot_type_hint(..., legacy_plan)`
- `_build_ingest_ir(..., legacy_plan=...)`
- `_adapt_ir_to_legacy_plan(...)`
- `_build_canonical_plan(..., materialize_compatibility=...)`

These may still be needed in part, but the planner should assume some of them
now exist mainly to bridge the earlier migration and may be simplifiable.

### 3. Legacy Fallback Decomposition Path In `chunker.py`

The explicit legacy path remains:

- `_legacy_decompose_single_atom(...)`

This path was intentionally retained for backward compatibility during phases 3
and 9, but phase 13 should reassess whether it still needs to exist as a full
runtime branch or whether it can be narrowed, isolated, or deleted.

### 4. Legacy-Shaped Emitter Helper Inputs In `emitter.py`

The emitter now reads runtime atoms/state/edges through canonical-first helpers,
but many helper signatures still accept:

- `macro_atoms`
- `state_models`
- `edge_definitions`

That is not automatically wrong, but some of those helpers may now be carrying
legacy-shaped arguments only because of migration history rather than current
need.

## Main Problems To Solve

### Problem 1: Too Many Helper Layers Still Exist

Even when the runtime is behaving correctly, too many helper layers still say
“legacy”, “compatibility”, or “runtime adaptation” without a clear boundary
between:

- truly supported compatibility exports
- temporary migration helpers that should now be deleted

### Problem 2: Some Tests Still Protect Transitional Internals

Several cleanup-oriented tests added during phases 9-10 were useful, but phase
13 should ensure test coverage now protects:

- canonical-first runtime behavior
- explicit export-only compatibility behavior where intentionally preserved

and not accidental transitional plumbing details.

### Problem 3: Fallback Paths Increase Maintenance Cost

The more transition-only branches remain, the harder later changes become,
especially in:

- decomposition
- state hoisting
- CDG/sub-graph emission
- procedural/runtime compatibility

## Phase 13 Deliverables

### 1. Explicit Compatibility Boundary

Phase 13 should leave one clear story for compatibility exports:

- which helpers are the supported export boundary
- which helpers are deleted because the runtime no longer needs them

Recommended direction:

- keep narrow runtime accessors if they still improve clarity
- remove broad “materialize everything” helpers that only support dead paths
- make it obvious which code is runtime and which code is export-only

### 2. Chunker Cleanup

Reduce or isolate transitional logic in the chunker.

Priority targets:

- narrow the `legacy_plan` threading through IR construction helpers where
  canonical facts already provide the needed data
- reassess whether `_adapt_ir_to_legacy_plan(...)` still needs multiple modes
- reassess whether `_legacy_decompose_single_atom(...)` should remain a full
  branch, an isolated compatibility shim, or be removed

### 3. Emitter Cleanup

Reduce emitter dependence on legacy-shaped helper signatures where the helper is
really consuming canonical runtime exports.

The main desired effect is less ambiguity, not a giant emitter rewrite.

### 4. Test Cleanup

Update tests so they assert:

- canonical-first runtime behavior first
- compatibility exports only where intentionally preserved
- protected-family output stability rather than migration scaffolding

## Required Interfaces With Prior Work

Interface from phases 9-12:

- canonical IR/planning is already the runtime source of truth
- compatibility exports remain available
- the phase-11 regression corpus and phase-12 operational tests provide the
  confidence baseline

Phase 13 should simplify on top of that base rather than reopen architectural
questions already settled.

Interface to later work:

- the result should make future maintenance simpler
- any remaining compatibility surface after phase 13 should be intentional and
  documented by code structure, not merely by history

## Deterministic vs LLM Responsibilities

Deterministic in phase 13:

- helper deletion and simplification
- boundary clarification
- test updates
- regression verification

LLM responsibilities in phase 13:

- none for the core work

## Data Model Changes

Expected additive or subtractive work:

- possible deletion of no-longer-needed helper functions
- possible narrowing of helper signatures or optional arguments
- possible movement of compatibility-export helpers into a smaller explicit
  access layer if that materially reduces confusion

Avoid:

- changing core canonical semantic models
- broad schema changes
- introducing a second new abstraction layer just to hide transitional code

## Rollout Plan

### Step 0. Lock The Regression Slice

Before deleting helpers, keep coverage for:

- canonical-first chunker behavior
- canonical-first emitter behavior
- stateful emission
- procedural emission
- regression-harness compatibility

### Step 1. Inventory Transitional Helpers

- classify helper functions as:
  - runtime-authoritative
  - compatibility-export-only
  - dead transitional scaffolding
- remove only the clearly dead category first

### Step 2. Simplify Models-Level Compatibility Access

- keep or refine the minimal runtime/export helper surface in `models.py`
- delete broader materialization helpers if runtime no longer needs them

### Step 3. Simplify Chunker Branching

- reduce `legacy_plan` and compatibility-mode branching where canonical context
  already supplies the needed information
- narrow or remove legacy-only decomposition paths if no longer justified

### Step 4. Simplify Emitter Inputs

- remove legacy-shaped helper plumbing that no longer affects output behavior
- keep explicit runtime-export access where still needed

### Step 5. Tighten Tests

- update tests to assert intended architecture directly
- ensure cleanup does not regress protected families

## Concrete File Plan

Expected implementation edits:

- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`

Likely tests:

- `tests/test_ingester_chunker.py`
- `tests/test_chunker_depth.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_procedural.py`
- optionally `tests/test_ingest_regression_harness.py`

## Regression Risks

Primary risks:

- deleting a helper that still supports a real compatibility path
- simplifying decomposition in a way that regresses older edge cases
- removing legacy-shaped emitter plumbing that still affects stateful or
  procedural outputs
- tests becoming too cleanup-specific and losing user-facing protection

Mitigations:

- classify helpers before deleting them
- preserve representative protected-family regression slices
- keep canonical-first behavior and output artifacts as the main test target
- prefer narrow explicit compatibility shims over silent behavior changes

## Test Plan

Minimum local regression slice:

- `pytest -q tests/test_ingester_chunker.py tests/test_chunker_depth.py`
- `pytest -q tests/test_ingester_emitter.py tests/test_ingest_stateful.py tests/test_ingest_procedural.py`
- optionally `pytest -q tests/test_ingest_regression_harness.py`

If a cleanup touches decomposition fallback or compatibility exports deeply,
expand to the phase-11/12 protected-family slice before closing the phase.

## Acceptance Criteria

Phase 13 is complete when:

- dead transitional helpers are removed or clearly narrowed
- compatibility exports are more obviously export-only
- runtime call paths are simpler to read and maintain
- protected-family behavior remains intact
- tests assert canonical-first architecture directly rather than preserving
  historical migration scaffolding

## Deferred To Later Work

Explicitly defer:

- semantic redesign
- another verification/repair refactor
- broad CI or documentation work that belongs to other recommendations
- large operational-surface changes already handled in phase 12
