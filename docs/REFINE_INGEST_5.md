# REFINE_INGEST Phase 5 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 5 tightens verification, narrows repair to mechanical failures, and makes
failure semantics explicit.

The objective is not “repair more.” It is:

- verify strictly
- repair only when the failure is narrow and mechanical
- surface semantic/design errors immediately with useful artifacts
- avoid long or flaky LLM repair loops masking upstream planning/emission bugs

## Scope Boundaries

In scope:

- classify verification failures into repairable vs non-repairable
- prefer deterministic repair for allowed repair classes
- reduce or remove LLM repair for semantic mismatches
- preserve and improve debug artifacts on failure
- keep mypy/ghost/message-cycle verification strict

Out of scope:

- major new emitter/planner behavior
- broad OpenAI/LLM router redesign
- full regression harness implementation beyond tests needed here

Key rule:

- semantic mismatches are design errors, not repair opportunities

## Current Code Touchpoints

Primary implementation surface:

- `sciona/ingester/graph.py`

Existing repair and verification helpers:

- `verify_types(...)`
- `verify_ghost(...)`
- `repair_types(...)`
- `repair_ghost(...)`
- `repair_message_cycle(...)`
- `route_after_typecheck(...)`
- `route_after_ghost(...)`

Existing deterministic repair helpers:

- `sciona/ingester/deterministic_type_fixer.py`
- `sciona/ingester/deterministic_ghost_fixer.py`
- `sciona/ingester/deterministic_cycle_breaker.py`

Prompt surfaces that should shrink in importance:

- `sciona/ingester/prompts.py`

Artifact/monitor surfaces:

- `sciona/ingester/monitor.py`
- typecheck debug publication in `graph.py`

Tests that should drive phase 5:

- `tests/test_deterministic_type_fixer.py`
- `tests/test_deterministic_ghost_fixer.py`
- `tests/test_deterministic_cycle_breaker.py`
- `tests/test_ingest_config_flatten.py`
- `tests/test_message_passing.py`
- `tests/test_ingest_monitor.py`
- any phase-4 emitter/stateful tests that now rely on fail-closed behavior

## Current Gaps

The current pipeline still uses bounded LLM repair loops as the default
response to many verification failures.

Current problems:

- semantic mistakes can still enter repair paths instead of failing fast
- `repair_types(...)` and `repair_ghost(...)` still route through LLM prompts,
  even though deterministic fixers already exist for some classes
- routing decisions are based mostly on pass/fail and retry count, not on
  failure class
- debug artifacts are improving, but failure payloads still do not clearly say
  “repairable mechanical” vs “non-repairable design error”

Phase 5 should turn repair from a generic fallback into a narrow, deterministic
post-processing step.

## Failure Taxonomy

Phase 5 should introduce an explicit verification failure taxonomy.

Recommended classes:

- `mechanical_import`
  - missing import
  - missing generated-module import
- `mechanical_annotation`
  - annotation normalization
  - trivial return wrapper mismatch
- `mechanical_reference`
  - simple name/reference mismatch that deterministic patchers already know
- `message_cycle`
  - cyclic message-passing topology repairable by deterministic breaker
- `semantic_output_binding`
  - wrong output source
  - nonexistent attribute read
- `semantic_state_rehydration`
  - missing fitted/config state before call
- `semantic_signature`
  - wrong arguments or call style
- `semantic_query_mutation`
  - metadata/query wrapper behaving like mutator
- `unknown_or_unclassified`

Only the first mechanical classes, plus specialized message-cycle repair,
should be repairable in phase 5.

## Repair Policy

### Repairable Failures

Allowed repair classes:

- missing imports
- safe annotation normalization
- trivial reference rewrites already covered by deterministic fixers
- message-passing cycle breaking

Required policy:

- use deterministic repair first
- if deterministic repair cannot classify/apply a patch, prefer fail-fast over
  escalating into generic LLM repair

LLM fallback should be allowed only if:

- there is already a deterministic fixer wrapper that intentionally delegates
  for a narrow known-safe remainder, and
- the resulting patch class is still mechanical

### Non-Repairable Failures

These must fail immediately:

- wrong call signature or argument ordering
- wrong output binding kind/source
- wrong state injection/rehydration
- invented outputs or state fields
- metadata/query semantic mismatches
- any canonical-IR/planning underspecification surfaced by phase 4

For these:

- publish artifacts
- classify failure as non-repairable semantic/design error
- stop the graph without entering generic repair loops

## Verification Flow Changes

Phase 5 should turn verification into:

1. verify
2. classify failure
3. if repairable and deterministic patch exists, apply once
4. re-verify
5. otherwise fail with artifacts

This should replace the current “fail -> bounded LLM repair loop” default.

### Type Verification

`verify_types(...)` should:

- preserve current multi-file checking
- classify errors before routing
- attach classification to state/debug artifacts

`repair_types(...)` should:

- prefer deterministic type fixer only
- stop incrementing retries for non-repairable semantic failures
- avoid generic LLM fix prompts for semantic mismatches

### Ghost Verification

`verify_ghost(...)` should:

- classify witness failures vs semantic wrapper failures

`repair_ghost(...)` should:

- use deterministic ghost fixer only for supported stub-like failures
- fail fast for semantic mismatches

### Message-Passing Cycle Repair

Keep `repair_message_cycle(...)` as the specialized deterministic path.

Required behavior:

- route there only for explicit cyclic-deadlock signatures
- keep it separate from generic ghost repair

## Failure Artifact Strategy

Phase 5 should preserve and improve existing staged artifacts.

Required artifacts on non-repairable failure:

- emitted bundle files at failure boundary
- raw DFG
- validated plan
- canonical IR
- planning graph if present
- verification error text
- failure classification payload

Recommended additions:

- `verification_failure.json`
  - stage
  - classifier result
  - repairable boolean
  - reason code
  - retry counts

Key rule:

- failures should be actionable from saved artifacts without rerunning the full
  ingest path

## Required Interfaces With Other Phases

Interface from phase 4:

- canonical emission now fails closed on underspecified semantics
- phase 5 must treat those failures as non-repairable design errors

Interface to phase 6:

- regression harness should track:
  - verification pass rate
  - deterministic repair count
  - non-repairable semantic failure count
  - LLM repair count, which should trend toward zero

## Deterministic vs LLM Responsibilities

Deterministic:

- failure classification
- import/annotation/reference fixes already supported
- cycle breaking
- artifact publication

LLM:

- ideally none for ingest repair by the end of phase 5
- if retained at all, only as a narrowly scoped fallback behind explicit
  mechanical classification

## Rollout Plan

### Step 0. Add Failure Classifier

Introduce a small deterministic classifier in `graph.py` or a new helper module
that inspects mypy/ghost error text and returns:

- `reason_code`
- `repairable`
- `repair_path`

### Step 1. Rework Type Repair Routing

- classify typecheck failures before routing
- only route to `repair_types` for repairable mechanical classes
- otherwise terminate with staged artifacts

### Step 2. Rework Ghost Repair Routing

- classify ghost failures before routing
- keep cycle detection on its own route
- only route generic ghost repair for deterministic ghost-fixer-supported cases

### Step 3. Narrow Repair Implementations

- make `repair_types(...)` deterministic-first and fail-fast
- make `repair_ghost(...)` deterministic-first and fail-fast
- keep retry counts meaningful only for actually attempted repairs

### Step 4. Improve Failure Artifacts

- write explicit failure classification payloads
- include canonical IR/planning info when present
- keep partial publication on failure

### Step 5. Expand Tests

- add routing tests for repairable vs non-repairable failures
- add fail-fast tests for semantic mismatches
- preserve deterministic fixer and message-cycle tests
- ensure monitor artifacts still publish correctly

## Concrete File Plan

Expected edits:

- `sciona/ingester/graph.py`
  - main routing, classification, and artifact changes
- optional new helper module
  - e.g. `sciona/ingester/verification_classifier.py`
- `sciona/ingester/deterministic_type_fixer.py`
  - only if small classifier-facing helper hooks are useful
- `sciona/ingester/deterministic_ghost_fixer.py`
  - only if small classifier-facing helper hooks are useful
- tests
  - `tests/test_deterministic_type_fixer.py`
  - `tests/test_deterministic_ghost_fixer.py`
  - `tests/test_deterministic_cycle_breaker.py`
  - `tests/test_message_passing.py`
  - optionally a new focused routing/classifier test file

## Regression Risks

Primary risks:

- classifier is too aggressive and blocks safe mechanical repairs
- classifier is too permissive and semantic bugs still enter repair loops
- artifact publication regresses during new early-fail paths

Mitigations:

- default unknown/unclassified failures to non-repairable
- keep deterministic fixer tests strong
- snapshot artifact publication in failure tests

## Test and Benchmark Plan

Direct classifier/routing tests:

- missing import -> repairable type fix
- annotation normalization -> repairable type fix
- wrong attribute output source -> non-repairable semantic fail
- wrong method signature -> non-repairable semantic fail
- cyclic message graph -> specialized deterministic cycle repair
- unknown ghost failure -> fail fast, not generic looping

Integration tests:

- existing config-flatten repair-loop tests updated to assert narrower repair
- message-passing routing tests preserved
- monitor/debug artifact tests confirm failure payload publication

Metrics to track:

- deterministic repair count
- LLM repair count
- non-repairable semantic failure count
- verification pass rate after emission improvements

## Acceptance Criteria

Phase 5 is complete when all of the following are true:

- verification failures are classified deterministically
- only narrow mechanical failures enter repair paths
- semantic/design failures surface immediately with useful artifacts
- deterministic type/ghost/cycle fixers remain functional
- generic LLM repair dependence is eliminated or reduced to a narrowly defined
  mechanical fallback
- current protected tests still pass

## Deferred to Later Phases

Phase 6:

- full regression harness/benchmark tracking across representative families

## Recommended Execution Order

1. Add deterministic failure classifier.
2. Rework verification routing around repairable vs non-repairable failures.
3. Narrow type and ghost repair implementations.
4. Improve failure artifact publication.
5. Expand tests and measure repair-count reduction.
