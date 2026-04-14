# Unified Artifact Phase 4: Direct Macro Retrieval

## Status

Drafted on April 14, 2026 as Phase 4 of
[Unified Artifact Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_IMPLEMENTATION_PLAN.md).

## Goal

Add a planner-stage macro retrieval path that can choose a published CDG before
falling back to direct atom grounding or fresh decomposition.

This phase is the main runtime change that improves the probability of finding
the best existing solution rather than reinventing it.

## Problem

The current planner direct path is declaration-oriented. It tries Hunter first
and only decomposes afterward. That is the wrong order for reusable macro
solutions whose contracts match the user goal better than any single primitive.

## Scope

This phase covers:

- macro retrieval before the current Hunter direct-match path
- macro candidate ranking based on contract, text, and metadata fit
- one-node artifact-selection execution paths when a macro match wins
- deterministic fallback to the existing Hunter and Architect flow when macro
  retrieval does not win

## Non-Goals

This phase does not:

- change template retrieval inside the architect
- change refinement-time substitution
- expose public artifact APIs
- refactor Hunter to verify arbitrary graphs as declarations

## Files In Scope

Primary runtime files:

- [sciona/services/planner_service.py](/Users/conrad/personal/sciona-matcher/sciona/services/planner_service.py)
- [sciona/runtime_paths.py](/Users/conrad/personal/sciona-matcher/sciona/runtime_paths.py)
- [sciona/services/hunter_service.py](/Users/conrad/personal/sciona-matcher/sciona/services/hunter_service.py)
- [sciona/services/models.py](/Users/conrad/personal/sciona-matcher/sciona/services/models.py)

Primary tests:

- planner and direct-path tests under [tests](/Users/conrad/personal/sciona-matcher/tests)

## Implementation Steps

### Step 1: Add a macro retrieval service boundary

Do not bury the new logic inline in the planner. Add a distinct helper or
service that returns:

- candidate artifact identity
- confidence / score breakdown
- derived execution metadata
- the reason macro retrieval was skipped or rejected

This keeps Phase 4 and Phase 5 from forking separate score semantics.

### Step 2: Define macro ranking features

Minimum ranking inputs:

- top-level IO contract fit
- goal text and dejargonized description similarity
- domain/modality hints
- publishability state
- verified leaf coverage
- visibility/tier constraints

The ranking should be deterministic. No LLM should sit in the retrieval loop.

### Step 3: Insert macro retrieval before Hunter

Update the planner flow to:

1. attempt direct macro retrieval
2. if strong enough, emit an artifact-selected path
3. otherwise continue to the current Hunter direct path
4. if that fails, continue to Architect decomposition as today

The current direct path telemetry should remain readable after this change.

### Step 4: Build artifact-selected execution packaging

When macro retrieval succeeds, return a minimal CDG or execution wrapper that
preserves:

- selected artifact identity and version
- why it was chosen
- what contract it satisfied
- enough structure for downstream execution and observability

This should not fake a declaration result.

### Step 5: Add deterministic fallback semantics

Define explicit reasons such as:

- no contract-compatible artifact
- macro score below threshold
- tier or visibility exclusion
- artifact present but insufficient verification coverage

These reasons should be visible in telemetry so macro retrieval can be tuned.

## Testing Plan

Add or extend tests for:

- macro retrieval winning over Hunter when a strong published CDG exists
- fallback to Hunter when macro retrieval is weak or filtered out
- fallback to decomposition when both direct paths fail
- telemetry distinguishing `macro_direct`, `single_agent_direct`, and
  decomposition paths
- threshold behavior on close-score candidates

## Worker Breakdown

Recommended ownership:

- one worker owns the planner/runtime-path implementation and its tests

Not recommended:

- concurrent edits to planner control flow from multiple workers

## Dependencies

- requires Phase 1
- requires meaningful artifact population from Phases 2 and 3
- can run in parallel with Phase 5 once shared score semantics are agreed

## Parallelization Notes

- safe to parallelize with Phase 5 because Phase 4 owns planner entry-point
  behavior, not architect/template reuse
- one integrator should still own the shared config keys and telemetry labels

## Risks

- weak macro ranking can become a noisy shortcut that blocks better leaf plans
- opaque thresholds will make retrieval behavior hard to debug
- returning macro selections as fake declaration matches will corrupt existing
  assumptions

## Exit Criteria

- the planner can select a published macro artifact before Hunter
- fallback semantics are deterministic and observable
- existing Hunter and decomposition paths still work when macro retrieval does
  not win
