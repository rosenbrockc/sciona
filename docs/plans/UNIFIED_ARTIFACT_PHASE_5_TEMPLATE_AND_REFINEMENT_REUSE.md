# Unified Artifact Phase 5: Template And Refinement Reuse

## Status

Drafted on April 14, 2026 as Phase 5 of
[Unified Artifact Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_IMPLEMENTATION_PLAN.md).

## Goal

Teach the architect and refinement loop to reuse published CDGs as macro
templates and substitution candidates.

This phase is about graph-level reuse after the planner has already entered the
structured path.

## Problem

Today the architect can benefit from template retrieval, but the system does
not treat published CDGs as first-class reusable exemplars. As a result, strong
existing decompositions do not reliably terminate or steer decomposition early.

## Scope

This phase covers:

- indexing published CDGs as reusable templates
- boosting strong published artifacts during template retrieval
- allowing high-confidence template matches to terminate or shortcut fresh
  decomposition
- checking for published macro substitutions before recursively refining a
  failed node

## Non-Goals

This phase does not:

- change the planner direct path ordering
- expose public artifact APIs
- replace leaf Hunter verification
- remove Memgraph as the graph retrieval layer

## Files In Scope

Primary runtime files:

- [sciona/architect/nodes.py](/Users/conrad/personal/sciona-matcher/sciona/architect/nodes.py)
- [sciona/architect/template_retriever.py](/Users/conrad/personal/sciona-matcher/sciona/architect/template_retriever.py)
- [sciona/architect/graph_retrieval.py](/Users/conrad/personal/sciona-matcher/sciona/architect/graph_retrieval.py)
- [sciona/orchestrator.py](/Users/conrad/personal/sciona-matcher/sciona/orchestrator.py)
- [sciona/graph_store.py](/Users/conrad/personal/sciona-matcher/sciona/graph_store.py)

Primary tests:

- architect, refinement, and telemetry tests under
  [tests](/Users/conrad/personal/sciona-matcher/tests)

## Implementation Steps

### Step 1: Add published CDGs to template retrieval inputs

Extend the retrieval input set so templates come from both:

- historical decomposition traces
- published CDG artifacts projected into Memgraph

The retrieval layer must preserve provenance so the runtime knows whether a
match came from a published artifact or an execution trace.

### Step 2: Add ranking boosts for trusted published macros

Boost, but do not blindly force, candidates with:

- strong topological similarity
- good contract alignment
- high verified leaf coverage
- publishable status
- good audit or uncertainty posture if available

This should make high-quality published CDGs more likely to win without
destroying legitimate dynamic template retrieval.

### Step 3: Allow template-driven early termination

When the architect sees a sufficiently strong published macro match, it should
be allowed to:

- stop further decomposition
- adopt the published structure directly
- attach provenance showing reuse rather than invention

This is the graph-level counterpart to Phase 4.

### Step 4: Add refinement-time macro substitution

Before recursively splitting a failed node, the orchestrator should check for a
published macro artifact that satisfies that subgoal's contract.

If found with sufficient confidence, substitute the macro artifact instead of
continuing to refine into smaller leaves.

### Step 5: Preserve leaf grounding boundaries

Published CDG reuse should change graph selection, not declaration verification.
Leaf Hunter logic should continue to own leaf-level candidate verification.

## Testing Plan

Add or extend tests for:

- template retrieval returning published CDGs alongside prior traces
- published CDGs winning when graph and contract similarity are strong
- architect early termination on strong published macro reuse
- refinement-time substitution avoiding unnecessary recursive expansion
- telemetry proving whether reuse came from a published macro or a historical
  trace

## Worker Breakdown

Recommended ownership:

- one worker owns architect/template/refinement reuse end to end

Not recommended:

- splitting `template_retriever`, `graph_retrieval`, and `orchestrator.py`
  across workers before the shared retrieval semantics are stable

## Dependencies

- requires Phase 1
- requires Phase 3
- can run in parallel with Phase 4 once shared macro-score semantics are fixed

## Parallelization Notes

- safe to parallelize with Phase 4 because this phase owns structured-path
  reuse, not planner entry-point choice
- one final integration pass should align threshold names, telemetry labels, and
  provenance fields

## Risks

- published CDGs can over-dominate retrieval if graph-similarity boosts are too
  aggressive
- refinement substitution can hide useful decomposition if confidence is not
  constrained
- mixing transient traces and published artifacts without provenance will make
  debugging difficult

## Exit Criteria

- published CDGs participate in template retrieval
- strong published matches can terminate decomposition early
- refinement checks published macro substitutions before recursive splitting
- provenance clearly distinguishes published reuse from trace reuse
