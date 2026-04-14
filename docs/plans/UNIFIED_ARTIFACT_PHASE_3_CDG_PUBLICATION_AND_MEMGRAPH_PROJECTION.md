# Unified Artifact Phase 3: CDG Publication And Memgraph Projection

## Status

Drafted on April 14, 2026 as Phase 3 of
[Unified Artifact Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_IMPLEMENTATION_PLAN.md).

## Goal

Make published CDGs first-class persisted artifacts and project them into
Memgraph as a derived graph retrieval index.

This phase is where composite algorithms stop being transient workflow outputs
and become reusable published catalog objects.

## Problem

The current system has graph retrieval infrastructure, but it does not have a
canonical catalog model for published CDGs. That means graph reuse is based on
historical traces and in-memory workflow artifacts rather than on durable
published macro solutions.

## Scope

This phase covers:

- deterministic persistence of published CDGs into `artifacts` and
  `artifact_versions`
- storage of node, edge, and binding structure in relational tables
- top-level contract derivation and graph summary fields
- one-way projection from canonical artifact versions into Memgraph
- rebuild and incremental-sync mechanics for the Memgraph projection

## Non-Goals

This phase does not:

- change planner direct retrieval order
- make the architect consume published CDGs yet
- expose public artifact catalog APIs
- replace Memgraph with relational queries

## Files In Scope

Primary matcher graph/runtime surfaces:

- [sciona/graph_store.py](/Users/conrad/personal/sciona-matcher/sciona/graph_store.py)
- [sciona/architect/graph_retrieval.py](/Users/conrad/personal/sciona-matcher/sciona/architect/graph_retrieval.py)
- [sciona/architect/template_retriever.py](/Users/conrad/personal/sciona-matcher/sciona/architect/template_retriever.py)

Primary publication surfaces:

- [sciona/api/routers/registry.py](/Users/conrad/personal/sciona-matcher/sciona/api/routers/registry.py)
- provider-owned publication helpers in
  [../sciona-atoms](</Users/conrad/personal/sciona-atoms>)

Primary tests:

- graph retrieval tests under [tests](/Users/conrad/personal/sciona-matcher/tests)

## Implementation Steps

### Step 1: Define a published CDG artifact contract

Specify the minimum persisted data for a published CDG version:

- top-level input/output contract
- node and edge rows
- binding rows to leaf artifacts where known
- `topo_hash`
- graph summary counts
- verification and coverage summary fields

This contract should be versioned and deterministic from the CDG content.

### Step 2: Add CDG publication/backfill logic

Implement a deterministic path that can:

- publish a CDG as an `artifact` with `artifact_kind = 'cdg'`
- write `artifact_version`
- persist nodes, edges, and bindings
- derive `content_hash` and `topo_hash`

The same code should support both first publication and repeat upsert of the
same content.

### Step 3: Derive top-level contract and graph summaries

Populate summary fields on `artifacts` or `artifact_versions` for:

- top-level input/output arity
- leaf count
- verified leaf coverage
- modality/domain hints if available

These fields are for quick filtering and ranking before graph-level retrieval.

### Step 4: Project published CDGs into Memgraph

Extend the graph-store path so published CDGs become a derived Memgraph index.
The projection should be rebuildable from Supabase alone.

Required behaviors:

- full rebuild from canonical artifact tables
- incremental upsert for new or changed CDG versions
- deterministic node labels and edge identity
- ability to distinguish published artifacts from transient workflow traces

### Step 5: Add sync/repair tooling

Provide operator-facing commands or scripts for:

- full Memgraph rebuild
- projection drift validation
- incremental sync

This phase should not assume Memgraph is always perfectly fresh.

## Testing Plan

Add or extend tests for:

- deterministic `content_hash` and `topo_hash`
- stable node/edge persistence from the same CDG input
- projection rebuild from canonical relational tables
- incremental sync of changed CDG versions
- clear separation between published CDG artifacts and transient execution
  traces in retrieval queries

## Worker Breakdown

Recommended ownership:

- one worker owns CDG persistence plus Memgraph projection end to end

Not recommended:

- splitting relational CDG persistence and Memgraph projection into unrelated
  workers before the canonical contract is stable

## Dependencies

- requires Phase 1
- can run in parallel with Phase 2

## Parallelization Notes

- safe to run in parallel with Phase 2 because the write sets are largely
  disjoint
- avoid concurrent edits to common publication helpers or shared scoring labels

## Risks

- top-level contract derivation can drift from actual graph semantics if not
  defined rigorously
- Memgraph projection may silently diverge from Supabase if incremental sync is
  not validated
- transient execution traces can pollute retrieval if they are not explicitly
  separated from published artifacts

## Exit Criteria

- published CDGs can be persisted as canonical artifact versions
- node, edge, and binding rows exist and are deterministic
- Memgraph can be rebuilt from Supabase-persisted published CDGs
- projection drift can be detected and repaired
