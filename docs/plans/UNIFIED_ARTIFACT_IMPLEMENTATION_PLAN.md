# Unified Artifact Implementation Plan

## Status

Drafted on April 14, 2026 as the worker-facing execution companion to
[UNIFIED_ARTIFACT_MODEL_AND_MACRO_RETRIEVAL_PLAN.md](/Users/conrad/personal/sciona-matcher/docs/UNIFIED_ARTIFACT_MODEL_AND_MACRO_RETRIEVAL_PLAN.md).

## Purpose

Turn the hybrid artifact design into an implementation sequence that coding
agents can execute without stepping on the same integration seams.

The target state is:

- Supabase is the canonical store for both published atoms and published CDGs
- Memgraph remains the derived graph index for structural retrieval and reuse
- the planner can choose a published macro artifact before leaf grounding
- the architect can reuse published CDGs as templates instead of only
  rediscovering them
- existing atom-facing APIs and runtime paths keep working during migration

## Current Repo Reality

The current runtime is split across two retrieval levels:

- atom retrieval is declaration-oriented through
  [sciona/services/planner_service.py](/Users/conrad/personal/sciona-matcher/sciona/services/planner_service.py),
  [sciona/services/hunter_service.py](/Users/conrad/personal/sciona-matcher/sciona/services/hunter_service.py),
  and [sciona/hunter/nodes.py](/Users/conrad/personal/sciona-matcher/sciona/hunter/nodes.py)
- graph reuse already exists through Memgraph-oriented code in
  [sciona/architect/template_retriever.py](/Users/conrad/personal/sciona-matcher/sciona/architect/template_retriever.py),
  [sciona/architect/graph_retrieval.py](/Users/conrad/personal/sciona-matcher/sciona/architect/graph_retrieval.py),
  and [sciona/graph_store.py](/Users/conrad/personal/sciona-matcher/sciona/graph_store.py)
- the persistent catalog and document model are still atom-only in the current
  Supabase schema and in [catalog.py](/Users/conrad/personal/sciona-matcher/sciona/api/routers/catalog.py)

There is real reuse value already present in the codebase, but the storage
model, publishability model, and runtime entry points are not yet aligned.

## Schema Ownership

All Supabase migration work for this plan should be treated as
`sciona-infra`-owned schema work.

Primary schema files therefore land in:

- [supabase/migrations](</Users/conrad/personal/sciona-infra/supabase/migrations>)

Matcher remains the runtime consumer and should only carry the code, tests, and
docs needed to use the schema.

## Phase Set

1. [UNIFIED_ARTIFACT_PHASE_1_SCHEMA_AND_COMPATIBILITY.md](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_PHASE_1_SCHEMA_AND_COMPATIBILITY.md)
2. [UNIFIED_ARTIFACT_PHASE_2_ATOM_COMPATIBILITY_AND_POPULATION.md](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_PHASE_2_ATOM_COMPATIBILITY_AND_POPULATION.md)
3. [UNIFIED_ARTIFACT_PHASE_3_CDG_PUBLICATION_AND_MEMGRAPH_PROJECTION.md](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_PHASE_3_CDG_PUBLICATION_AND_MEMGRAPH_PROJECTION.md)
4. [UNIFIED_ARTIFACT_PHASE_4_DIRECT_MACRO_RETRIEVAL.md](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_PHASE_4_DIRECT_MACRO_RETRIEVAL.md)
5. [UNIFIED_ARTIFACT_PHASE_5_TEMPLATE_AND_REFINEMENT_REUSE.md](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_PHASE_5_TEMPLATE_AND_REFINEMENT_REUSE.md)
6. [UNIFIED_ARTIFACT_PHASE_6_API_CATALOG_AND_CUTOVER.md](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_PHASE_6_API_CATALOG_AND_CUTOVER.md)

## Dependency Structure

### Hard dependencies

- Phase 1 must land first. It defines the canonical relational contract.
- Phase 4 depends on Phase 1 plus a minimally working artifact population path
  from Phases 2 and 3.
- Phase 5 depends on Phase 1 and Phase 3 because it needs published CDGs in the
  Memgraph projection path.
- Phase 6 depends on Phase 1 and should wait until the runtime contracts in
  Phases 4 and 5 are stable enough to expose publicly.

### Soft dependencies

- Phase 2 and Phase 3 can proceed in parallel after Phase 1.
- Phase 4 and Phase 5 can proceed in parallel once artifact and CDG rows exist,
  but they should not share ownership of retrieval-threshold wiring or common
  telemetry helpers in the same change wave.

## Parallelization Analysis

The best execution model is dependency waves plus strict file ownership.

### Wave 0: Schema foundation

Contains:

- Phase 1

Reason:

- It defines the artifact contract that every later phase consumes.
- It touches the infra migration tree and compatibility SQL, which should not
  be edited concurrently.

Recommendation:

- Use one worker only.

### Wave 1: Canonical population branches

Contains:

- Phase 2 core work
- Phase 3 core work

Reason:

- atom back-compat population and CDG publication are both blocked on the Phase
  1 schema, but they are otherwise mostly disjoint.

Caveats:

- both phases rely on the same `artifact_is_publishable()` contract, so that
  contract should stay Phase 1-owned
- Phase 2 should own atom dual-write and atom migration logic
- Phase 3 should own CDG ingestion, node/edge persistence, and Memgraph sync

Recommended split:

- Worker 1: Phase 2 in matcher publish paths plus provider-owned seeding
  changes in `../sciona-atoms`
- Worker 2: Phase 3 in CDG publication, graph projection, and Memgraph tests

### Wave 2: Runtime reuse branches

Contains:

- Phase 4 core work
- Phase 5 core work

Reason:

- direct macro retrieval and template/refinement reuse are distinct runtime
  entry points
- one is planner/runtime-path centric
- the other is architect/orchestrator/Memgraph centric

Caveats:

- do not let both workers independently invent different macro-score semantics
- keep one owner for config thresholds and shared retrieval labels

Recommended split:

- Worker 3: Phase 4 in `planner_service.py`, `runtime_paths.py`, and
  planner-facing tests
- Worker 4: Phase 5 in `architect/`, `orchestrator.py`, `graph_store.py`, and
  template/refinement tests

### Wave 3: Public serving and cleanup

Contains:

- Phase 6

Reason:

- API and catalog cutover should be last once internal contracts stabilize

Recommendation:

- Use one worker or one final integrator.

## Shared Hotspots

The following files or areas should not be owned by multiple workers in the
same wave:

- [../sciona-infra/supabase/migrations](</Users/conrad/personal/sciona-infra/supabase/migrations>)
- [sciona/services/planner_service.py](/Users/conrad/personal/sciona-matcher/sciona/services/planner_service.py)
- [sciona/runtime_paths.py](/Users/conrad/personal/sciona-matcher/sciona/runtime_paths.py)
- [sciona/orchestrator.py](/Users/conrad/personal/sciona-matcher/sciona/orchestrator.py)
- [sciona/architect/template_retriever.py](/Users/conrad/personal/sciona-matcher/sciona/architect/template_retriever.py)
- [sciona/architect/graph_retrieval.py](/Users/conrad/personal/sciona-matcher/sciona/architect/graph_retrieval.py)
- [sciona/api/routers/catalog.py](/Users/conrad/personal/sciona-matcher/sciona/api/routers/catalog.py)

## Recommended Delivery Order

1. Phase 1
2. Phases 2 and 3 in parallel
3. Short integration pass for shared artifact score semantics
4. Phases 4 and 5 in parallel
5. Runtime integration pass for shared macro-selection telemetry
6. Phase 6

## Exit Criteria For The Whole Plan

The broader artifact effort should be considered implemented when all of the
following are true:

- published atoms and published CDGs both exist in the canonical artifact model
- Memgraph can be rebuilt from persisted CDG artifact versions
- the planner can choose a macro artifact before falling back to leaf grounding
- the architect can reuse published CDGs during decomposition and refinement
- catalog/document APIs can serve both artifact kinds without breaking atom
  compatibility
- publishability, descriptions, uncertainty, and audit data follow one shared
  artifact contract rather than atom-only special cases
