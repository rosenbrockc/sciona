# Unified Artifact Phase 6: API, Catalog, And Cutover

## Status

Drafted on April 14, 2026 as Phase 6 of
[Unified Artifact Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_IMPLEMENTATION_PLAN.md).

## Goal

Expose the unified artifact model through stable API and catalog surfaces, then
finish the runtime cutover without breaking atom compatibility.

This phase turns the internal artifact work into a supported platform contract.

## Problem

Even if the schema and runtime reuse work land, the platform still remains
atom-only from the perspective of API consumers and document-serving routes
until the catalog and RPC layer are updated.

## Scope

This phase covers:

- artifact-facing served views and RPC adoption in matcher
- API endpoints for searching and reading artifacts
- compatibility preservation for existing atom endpoints
- runtime/caller cutover from atom-only assumptions to artifact-aware ones
- operator validation, docs, and cleanup of temporary compatibility seams

## Non-Goals

This phase does not:

- redesign the whole frontend
- remove Memgraph
- remove the atom compatibility layer immediately
- invent new publishability semantics

## Files In Scope

Primary matcher API/runtime files:

- [sciona/api/routers/catalog.py](/Users/conrad/personal/sciona-matcher/sciona/api/routers/catalog.py)
- [sciona/api/deps.py](/Users/conrad/personal/sciona-matcher/sciona/api/deps.py)
- [sciona/commands/catalog_cmds.py](/Users/conrad/personal/sciona-matcher/sciona/commands/catalog_cmds.py)
- [sciona/api/snapshot.py](/Users/conrad/personal/sciona-matcher/sciona/api/snapshot.py)

Primary schema files:

- [supabase/migrations](</Users/conrad/personal/sciona-infra/supabase/migrations>)

Primary tests:

- catalog, snapshot, and local integration tests under
  [tests](/Users/conrad/personal/sciona-matcher/tests)

## Implementation Steps

### Step 1: Adopt artifact-facing SQL surfaces

Move matcher reads toward:

- `catalog_artifacts_served`
- `get_artifact_document(...)`

while preserving:

- `catalog_atoms_served`
- `get_atom_document(...)`

for compatibility.

### Step 2: Add artifact-aware API routes

Introduce routes such as:

- `/catalog/search-artifacts`
- `/catalog/artifact/{fqdn}`

Keep the current atom routes available, either as wrappers or filtered
artifact-kind views.

### Step 3: Keep manifest and local cache compatibility explicit

Decide whether the local SQLite manifest remains atom-only for now or becomes
artifact-aware. If it remains atom-only, document that choice explicitly and do
not let ad hoc partial artifact rows leak into it.

### Step 4: Add operator validation and telemetry

Add checks proving:

- artifact and atom endpoints return consistent atom results
- CDG artifacts are served with the expected metadata bundle
- visibility tier and publishability rules are enforced for both artifact kinds

### Step 5: Retire temporary compatibility seams selectively

Only after the artifact-facing APIs and runtime paths are stable should this
phase remove or simplify temporary dual-read/dual-write helpers introduced in
earlier phases.

## Testing Plan

Add or extend tests for:

- artifact search and document routes
- atom compatibility routes after artifact cutover
- snapshot/catalog behavior under mixed `atom` and `cdg` artifact data
- local Supabase integration for artifact-facing views and RPCs
- regression tests proving current atom consumers still work

## Worker Breakdown

Recommended ownership:

- one worker owns API/catalog cutover plus integration tests

Not recommended:

- splitting router updates, snapshot behavior, and compatibility cleanup across
  unrelated workers

## Dependencies

- requires Phase 1
- should follow stable progress in Phases 4 and 5

## Parallelization Notes

- best treated as the final phase
- avoid parallel edits to catalog API surfaces while runtime semantics are still
  changing underneath them

## Risks

- premature API cutover can expose incomplete artifact semantics externally
- mixed artifact and atom document logic can create subtle compatibility drift
- manifest readers may be destabilized if artifact support is introduced
  implicitly rather than as an explicit contract

## Exit Criteria

- artifact-facing catalog/document APIs exist and pass integration tests
- atom-facing APIs still work through compatibility surfaces
- runtime callers use the unified artifact model where appropriate
- temporary migration seams are reduced to the minimum needed for compatibility
