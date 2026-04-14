# Unified Artifact Phase 2: Atom Compatibility And Population

## Status

Drafted on April 14, 2026 as Phase 2 of
[Unified Artifact Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_IMPLEMENTATION_PLAN.md).

## Goal

Populate the new artifact model from the existing atom catalog and keep future
atom writes compatible while the runtime remains atom-centric.

This phase makes the new schema real for the already-published atom corpus.

## Problem

Phase 1 only creates empty artifact tables. Until atoms are migrated and new
atom writes maintain artifact rows, the runtime cannot safely depend on the new
model.

## Scope

This phase covers:

- backfilling existing atom rows into `artifacts` and `artifact_versions`
- populating shared `artifact_*` metadata from the current atom tables
- keeping publish/update flows in sync for future atom writes
- adding compatibility reporting for drift between atom and artifact records

## Non-Goals

This phase does not:

- publish CDGs as first-class artifacts
- change planner retrieval order
- change architect/template reuse behavior
- expose artifact-first public APIs
- remove the legacy atom tables

## Files In Scope

Primary matcher surfaces:

- [sciona/api/routers/registry.py](/Users/conrad/personal/sciona-matcher/sciona/api/routers/registry.py)
- [sciona/api/snapshot.py](/Users/conrad/personal/sciona-matcher/sciona/api/snapshot.py)
- [tests/test_supabase_snapshot.py](/Users/conrad/personal/sciona-matcher/tests/test_supabase_snapshot.py)

Primary provider-owned population surfaces:

- [supabase_seed.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/supabase_seed.py>)
- [supabase_backfill.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/supabase_backfill.py>)

Primary schema validation surfaces:

- [tests/test_supabase_local_integration.py](/Users/conrad/personal/sciona-matcher/tests/test_supabase_local_integration.py)

## Implementation Steps

### Step 1: Backfill existing atoms into artifacts

Create a deterministic migration/backfill path that:

- creates one `artifact` per current atom
- creates one `artifact_version` per current latest atom version
- preserves `content_hash`, semver, source metadata, and publishability state

The backfill must be rerunnable and order-independent.

### Step 2: Populate shared metadata tables

Backfill the artifact-parallel metadata tables from the current atom-backed
tables:

- IO specs
- parameters
- descriptions
- references
- audit rollups and evidence
- uncertainty
- verification matches
- hyperparams
- benchmarks where present

Prefer SQL- or script-driven deterministic transforms over ad hoc application
code.

### Step 3: Add ongoing atom dual-write or artifact-first write logic

Future atom publication and update paths must keep artifact tables in sync.

Acceptable implementations:

- artifact-first write with compatibility writes to atom tables
- existing atom write plus deterministic artifact mirror write

Unacceptable implementation:

- "run a separate repair job later" as the normal publish path

### Step 4: Add drift detection

Add validation that can detect:

- atom rows without artifact rows
- mismatched publishability or visibility state
- mismatched latest-version linkage
- metadata present on one side but missing on the other

This should run in local validation and CI-oriented checks even if Phase 6 has
not landed yet.

### Step 5: Keep manifest and document readers stable

Any matcher paths that still expect atom-only rows should continue to function.
This phase should not force the runtime to read artifacts yet.

## Testing Plan

Add or extend tests for:

- deterministic atom-to-artifact backfill
- rerunnable population without duplicates
- publish/update paths that keep atom and artifact rows aligned
- `get_atom_document(...)` and atom manifest generation after artifact
  backfill
- validation failures on intentionally drifted fixtures

## Worker Breakdown

Recommended ownership:

- one worker owns matcher publish-path changes
- the same worker may also own the provider-owned seeding/backfill changes in
  `../sciona-atoms`

Not recommended:

- splitting matcher publish logic and provider seeding into separate workers
  unless write ownership is extremely clean

## Dependencies

- requires Phase 1
- can run in parallel with Phase 3

## Parallelization Notes

- safe to parallelize against Phase 3 because this phase owns atom migration and
  compatibility, not CDG publication
- do not edit shared artifact publishability SQL in parallel with Phase 3

## Risks

- partial dual-write can create silent drift if not verified aggressively
- backfilling publishability too early can freeze stale quality state
- artifact-first writes may expose missing fields that the current atom publish
  path never populated

## Exit Criteria

- the current published atom corpus exists in `artifacts` and
  `artifact_versions`
- shared artifact metadata is populated for atoms
- normal atom writes keep artifact rows aligned
- drift detection exists and passes on local integration fixtures
