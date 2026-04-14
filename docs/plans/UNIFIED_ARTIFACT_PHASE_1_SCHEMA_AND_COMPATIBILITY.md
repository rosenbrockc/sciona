# Unified Artifact Phase 1: Schema And Compatibility

## Status

Drafted on April 14, 2026 as Phase 1 of
[Unified Artifact Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_IMPLEMENTATION_PLAN.md).

## Goal

Create the additive schema needed for a first-class artifact model without
breaking the existing atom-facing runtime.

This phase establishes the canonical relational contract for both atoms and
CDGs and preserves compatibility long enough for a staged cutover.

## Why This Phase Comes First

Every later phase depends on a stable answer to three questions:

- where canonical artifact identity lives
- how versioned metadata is shared across atoms and CDGs
- how existing atom-only readers continue to function during migration

Without that contract, later workers will either invent incompatible write
paths or couple runtime changes to temporary data hacks.

## Schema Ownership

All migration work for this phase should land in `sciona-infra`, not matcher.

Primary schema location:

- [supabase/migrations](</Users/conrad/personal/sciona-infra/supabase/migrations>)

Matcher should only carry the consuming code and tests that validate the new
contract.

## Scope

Phase 1 should do all of the following:

1. Add `artifacts` and `artifact_versions`.
2. Add shared `artifact_*` metadata tables parallel to the current atom tables.
3. Add CDG structure tables:
   - `artifact_cdg_nodes`
   - `artifact_cdg_edges`
   - `artifact_cdg_bindings`
4. Add `artifact_is_publishable()` and artifact-facing served/document SQL.
5. Preserve existing atom-facing views, RPCs, and publishability behavior during
   the transition.

## Non-Goals

Phase 1 should not:

- migrate existing atom rows into the new tables
- ingest published CDGs
- change planner or architect runtime behavior
- expose new public API routes
- remove atom tables

Those belong to later phases.

## Files In Scope

Primary schema files:

- [supabase/migrations](</Users/conrad/personal/sciona-infra/supabase/migrations>)
- [supabase/sql](</Users/conrad/personal/sciona-infra/supabase/sql>)

Primary matcher validation surfaces:

- [tests/test_supabase_local_integration.py](/Users/conrad/personal/sciona-matcher/tests/test_supabase_local_integration.py)
- [scripts/validate_supabase_phase0.sh](/Users/conrad/personal/sciona-matcher/scripts/validate_supabase_phase0.sh)

## Implementation Steps

### Step 1: Define the core artifact tables

Add `artifacts` and `artifact_versions` with:

- `artifact_kind` constrained to `atom` or `cdg`
- unique `fqdn`
- unique `(artifact_id, semver)`
- unique `content_hash`
- room for top-level contract summary fields and graph summary fields

Keep UUID primary keys. Treat `content_hash` as immutable artifact identity and
`(fqdn, semver)` as the public version lookup key.

### Step 2: Define shared artifact metadata tables

Create the artifact-parallel tables for:

- IO specs
- parameters
- descriptions
- references
- audit rollups
- audit evidence
- uncertainty estimates
- verification matches
- hyperparams
- benchmarks

Use `artifact_id` and `version_id` as the relational anchor. Do not couple the
new tables to atom-only IDs.

### Step 3: Define CDG structure persistence

Add:

- `artifact_cdg_nodes`
- `artifact_cdg_edges`
- `artifact_cdg_bindings`

The node/edge rows are not a Memgraph replacement. They are the canonical,
versioned relational persistence layer that Memgraph will project from later.

### Step 4: Add artifact publishability and serving SQL

Implement:

- `artifact_is_publishable()`
- `catalog_artifacts_served`
- `get_artifact_document(request_fqdn TEXT)`

Keep the initial implementation additive and conservative. The atom-facing
surfaces should remain available.

### Step 5: Preserve atom compatibility

Do not break:

- `catalog_atoms_served`
- `get_atom_document(...)`
- current atom publishability logic used by existing API/runtime paths

The safest shape is:

- keep atom tables in place for now
- add artifact-facing SQL beside them
- leave cutover and dual-write to Phase 2

### Step 6: Add validation SQL and local integration coverage

Extend infra-owned validation SQL to prove:

- the artifact tables exist
- constraints and uniqueness rules are correct
- artifact-facing views and RPCs compile
- current atom-facing views and RPCs still work

## Testing Plan

Add or extend tests for:

- artifact table creation and constraints
- `artifact_is_publishable()` behavior on minimal valid/invalid fixtures
- `catalog_artifacts_served` and `get_artifact_document(...)`
- `catalog_atoms_served` and `get_atom_document(...)` compatibility after the
  migration
- matcher local Supabase integration against the infra-owned project root

## Worker Breakdown

Recommended ownership:

- one worker owns the infra migration, validation SQL, and matcher integration
  coverage

Not recommended:

- separate workers editing the same migration series
- concurrent edits to artifact and atom compatibility SQL

## Risks And Decisions

### Compatibility direction

Do not prematurely turn atom views into wrappers over incomplete artifact data.
Add the artifact surfaces first; move reads later.

### Metadata duplication

There will be temporary duplication between atom tables and artifact tables.
That is acceptable in this phase. Correctness and compatibility matter more
than immediate normalization.

### Publishability scope

`artifact_is_publishable()` should preserve the current atom quality bar and add
CDG-specific checks only where they are structurally necessary.

## Exit Criteria

Phase 1 is complete when:

- the artifact and CDG structure tables exist in infra-owned migrations
- artifact-facing views and RPCs compile and pass integration tests
- current atom-facing views and RPCs still pass
- later phases can write artifact rows without inventing ad hoc schema
