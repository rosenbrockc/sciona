# AGEOA Deprecation Implementation Plan

## Status

Drafted on April 14, 2026 as the worker-facing execution plan for removing
`ageoa` as a first-class namespace across the provider repos, Supabase, matcher,
and the manifest pipeline.

## Purpose

The intended end state is strict:

- every supported atom and CDG lives under `sciona`-owned provider repos
- local Supabase contains no `ageoa.*` catalog rows
- matcher has no runtime dependency on `ageoa` and no `ageoa` fallback path
- the SQLite manifest is generated only from canonical `sciona.*` identities
- `../ageo-atoms` is no longer treated as an active provider repo

This is a deprecation plan, not an alias plan. The goal is to migrate the
content, cut over the references, and then delete the legacy namespace from the
active system.

## Current Repo Reality

The local system is still mixed:

- local Supabase currently contains `1007` atom rows
- `506` are `ageoa.*`
- `501` are `sciona.*`
- provider discovery and seed logic in
  [provider_inventory.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/provider_inventory.py>)
  and [supabase_seed.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/supabase_seed.py>)
  still explicitly support `ageo-atoms`
- many provider-owned metadata files in `sciona-atoms*` still carry
  `ageoa...@ageoa/...` identities because they were migrated from the legacy
  repo but not fully rewritten yet
- matcher still contains real `ageoa` references in:
  - ghost simulation docs and tests
  - source-catalog tests
  - ingest prompts and historical plans
  - some helper scripts and fixtures

There are also still legacy-generated names like
`computekurtosissqi` in active provider source. Those are not stale database
rows; they are still real registered atoms and need a separate naming
normalization pass during migration.

## Scope

This plan covers all remaining `ageoa` references that matter for runtime,
catalog, and manifest correctness:

- provider source ownership
- provider metadata and references
- Supabase seeding and backfills
- matcher runtime/config/docs/tests
- manifest export inputs
- final enforcement so `ageoa` cannot re-enter the system

It does not require preserving backwards compatibility for the legacy namespace.

## Phase Set

1. [AGEOA_DEPRECATION_PHASE_1_CANONICAL_NAMESPACE_AND_POLICY.md](/Users/conrad/personal/sciona-matcher/docs/plans/AGEOA_DEPRECATION_PHASE_1_CANONICAL_NAMESPACE_AND_POLICY.md)
2. [AGEOA_DEPRECATION_PHASE_2_PROVIDER_CONTENT_MIGRATION.md](/Users/conrad/personal/sciona-matcher/docs/plans/AGEOA_DEPRECATION_PHASE_2_PROVIDER_CONTENT_MIGRATION.md)
3. [AGEOA_DEPRECATION_PHASE_3_METADATA_AND_SEED_CUTOVER.md](/Users/conrad/personal/sciona-matcher/docs/plans/AGEOA_DEPRECATION_PHASE_3_METADATA_AND_SEED_CUTOVER.md)
4. [AGEOA_DEPRECATION_PHASE_4_MATCHER_RUNTIME_AND_TEST_REMOVAL.md](/Users/conrad/personal/sciona-matcher/docs/plans/AGEOA_DEPRECATION_PHASE_4_MATCHER_RUNTIME_AND_TEST_REMOVAL.md)
5. [AGEOA_DEPRECATION_PHASE_5_SUPABASE_AND_MANIFEST_REGENERATION.md](/Users/conrad/personal/sciona-matcher/docs/plans/AGEOA_DEPRECATION_PHASE_5_SUPABASE_AND_MANIFEST_REGENERATION.md)
6. [AGEOA_DEPRECATION_PHASE_6_RETIREMENT_AND_ENFORCEMENT.md](/Users/conrad/personal/sciona-matcher/docs/plans/AGEOA_DEPRECATION_PHASE_6_RETIREMENT_AND_ENFORCEMENT.md)

## Dependency Structure

### Hard dependencies

- Phase 1 must land first. Every later change depends on one explicit rule for
  canonical ownership and naming.
- Phase 3 depends on Phase 2 because the seed/backfill cutover should only
  happen after provider content and metadata are migrated.
- Phase 5 depends on Phases 3 and 4 because the regenerated Supabase and
  manifest should already reflect the canonical provider set and matcher should
  no longer expect `ageoa`.
- Phase 6 is last by design.

### Soft dependencies

- Phase 2 and parts of Phase 4 can run in parallel once Phase 1 is settled.
- Within Phase 2, work can be split by provider repo family as long as file
  ownership is disjoint.
- Within Phase 4, docs/tests can move in parallel with runtime/config cleanup,
  but shared fixtures should stay with one owner.

## Parallelization Analysis

### Wave 0: Canonical policy

Contains:

- Phase 1

Reason:

- the migration fails if different workers make different assumptions about
  whether `ageoa` remains supported, whether aliases survive, or how names are
  normalized

Recommendation:

- one worker only

### Wave 1: Provider migration branches

Contains:

- Phase 2 repo-family migration work
- Phase 4 matcher-side docs/test prep that does not change runtime behavior yet

Reason:

- content migration is mostly provider-repo-local
- matcher documentation and fixture preparation can advance in parallel without
  changing the live runtime contract

Recommended split:

- Worker 1: `sciona-atoms` core and shared library families
- Worker 2: `sciona-atoms-signal`
- Worker 3: `sciona-atoms-bio`
- Worker 4: `sciona-atoms-fintech`
- Worker 5: `sciona-atoms-physics`
- Worker 6: matcher docs/test inventory cleanup

### Wave 2: Cutover branches

Contains:

- Phase 3
- Phase 4 runtime/config removal

Reason:

- once content is migrated, provider seeding and matcher runtime can both stop
  honoring `ageoa`

Caveat:

- both phases should agree on the same canonical namespace list and provider
  roots

Recommended split:

- Worker 7: provider seed/backfill cutover in `../sciona-atoms`
- Worker 8: matcher runtime/config/test cutover in `sciona-matcher`

### Wave 3: Regeneration and hard enforcement

Contains:

- Phase 5
- Phase 6

Reason:

- regeneration must happen after the cutover
- enforcement should only land after regenerated outputs are green

Recommendation:

- one final integrator or one final worker wave

## Shared Hotspots

These areas should not be owned by multiple workers in the same wave:

- [provider_inventory.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/provider_inventory.py>)
- [supabase_seed.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/supabase_seed.py>)
- [supabase_backfill.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/supabase_backfill.py>)
- [sources.yml](/Users/conrad/personal/sciona-matcher/sources.yml)
- [sciona/synthesizer/ghost_sim.py](/Users/conrad/personal/sciona-matcher/sciona/synthesizer/ghost_sim.py)
- matcher tests that still assert `ageoa`-specific behavior

## Recommended Delivery Order

1. Phase 1
2. Phase 2 provider family migrations and Phase 4 docs/test prep in parallel
3. Phase 3 and Phase 4 runtime/config removal in parallel
4. Phase 5 clean Supabase reset/reseed/rebackfill and manifest regeneration
5. Phase 6 enforcement and repo retirement

## Exit Criteria For The Whole Plan

The `ageoa` deprecation effort should be considered complete only when all of
the following are true:

- no supported provider repo emits canonical `ageoa.*` atoms or CDGs
- `../ageo-atoms` is not part of the active provider inventory
- local Supabase contains zero `ageoa.*` rows in `atoms`
- matcher has no `ageoa` runtime fallback and no required `ageoa` import path
- SQLite manifest export contains zero `ageoa.*` atoms or artifacts
- CI or validation tooling fails closed if new `ageoa` references are added
