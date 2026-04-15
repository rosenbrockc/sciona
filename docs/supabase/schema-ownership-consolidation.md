# Supabase Schema Ownership Consolidation

This document defines the planned ownership split for the local and deployed
Supabase schema across `sciona-matcher`, `sciona-infra`, and the sibling atom
provider repos.

## Decision

The target end state is:

- `../sciona-infra` is the single authority for Supabase schema:
  - `supabase/config.toml`
  - `supabase/migrations/*`
  - `supabase/seed.sql`
  - `supabase/sql/*`
  - schema validation and reset operator flow
- `../sciona-atoms` owns provider-derived Supabase population tooling:
  - sibling provider discovery
  - core catalog seeding
  - file-backed backfills
- `sciona-matcher` becomes a consumer of the schema and populated data:
  - manifest export
  - snapshot readers
  - runtime/catalog consumers
  - tests that point at the infra-owned local Supabase stack

## Why

The API and frontend already depend on DB-side joins, views, and RPCs rather
than only raw table reads.

Examples in the current codebase:

- catalog search uses `search_atoms_hybrid` and `catalog_atoms_served`
- atom detail uses `get_atom_document`
- dashboard endpoints use `get_originator_impact`, `get_atom_benchmarks`,
  `originator_impact`, and `atom_authors`
- `../sciona-infra/frontend` consumes those dashboard/catalog endpoints

That means the schema cannot be split informally between repos without local
drift and reset ambiguity.

## Current Problem

Today the migration trees diverge:

- matcher stops at `20260402000000_enterprise_auth.sql`
- infra also has:
  - `20260405000000_badges_and_referrals.sql`
  - `20260407000000_architect_leaderboard.sql`
  - `20260407100000_reputation_system.sql`

As a result:

- a reset from matcher drops infra-only objects
- a local DB can be ahead of matcher git without that being visible in matcher
- operator runbooks are not reproducible from one repository alone

## Ownership Rules

Once this plan is in effect:

- new schema migrations are added only in `../sciona-infra`
- matcher does not add new migrations locally
- atom/provider repos do not own schema, only data-population logic
- local reset/reseed starts from the infra migration tree, then runs
  provider-owned population tooling

## Staged Cutover

### Phase 1: Freeze Matcher Schema Ownership

- Treat matcher's `supabase/migrations` as legacy compatibility copies.
- Do not add any new migrations there.
- Add explicit docs in both repos that infra is the target authority.

Status:

- documentation phase can happen immediately

### Phase 2: Move Local Operator Flow To Infra

Change local reset/reseed guidance so the authoritative flow becomes:

1. run `supabase db reset --local` from `../sciona-infra`
2. run infra seed/validation SQL from `../sciona-infra`
3. run provider-owned seeding/backfills from `../sciona-atoms`
4. run matcher manifest/export consumers against that DB

Practical implication:

- matcher docs should stop presenting its own `supabase/` tree as the
  authoritative reset root
- matcher runbooks should point to infra-owned reset commands

### Phase 3: Repoint Matcher Tests And Tooling

Matcher still has local assumptions tied to `repo_root/supabase`.

Current blockers include:

- `tests/conftest.py` reading `repo_root / "supabase" / "config.toml"`
- local Supabase tests starting/stopping/resetting the stack from matcher cwd
- scripts like `scripts/validate_supabase_phase0.sh` reading matcher-local
  `supabase/seed.sql` and `supabase/sql/*`

Required changes:

- parameterize matcher local-Supabase tests so they can target the infra repo
- add an env var or fixture override for the Supabase project root
- update validation scripts to call infra paths or move those scripts into infra

Current matcher seam:

- `SCIONA_SUPABASE_PROJECT_ROOT=/Users/conrad/personal/sciona-infra`

Matcher should not delete its local `supabase/` tree before this phase is done.

### Phase 4: Keep Population Ownership In Provider Repos

After the schema root moves to infra:

- keep provider discovery and core/file-backed backfills in `../sciona-atoms`
- do not move that logic into infra unless infra becomes the general data
  ingestion orchestrator

The intended flow is:

- infra owns schema
- atoms owns population
- matcher owns manifest/export consumption

### Phase 5: CI And Developer Workflow Cutover

Update CI and local commands so:

- schema validation runs from infra
- provider seeding runs from atoms
- matcher tests against the infra-owned local stack

Good target command layering:

1. `../sciona-infra`: reset / migrate / validate schema
2. `../sciona-atoms`: seed providers and backfill corpus metadata
3. `sciona-matcher`: export manifest, run consumer/runtime tests

### Phase 6: Remove Matcher Migration Duplication

Only after Phases 2-5 are complete:

- remove matcher's duplicated migration files, or
- replace them with a minimal compatibility stub plus clear pointer to infra

Do not remove matcher's `supabase/` tree until all local tests, scripts, and CI
jobs have stopped assuming it is the Supabase project root.

## Recommended Immediate Next Steps

1. Declare `../sciona-infra` the authoritative migration owner in docs now.
2. Add a matcher test/config seam for `SCIONA_SUPABASE_PROJECT_ROOT` or similar.
3. Update matcher local Supabase tests to use infra's `supabase/config.toml`.
4. Move schema validation commands and runbooks to infra-owned paths.
5. Only then retire matcher's migration copy.

## Non-Goals

This plan does not require creating a fourth dedicated schema repo now.

A separate schema-only repo is possible later, but it would add another release
and coordination boundary immediately. The simpler near-term fix is to
consolidate schema ownership into `../sciona-infra`, which already contains the
superset migration tree and the API/frontend surfaces that depend on it.
