# Local Supabase Reset And Reseed

> Current canonical replay instructions live in
> `/Users/conrad/personal/sciona-atoms/docs/SUPABASE_REPLAY.md`.
> This matcher-local document is historical and may reference older per-table
> backfill scripts.

This runbook resets the local Supabase instance, reapplies the checked-in schema,
and repopulates it from the current sibling atom-provider repos before any
SQLite manifest build.

This document describes the current matcher-local flow. The planned ownership
cutover to `../sciona-infra` is tracked in
`docs/supabase/schema-ownership-consolidation.md`.

## Why This Exists

The local DB can drift in two ways:

- schema drift: local Supabase may have migrations applied from sibling repos
  such as `../sciona-infra/supabase/migrations/20260405000000_badges_and_referrals.sql`
- data drift: the DB may contain runtime/test fixtures instead of the real atom
  corpus

At the time of writing, the local DB is in both states:

- schema includes `20260405000000_badges_and_referrals.sql`
- `public.atoms` contains only `pkg.*` fixture rows, not `sciona.atoms.*`
  production atoms

Do not build `manifest.sqlite` from that state.

## Preconditions

- local Supabase stack is running
- Supabase CLI is installed locally
- the repo venv is available at `/Users/conrad/personal/sciona-matcher/.venv`
- sibling provider repos exist beside this repo:
  - `../sciona-atoms`
  - `../sciona-atoms-signal`
  - `../sciona-atoms-ml`
  - `../sciona-atoms-bio`
  - `../sciona-atoms-fintech`
  - `../sciona-atoms-physics`
  - `../sciona-atoms-robotics`
  - `../sciona-atoms-cs`
- schema owner repo exists beside this repo:
  - `../sciona-infra`

## Step 0: Reconcile The Schema Baseline

Before any reset, make the migration source of truth explicit.

Current state:

- this repo vendors migrations through `20260402000000_enterprise_auth.sql`
- `../sciona-infra` also contains:
  - `20260405000000_badges_and_referrals.sql`
  - `20260407000000_architect_leaderboard.sql`
  - `20260407100000_reputation_system.sql`

Recommended rule:

- if this local Supabase instance is meant to support the full app stack, vendor
  the missing migrations into this repo before reset
- if this instance is only for manifest/catalog work, decide explicitly whether
  badges/referrals/reputation are in or out of scope, then keep one migration
  tree authoritative

Do not run a destructive reset while the migration tree is ambiguous.

## Step 1: Reset Local Supabase

Once the migration baseline is settled:

```bash
cd /Users/conrad/personal/sciona-matcher
supabase db reset --local
```

If the migration baseline intentionally includes files from `../sciona-infra`,
bring them into the active local migration tree first, then rerun the reset.

## Step 2: Seed Baseline SQL

If the reset path does not already run the local seed:

```bash
cd /Users/conrad/personal/sciona-matcher
psql postgresql://postgres:postgres@127.0.0.1:54322/postgres \
  -v ON_ERROR_STOP=1 \
  -f supabase/seed.sql
```

Optional validation:

```bash
cd /Users/conrad/personal/sciona-matcher
psql postgresql://postgres:postgres@127.0.0.1:54322/postgres \
  -v ON_ERROR_STOP=1 \
  -f supabase/sql/phase0_validation.sql
```

## Step 3: Export Connection Environment

Use the local stack directly:

```bash
export SUPABASE_URL=http://127.0.0.1:54321
export SUPABASE_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres
export SUPABASE_SERVICE_ROLE_KEY=...
export SUPABASE_SERVICE_KEY="${SUPABASE_SERVICE_ROLE_KEY}"
```

The service-role key should come from the local Supabase status/config, not a
hosted project.

## Step 4: Populate Core Atom State

The canonical entrypoint now lives in `../sciona-atoms`.

The current core seeding surface is:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=../sciona-atoms/src \
  /Users/conrad/personal/sciona-matcher/.venv/bin/python \
  ../sciona-atoms/scripts/supabase_seed.py --json
```

To apply the base rows and bootstrap the deterministic local owner:

```bash
cd /Users/conrad/personal/sciona-matcher
PYTHONPATH=../sciona-atoms/src \
  /Users/conrad/personal/sciona-matcher/.venv/bin/python \
  ../sciona-atoms/scripts/supabase_seed.py \
  --apply \
  --ensure-owner \
  --database-url "${SUPABASE_DATABASE_URL}"
```

Current scope:

- federated providers under `../sciona-atoms*`
- active provider metadata owned by the `../sciona-atoms*` repos

Required outcome:

- `atom_source_repositories` reflects real sibling repos, not `runtime-repo-*`
- `atoms` contains real provider-backed FQDNs
- no `pkg.*` fixture rows remain

Current limitations:

- this seeder currently owns only `atom_source_repositories` and `atoms`
- it intentionally defers `atom_versions`, `hyperparams`, and
  `atom_benchmarks`
- duplicate FQDN collisions are surfaced in the dry-run summary and should be
  reviewed before applying

## Step 5: Run File-Backed Backfills

Run these only after the core `atoms` table is populated with the real corpus.

Wave A:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_io_specs.py
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_parameters.py
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_technical_descriptions.py
```

Wave B:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_references_registry.py
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_references.py
```

Wave C:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_audit_rollups.py
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_audit_evidence.py
```

Wave D:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_uncertainty.py
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_verification_matches.py
```

Then:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/backfill_dejargonized_descriptions.py --mode heuristic
```

Use `--mode heuristic` for a cheap local completeness pass. Use OpenAI-backed
generation only when you want the final text quality.

## Step 6: Recompute Dependent State

After the pillar tables are loaded:

- recompute `atoms.is_publishable`
- refresh any derived views/materialized data needed by search/runtime paths
- optionally seed contribution/entitlement state if the full app stack is in
  scope for this local environment

## Step 7: Embeddings

Only run embeddings after the real catalog exists:

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python scripts/generate_embeddings.py --backfill
```

This requires `OPENAI_API_KEY`.

## Step 8: Validate Before Manifest Export

Minimum validation queries:

```sql
select count(*) from public.atoms where fqdn like 'pkg.%';
select count(*) from public.atoms where fqdn like 'sciona.atoms.%';
select count(*) from public.atom_source_repositories where repo_name like 'runtime-repo-%';
select count(*) from public.atom_embeddings;
```

Expected shape:

- `pkg.%` count is `0`
- `runtime-repo-%` count is `0`
- at least one real provider namespace is present
- embedding coverage is consistent with the publishable atom set

## Long-Term Ownership

The target end state is not “these scripts stay in matcher forever.”

The target end state is:

- one canonical Supabase population/update toolchain lives in `../sciona-atoms`
- that toolchain reads the sibling provider inventory from a single registry
- matcher consumes Supabase/manifest outputs but does not remain the owner of
  the data-population pipeline

The current work is the bridge toward that state:

- remove repo-specific hard-coding
- make provider discovery come from `sources.yml` and shared helpers
- then move the stabilized script set into `../sciona-atoms`
