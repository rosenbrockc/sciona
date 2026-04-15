# Catalog Completeness Phase 2: Backfill Resolution And Registration

## Goal

Reduce the remaining `skipped_no_atom` backfill gaps by making provider metadata
and the registered catalog agree on what is actually first-class.

## Why This Phase Matters

The current replay is clean, but it still leaves a large amount of provider
metadata unmatched. That means the catalog is incomplete in a very specific
way: the data exists, but the catalog cannot attach it to canonical atoms.

This phase is the highest-leverage completeness phase because it improves:

- IO spec coverage
- parameter coverage
- technical/dejargonized description coverage
- references coverage
- audit rollup/evidence coverage
- verification match coverage

## Scope

This phase covers two kinds of work:

- registering concrete reusable helpers that are currently missing from the
  catalog
- pruning or rewriting provider metadata that refers to abstract/generated/CDG
  node names that should not be catalog atoms

It does not cover macro artifact publication itself. That belongs to Phase 3.

## Primary Repos

- `../sciona-atoms`
- `../sciona-atoms-signal`
- `../sciona-atoms-bio`
- `../sciona-atoms-fintech`
- `../sciona-atoms-physics`
- `../sciona-atoms-robotics`

## Worker Ownership

Recommended split by provider repo:

- Worker A: `sciona-atoms` shared inference/state-estimation families
- Worker B: `sciona-atoms-signal`
- Worker C: `sciona-atoms-bio`
- Worker D: `sciona-atoms-fintech`
- Worker E: `sciona-atoms-physics`
- Worker F: `sciona-atoms-robotics`

Each worker owns only its repo’s:

- `references.json`
- `cdg.json` / `*_cdg.json`
- `uncertainty.json`
- `matches.json`
- newly registered atom modules and tests in that repo

## Tasks

1. Classify unmatched metadata names.
   - Split each remaining unmatched name into:
     - should become a registered atom
     - should be rewritten to an existing registered atom
     - should be dropped as a non-catalog abstract/generated node

2. Register reusable concrete helpers.
   - Convert declaration/probe-backed or helper-style functions into registered
     atoms when they make sense as reusable primitives.
   - Add import smoke tests or focused runtime tests per family.

3. Rewrite provider metadata to canonical symbols.
   - Fix `references.json`, CDG stage names, uncertainty files, and match files
     so they point at real registered atoms.
   - Remove metadata entries that are intentionally non-catalog abstractions.

4. Replay provider-owned backfill tests locally.
   - Validate path/namespace derivation.
   - Validate no legacy namespace regressions.
   - Validate dedupe/idempotence remains intact.

5. Measure skip-count improvement.
   - Re-run the clean local replay.
   - Capture before/after `skipped_no_atom` counts by backfill family.

## Validation

- provider repo import smoke tests
- `../sciona-atoms/tests/test_provider_inventory.py`
- `../sciona-atoms/tests/test_supabase_backfill.py`
- `../sciona-atoms/tests/test_supabase_seed.py`
- live replay count comparison against the previous baseline

## Exit Criteria

- the major backfill families show materially lower `skipped_no_atom` counts
- any remaining unmatched rows are explicitly classifiable as non-catalog
  metadata, not just unresolved drift
- newly registered helpers are seeded and versioned correctly
