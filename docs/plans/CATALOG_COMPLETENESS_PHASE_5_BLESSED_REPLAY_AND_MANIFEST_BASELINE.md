# Catalog Completeness Phase 5: Blessed Replay And Manifest Baseline

## Goal

Produce the first documented, trustworthy local Supabase replay and SQLite
manifest baseline after the completeness phases land.

## Why This Phase Matters

The manifest export path already works. What is missing is one clean baseline
run after the completeness work has stabilized, with explicit counts and known
validation checks.

## Scope

This phase covers:

- local infra-root Supabase reset
- provider-owned seed replay
- provider-owned file-backed backfill replay
- artifact sync replay
- benchmark replay
- tiered manifest export
- count capture and integrity validation

## Primary Repos

- `../sciona-infra`
- `../sciona-atoms`
- `sciona-matcher`

## Worker Ownership

One integrator only.

This phase should not be split across multiple workers because it is the final
baseline assembly and verification pass.

## Tasks

1. Reset local Supabase from infra.
   - Use the infra-owned migration tree only.

2. Replay provider seed and file-backed backfills.
   - Capture all summary counts.
   - Compare against the prior baseline.

3. Replay artifact sync and benchmark sync.
   - Ensure both atoms and artifacts are represented as intended.

4. Export tiered SQLite manifests.
   - Verify each tier is written successfully.
   - Verify exporter integrity guards still pass.

5. Record the baseline.
   - Document final local counts.
   - Document remaining known skip classes, if any.
   - Document benchmark/artifact coverage levels.

## Validation

- direct DB count query
- direct SQLite queries against each exported tier manifest
- fail-closed exporter guard remains green
- focused tests for any paths changed during the final integration pass

## Exit Criteria

- one clean replay is documented end to end
- the exported manifest tiers are consistent with the recorded Supabase counts
- remaining gaps are narrow, enumerated, and no longer architectural
