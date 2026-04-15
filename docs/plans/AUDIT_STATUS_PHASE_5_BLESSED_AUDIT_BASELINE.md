# Audit Status Phase 5: Blessed Audit Baseline

## Goal

Record the first post-audit-improvement baseline where the public manifest count
and the remaining non-publishable inventory are both known and trusted.

## Why This Phase Matters

The end of this work is not merely “more audited atoms exist.” The end state is
one documented replay/export baseline that future audit promotion work can be
measured against.

## Scope

This phase covers:

- one clean infra-root reset
- one clean seed/backfill replay
- public and developer manifest export
- recorded publishable/non-publishable counts
- recorded remaining family buckets

## Primary Repos

- `../sciona-infra`
- `../sciona-atoms`
- `sciona-matcher`

## Worker Ownership

One integrator only.

## Tasks

1. Run the final clean replay from scratch.

2. Export both:
   - public tier manifests
   - developer manifest

3. Record the new baseline counts.
   - total atoms
   - public publishable atoms
   - developer manifest atoms
   - remaining non-publishable atoms by provider/family

4. Record the remaining true blockers.
   - families still awaiting audit bundles
   - families intentionally deferred
   - any remaining ingest defects

5. Treat that baseline as the new ratchet point.
   - later work should only move the public count upward by real audit
     completion.

## Validation

- direct DB queries for final counts
- direct SQLite queries against exported public and developer manifests
- no legacy namespace rows
- no unaudited atoms in public manifests

## Exit Criteria

- one clean audited baseline is documented
- the public manifest count is strictly audit-backed
- the remaining non-publishable inventory is narrow, classified, and assigned
  to known provider work
