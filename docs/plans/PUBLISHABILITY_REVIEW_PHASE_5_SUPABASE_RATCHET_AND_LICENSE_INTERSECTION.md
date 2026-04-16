# Publishability Review Phase 5: Supabase Ratchet And License Intersection

## Goal

Replay the improved review bundles into Supabase and measure the real public
publication surface after audit and license filtering intersect.

## Tasks

1. Reset or replay the local Supabase baseline.
2. Re-seed atoms and versions.
3. Re-run file-backed backfills.
4. Re-apply license metadata.
5. Recompute publishability under the strict gate.
6. Export public and developer manifests.
7. Measure:
   - publishable atoms
   - license-approved atoms
   - public manifest atoms
   - benchmark coverage within the newly public slice

## Ratchet Rule

Do not lower the public bar to make counts go up.

Counts may increase only because:

- metadata became complete
- reviews became approved
- provenance/license classification became strong enough

## Parallelization

Single integrator only. This phase owns the shared DB and manifest outputs.

## Exit Criteria

- the new baseline is reproducible
- improvements are attributable to real review artifacts
- remaining unpublished atoms are reclassified from the new replay
