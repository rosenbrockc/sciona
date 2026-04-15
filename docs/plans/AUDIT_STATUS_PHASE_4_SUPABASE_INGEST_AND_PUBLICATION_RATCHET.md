# Audit Status Phase 4: Supabase Ingest And Publication Ratchet

## Goal

Replay the provider audit bundles into local Supabase and enforce the strict
publication ratchet so publishability rises only from real audit coverage.

## Why This Phase Matters

Provider artifacts do not matter until the seed/backfill path ingests them
cleanly and the resulting publishability increase is measurable.

## Scope

This phase covers:

- provider seed replay
- provider file-backed backfill replay
- publishability count changes
- strict public vs developer manifest behavior verification

This phase does not cover:

- new audit artifact authoring
- benchmark broadening beyond what the provider bundles already support

## Primary Repos

- `../sciona-infra`
- `../sciona-atoms`
- `sciona-matcher`

## Worker Ownership

One integrator only for the shared replay and ratchet logic.

Provider workers may assist by fixing replay issues in their repos, but one
integrator should own the replay sequence and the shared backfill surfaces.

## Tasks

1. Reset local Supabase from infra.

2. Replay provider seed and file-backed backfills.
   - Capture deltas in:
     - publishable atom count
     - non-publishable atom count
     - missing-surface counts

3. Verify the publication ratchet.
   - Public export remains strictly `is_publishable = true`.
   - Developer manifest still includes unpublished atoms for iteration.

4. Classify replay failures.
   - Provider-owned artifact defects go back to the owning worker.
   - Shared ingestion defects stay with the integrator.

5. Repeat until the promoted provider slices ingest cleanly.

## Validation

- clean local replay succeeds
- public publishable count increases only on genuinely audited atoms
- developer manifest count can exceed public count without changing the public
  bar
- no new publication fallback paths appear in code

## Exit Criteria

- at least one material replay pass increases the public publishable count
- the increase is traceable to provider audit bundle completion, not fallback
  logic
- the remaining non-publishable inventory is reduced and reclassified
