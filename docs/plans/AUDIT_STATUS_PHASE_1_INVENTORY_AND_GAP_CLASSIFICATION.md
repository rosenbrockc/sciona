# Audit Status Phase 1: Inventory And Gap Classification

## Goal

Produce the canonical inventory of remaining non-publishable atoms, grouped by
provider repo and family, and classify which provider-owned artifacts are
missing for each group.

## Why This Phase Matters

The remaining non-publishable atoms are not evenly distributed. The work needs
to be routed to the right provider owners instead of handled as an undifferentiated
Supabase cleanup.

## Scope

This phase covers:

- extracting the live non-publishable inventory from local Supabase
- grouping by provider repo, namespace family, and artifact path
- classifying missing audit surfaces:
  - IO specs
  - parameter manifests
  - technical descriptions
  - dejargonized descriptions
  - references
  - audit rollups
- documenting which groups already have draft audit artifacts and which have
  none

## Primary Repos

- `../sciona-atoms`
- `../sciona-atoms-signal`
- sibling provider repos as read-only inputs
- `sciona-matcher` for plan output only

## Worker Ownership

One integrator only.

This phase should not be split because the resulting inventory must become the
single source of truth for the later provider workers.

## Tasks

1. Query the live non-publishable atom inventory from local Supabase.
   - Group by namespace family and provider ownership.

2. Map each family bucket to the real provider repo and artifact roots.
   - Prefer repo-owned manifests and metadata paths over runtime-derived guesses.

3. Classify each bucket into one of three states:
   - `draft audit present but incomplete`
   - `metadata present but no audit bundle`
   - `no usable audit artifacts yet`

4. Produce the worker ownership ledger.
   - For each bucket, list the repo, family, expected artifact files, and
     target output count.

5. Freeze the ratchet policy for later phases.
   - No atom becomes publishable unless the provider-owned audit artifacts are
     present and ingest cleanly.

## Validation

- every non-publishable atom is assigned to a provider/family bucket
- every bucket names a concrete repo owner
- every bucket has a stated missing-surface classification
- the sum of bucket counts matches the live non-publishable count

## Exit Criteria

- one canonical provider/family inventory exists for all remaining
  non-publishable atoms
- later workers can take disjoint ownership of their buckets without ambiguity
