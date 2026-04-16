# Atom License Compliance Phase 6: Blessed Compliance Baseline

## Goal

Run a clean local replay and produce the first trustworthy license-aware catalog baseline.

## Scope

- infra-owned schema reset
- provider-owned seed/backfill
- matcher manifest export
- compliance reporting

## Tasks

1. Reset local Supabase from `../sciona-infra`.
2. Reseed provider data including version-scoped license metadata.
3. Rebuild any artifact projections needed for macro retrieval.
4. Export public and developer manifests.
5. Produce a compliance summary:
   - atoms by license status
   - artifacts by license status
   - atoms excluded from public manifest on license grounds
   - unresolved legal-review backlog

## Parallelization

- none for the final blessed replay; use a single integrator

## Acceptance

- replay succeeds end to end
- public manifest excludes unknown or restricted license rows
- developer manifest behavior matches the configured override policy
- compliance baseline is saved as a reference artifact
