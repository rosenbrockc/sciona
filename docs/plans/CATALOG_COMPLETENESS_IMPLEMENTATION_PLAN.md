# Catalog Completeness Implementation Plan

## Status

Drafted on April 15, 2026 as the worker-facing execution plan for the remaining
Supabase and manifest completeness loops after the `ageoa` deprecation cutover.

## Purpose

The remaining work is no longer about schema ownership or namespace migration.
Those loops are closed.

The open work is now about catalog completeness and trust depth:

- atom benchmark coverage is still thin
- many backfill rows still skip because provider metadata names do not resolve
  to registered catalog atoms
- CDG artifacts exist, but artifact-side evidence is still behind atom-side
  evidence
- generic family skeletons are too abstract to function as trustworthy macro
  artifacts
- the SQLite manifest export path works, but the upstream catalog is not yet
  complete enough to treat each replay as a fully blessed baseline

The target state for this plan is:

- benchmark coverage exists for both concrete atoms and concrete CDG artifacts
- unresolved `skipped_no_atom` backfill gaps are materially reduced or
  explicitly classified as non-catalog metadata
- published macro artifacts are concrete enough to bind and benchmark
- CDG artifacts have evidence, uncertainty, verification, and benchmark support
  close to atom parity
- one clean reset/reseed/rebackfill/export run produces a documented local
  manifest baseline that can be trusted

## Current Repo Reality

As of the latest clean local replay:

- Supabase contains `501` canonical `sciona.*` atoms and `0` legacy `ageoa.*`
  atoms
- the seed path writes `atom_source_repositories`, `atoms`,
  `atom_versions`, and `hyperparams`
- file-backed backfills populate references, descriptions, IO specs,
  audit rollups/evidence, uncertainty, and verification matches
- tiered SQLite manifest export succeeds and is guarded against legacy rows
- benchmark suite/result manifest tooling exists, but atom-level benchmark
  results are still sparse
- CDG artifact retrieval and sync exist, but artifact-side evidence population
  is still partial

Open operational gaps from the latest replay:

- `atom_benchmarks` is still not a broad, useful comparative surface
- `io-specs.skipped_no_atom = 430`
- `parameters.skipped_no_atom = 406`
- `technical-descriptions.skipped_no_atom = 406`
- `references.skipped_no_atom = 47`
- `audit-rollups.skipped_no_atom = 406`
- `audit-evidence.skipped_no_atom = 406`
- `verification-matches.skipped_no_atom = 262`
- generic family skeletons still do not bind because they remain abstract
  scaffolds rather than concrete exemplars

## Workstreams

This plan groups the remaining work into five execution phases:

1. [CATALOG_COMPLETENESS_PHASE_1_ATOM_BENCHMARK_COVERAGE.md](/Users/conrad/personal/sciona-matcher/docs/plans/CATALOG_COMPLETENESS_PHASE_1_ATOM_BENCHMARK_COVERAGE.md)
2. [CATALOG_COMPLETENESS_PHASE_2_BACKFILL_RESOLUTION_AND_REGISTRATION.md](/Users/conrad/personal/sciona-matcher/docs/plans/CATALOG_COMPLETENESS_PHASE_2_BACKFILL_RESOLUTION_AND_REGISTRATION.md)
3. [CATALOG_COMPLETENESS_PHASE_3_CONCRETE_MACRO_ARTIFACTS.md](/Users/conrad/personal/sciona-matcher/docs/plans/CATALOG_COMPLETENESS_PHASE_3_CONCRETE_MACRO_ARTIFACTS.md)
4. [CATALOG_COMPLETENESS_PHASE_4_ARTIFACT_EVIDENCE_PARITY.md](/Users/conrad/personal/sciona-matcher/docs/plans/CATALOG_COMPLETENESS_PHASE_4_ARTIFACT_EVIDENCE_PARITY.md)
5. [CATALOG_COMPLETENESS_PHASE_5_BLESSED_REPLAY_AND_MANIFEST_BASELINE.md](/Users/conrad/personal/sciona-matcher/docs/plans/CATALOG_COMPLETENESS_PHASE_5_BLESSED_REPLAY_AND_MANIFEST_BASELINE.md)

## Dependency Structure

### Hard dependencies

- Phase 5 is last by design. It depends on the meaningful outputs of Phases 1
  through 4.
- Phase 4 depends on concrete published artifacts from Phase 3 and on at least
  some benchmark/evidence inputs being stable.

### Soft dependencies

- Phase 1 and Phase 2 can start immediately and run in parallel.
- Phase 3 can start once Phase 2 begins producing concrete registered leaves or
  once enough concrete provider-owned primitives already exist for a given
  family.
- Phase 4 can begin on a narrow slice once at least one concrete artifact from
  Phase 3 exists; it does not require every exemplar family to be finished.

## Parallelization Analysis

The work is highly parallelizable if ownership stays disjoint.

### Wave 0: Foundation inventory and first implementation branches

Contains:

- Phase 1
- Phase 2

Reason:

- benchmark completion and backfill resolution are independent enough to proceed
  at the same time
- both are upstream of the final blessed replay

Recommended split:

- Worker A: benchmark suite/result completion and `atom_benchmarks` seeding
- Worker B: provider registration/backfill resolution in `sciona-atoms`
- Worker C: provider metadata cleanup in `sciona-atoms-signal`
- Worker D: provider metadata cleanup in `sciona-atoms-bio`
- Worker E: provider metadata cleanup in `sciona-atoms-fintech`
- Worker F: provider metadata cleanup in `sciona-atoms-physics`
- Worker G: provider metadata cleanup in `sciona-atoms-robotics`

### Wave 1: Concrete macro artifact publication

Contains:

- Phase 3

Reason:

- this phase touches shared skeleton asset families, artifact publication
  surfaces, and matcher runtime ranking
- it benefits from parallel family ownership, but one integrator should own the
  shared matcher sync/retrieval seams

Recommended split:

- Worker H: signal/macrophysiology exemplars
- Worker I: state-estimation exemplars
- Worker J: inference exemplars
- Worker K: matcher-side artifact sync/retrieval integration

### Wave 2: Artifact evidence parity

Contains:

- Phase 4

Reason:

- once concrete macro artifacts exist, evidence population can be expanded for
  bindings, verification matches, uncertainty, audit evidence, and benchmarks

Recommended split:

- Worker L: artifact verification/bindings
- Worker M: artifact uncertainty/audit evidence
- Worker N: artifact benchmark ingestion and catalog document hydration

### Wave 3: Blessed replay and baseline

Contains:

- Phase 5

Reason:

- this phase is integrator-owned and should be run only after the earlier waves
  stabilize

Recommendation:

- one integrator worker or the main rollout only

## Shared Hotspots

Do not assign these files to multiple workers in the same wave:

- [../sciona-atoms/src/sciona/atoms/supabase_seed.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/supabase_seed.py>)
- [../sciona-atoms/src/sciona/atoms/supabase_backfill.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/supabase_backfill.py>)
- [sciona/services/skeleton_catalog_sync.py](/Users/conrad/personal/sciona-matcher/sciona/services/skeleton_catalog_sync.py)
- [sciona/services/catalog_artifact_retrieval.py](/Users/conrad/personal/sciona-matcher/sciona/services/catalog_artifact_retrieval.py)
- [sciona/architect/assets/skeletons](</Users/conrad/personal/sciona-matcher/sciona/architect/assets/skeletons>)
- [sciona/benchmarks/provider_results.py](/Users/conrad/personal/sciona-matcher/sciona/benchmarks/provider_results.py)
- [scripts/export_manifest.py](/Users/conrad/personal/sciona-matcher/scripts/export_manifest.py)

## Recommended Delivery Order

1. Run Phases 1 and 2 in parallel
2. Short integration pass to reconcile any new registered atoms with benchmark
   result inputs
3. Run Phase 3
4. Run Phase 4 on at least one concrete exemplar slice, then broaden
5. Run Phase 5 cleanly from reset through manifest export

## Exit Criteria

This plan should be considered complete only when all of the following are
true:

- the benchmark manifest path produces meaningful atom-level benchmark rows
- the remaining `skipped_no_atom` counts are either materially reduced or
  explicitly classified and documented as non-catalog artifacts
- at least the key macro families are represented by concrete published CDG
  exemplars rather than abstract-only skeletons
- published CDG artifacts have bindings, verification, uncertainty, audit
  evidence, and benchmark support close to atom parity
- a clean local Supabase reset/reseed/rebackfill/export run produces a
  documented manifest baseline with known counts and no integrity surprises
