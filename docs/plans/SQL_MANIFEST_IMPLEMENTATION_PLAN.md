# SQL Manifest Implementation Plan

## Status

Drafted on April 14, 2026 from the phase summary in
[SQL_MANIFEST.md](/Users/conrad/personal/sciona-matcher/SQL_MANIFEST.md).

This document is the worker-facing parent plan for the `manifest.sqlite`
roadmap. It preserves the original phase structure while adapting it to the
current code layout in this repo.

## Purpose

The repo already has a working Supabase-backed manifest export path for:

- approved atoms
- approved hyperparameters
- benchmark rows
- audit rollups
- dejargonized descriptions

What it does not yet have is the runtime surface needed for the hunter and
architect to use that manifest as a first-class discovery source for atoms that
are not locally installed.

The goal of this plan is to close that gap without destabilizing the existing
offline manifest readers.

## Current Repo Reality

The current implementation surface is narrower than the summary in
`SQL_MANIFEST.md` assumes:

- [sciona/api/snapshot.py](/Users/conrad/personal/sciona-matcher/sciona/api/snapshot.py)
  already fetches and writes `atoms`, `hyperparams`, `benchmarks`,
  `audit_rollups`, and `descriptions`, but does not yet include `io_specs`,
  manifest metadata, or tiered export.
- [sciona/commands/runtime_helpers.py](/Users/conrad/personal/sciona-matcher/sciona/commands/runtime_helpers.py)
  loads the architect catalog from built-ins, saved catalogs, and configured
  source registries only.
- [sciona/indexer/builder.py](/Users/conrad/personal/sciona-matcher/sciona/indexer/builder.py)
  and [sciona/indexer/unified.py](/Users/conrad/personal/sciona-matcher/sciona/indexer/unified.py)
  only model a single-store semantic index today.
- [sciona/commands/catalog_cmds.py](/Users/conrad/personal/sciona-matcher/sciona/commands/catalog_cmds.py)
  downloads a single fixed manifest object and does not know about visibility
  tiers.
- [sciona/architect/hyperparams.py](/Users/conrad/personal/sciona-matcher/sciona/architect/hyperparams.py)
  and [sciona/ecosystem/benchmarks.py](/Users/conrad/personal/sciona-matcher/sciona/ecosystem/benchmarks.py)
  read the manifest without freshness or integrity checks.

That means the implementation should not be treated as one broad worker task.
There are real shared seams and a few places where the most efficient path is
to serialize integration even if the high-level phases are conceptually
independent.

## Phase Set

This implementation is broken into six manifest phases:

1. Schema And Metadata Foundation
2. Manifest Catalog Seeding
3. Manifest Semantic Index
4. Tiered Export And Sync
5. CI Export Workflow
6. Integrity And Freshness

Each phase is documented separately:

- [SQL_MANIFEST_PHASE_A_SCHEMA_AND_METADATA.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_A_SCHEMA_AND_METADATA.md)
- [SQL_MANIFEST_PHASE_B_MANIFEST_CATALOG_SEEDING.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_B_MANIFEST_CATALOG_SEEDING.md)
- [SQL_MANIFEST_PHASE_C_MANIFEST_SEMANTIC_INDEX.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_C_MANIFEST_SEMANTIC_INDEX.md)
- [SQL_MANIFEST_PHASE_D_TIERED_EXPORT_AND_SYNC.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_D_TIERED_EXPORT_AND_SYNC.md)
- [SQL_MANIFEST_PHASE_E_CI_EXPORT_WORKFLOW.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_E_CI_EXPORT_WORKFLOW.md)
- [SQL_MANIFEST_PHASE_F_INTEGRITY_AND_FRESHNESS.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_F_INTEGRITY_AND_FRESHNESS.md)

## Dependency Structure

The phases are not purely linear, but they are also not free to run fully in
parallel.

### Hard dependencies

- Phase A must land before Phases B, C, D, and F because those phases depend on
  the `io_specs` and `manifest_metadata` contract.
- Phase D must land before Phase E because CI export needs the tiered export
  functions and tier-aware artifact naming.
- Phase F depends on the metadata table shape introduced in Phase A.

### Soft dependencies

- Phase B and Phase C can start after Phase A, but both eventually need to
  touch [sciona/commands/runtime_helpers.py](/Users/conrad/personal/sciona-matcher/sciona/commands/runtime_helpers.py).
  That integration seam should be owned by one worker or one final integrator,
  not edited concurrently.
- Phase D and Phase F both touch
  [sciona/api/snapshot.py](/Users/conrad/personal/sciona-matcher/sciona/api/snapshot.py).
  Their conceptual dependencies are light, but their write-set overlap is real.
- Phase C should reuse the same embedder family as the persisted local index.
  Mixed embedding spaces must not be score-merged.

## Parallelization Analysis

The right execution model is dependency waves plus explicit ownership of shared
files.

### Wave 0: Schema contract

Contains:

- Phase A

Reason:

- It defines the manifest row shapes that every later phase consumes.
- It is centered on `snapshot.py`, which is already a shared hotspot.

Recommendation:

- Use one worker only for Phase A.

### Wave 1: Runtime and distribution branches

Contains:

- Phase B core work
- Phase C core work
- Phase D core work

Reason:

- After the schema stabilizes, catalog seeding, semantic retrieval, and
  distribution tiering can advance mostly independently.

Caveats:

- Phase B and Phase C should not both edit `runtime_helpers.py` at the same
  time.
- Phase D should own the tiering changes in `snapshot.py` and
  `catalog_cmds.py`.

Recommended split:

- Worker 1: Phase B core in `source_catalog.py` plus tests.
- Worker 2: Phase C core in `indexer/` plus tests.
- Worker 3: Phase D in `snapshot.py`, `catalog_cmds.py`, `cli.py`, and tests.
- Final integrator: the `runtime_helpers.py` changes that connect B and C to
  the live runtime.

### Wave 2: Hardening

Contains:

- Phase F

Reason:

- Freshness and integrity checks are cheap to add once the metadata contract is
  real.
- This phase is not on the critical path for first runtime value.

Recommendation:

- Start after Phase A and after the metadata-writing shape in Phase D is
  stable, even though D is not a hard dependency.

### Wave 3: Operationalization

Contains:

- Phase E

Reason:

- CI should be the last step. It depends on the export interface settling
  first.

## Worker Planning Guidance

These phases are suitable for worker execution, but only if ownership is kept
clean:

- Do not assign the same shared file to multiple workers in the same wave.
- Treat `snapshot.py` and `runtime_helpers.py` as integration hotspots.
- Prefer worker tasks that own one module cluster and one test cluster.
- Use one integrator worker or one final human pass to merge the runtime seams.

## Recommended Delivery Order

1. Phase A
2. Phase D
3. Phase B core and Phase C core in parallel
4. Runtime integration pass for B and C
5. Phase F
6. Phase E

This order slightly differs from the summary table in `SQL_MANIFEST.md`. The
reason is pragmatic: tiered export is cheap to land once `snapshot.py` is open,
while B and C share a runtime seam that benefits from a short integration pass
after both cores exist.

## Exit Criteria For The Whole Plan

The broader manifest effort should be considered implemented when all of the
following are true:

- the manifest schema contains `io_specs` and `manifest_metadata`
- the architect catalog can seed non-local primitives from `manifest.sqlite`
- the hunter can retrieve manifest-backed declarations through the semantic
  index path without corrupting score semantics
- `sciona catalog sync` can download the correct tiered artifact
- CI can export tiered manifests manually
- manifest readers warn on stale artifacts and can validate basic integrity
