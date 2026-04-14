# Expansion / Rewrite / Skeleton Registration Plan

## Goal

Convert the remaining declaration-style expansion helpers into properly registered atoms so they seed into Supabase, participate in Hunter retrieval, bind cleanly from published CDG artifacts, and eventually flow into the SQLite manifest without special cases.

This plan treats the `signal_event_rate` conversion as the template implementation:

- provider module functions remain the runtime implementation
- each reusable helper gets a typed Ghost witness plus `@register_atom`
- probe metadata remains as compatibility and test surface, not the primary catalog source
- reseed + skeleton sync become the acceptance gate

## Current State

Completed:

- `sciona.atoms.expansion.signal_event_rate.*` is now fully registered in `../sciona-atoms-signal`
- local Supabase reseed now inserts those atoms and versions
- `cdg.skeleton.signal_detect_measure` now resolves `3` live `artifact_cdg_bindings`

Remaining inventory:

| Family | Repo | Remaining targets |
| --- | --- | ---: |
| `belief_propagation` | `../sciona-atoms` | 4 |
| `divide_and_conquer` | `../sciona-atoms` | 4 |
| `kalman_filter` | `../sciona-atoms` | 4 |
| `particle_filter` | `../sciona-atoms` | 4 |
| `sequential_filter` | `../sciona-atoms` | 4 |
| `graph_signal_processing` | `../sciona-atoms-signal` | 4 |
| `signal_detect_measure` | `../sciona-atoms-signal` | 4 |
| `signal_filter` | `../sciona-atoms-signal` | 4 |
| `signal_transform` | `../sciona-atoms-signal` | 4 |

Total remaining registration targets: `36`

Important constraint:

- there is no standalone `src/sciona/probes/rewrite/` tree today
- rewrite metadata currently lives inside expansion assets such as `../sciona-atoms-signal/data/expansions/signal_event_rate.json`
- so "rewrite conversion" means aligning those asset-level rewrite operations with real registered helper atoms, not migrating a separate Python subsystem

## Worker Pattern

Each family worker should own one module and its tests only.

Expected write scope per worker:

- one `src/sciona/atoms/expansion/<family>.py` module
- one or two focused tests under the same repo's `tests/`
- optional asset/probe compatibility updates for that family only

Required implementation pattern:

1. Add typed Ghost witnesses next to the existing runtime functions or in a family-local `witnesses` helper.
2. Decorate each reusable wrapper with `@register_atom`.
3. Keep existing declaration/probe constants for compatibility until downstream callers are removed.
4. Add a smoke test asserting import plus `list_registered()` membership.
5. Preserve runtime behavior. This is a catalog correction, not an algorithm rewrite.

Required validation per family:

1. family tests pass
2. import/registration smoke passes
3. provider reseed inserts the expected FQDNs
4. if the family feeds a skeleton CDG, rerun skeleton sync and confirm new bindings appear

## Phase 1: Core Expansion Families

Scope:

- `../sciona-atoms/src/sciona/atoms/expansion/belief_propagation.py`
- `../sciona-atoms/src/sciona/atoms/expansion/divide_and_conquer.py`
- `../sciona-atoms/src/sciona/atoms/expansion/kalman_filter.py`
- `../sciona-atoms/src/sciona/atoms/expansion/particle_filter.py`
- `../sciona-atoms/src/sciona/atoms/expansion/sequential_filter.py`

Why first:

- these families back the generic skeleton families already exposed at the macro level
- registering them lets `cdg.skeleton.family.divide_and_conquer.v1` and `cdg.skeleton.family.sequential_filter.v1` move from metadata-only assets toward real bound artifacts

Parallelization:

- safe to run with up to `5` workers in parallel
- each worker owns one family module plus its tests
- integration pass after all five land: single reseed + single skeleton sync

Acceptance criteria:

- all `20` remaining core-repo expansion helpers are registered
- Supabase reseed inserts their `atoms` and `atom_versions`
- generic family skeletons gain non-zero `artifact_cdg_bindings`

## Phase 2: Signal Expansion Families

Scope:

- `../sciona-atoms-signal/src/sciona/atoms/expansion/graph_signal_processing.py`
- `../sciona-atoms-signal/src/sciona/atoms/expansion/signal_detect_measure.py`
- `../sciona-atoms-signal/src/sciona/atoms/expansion/signal_filter.py`
- `../sciona-atoms-signal/src/sciona/atoms/expansion/signal_transform.py`

Why second:

- these sit closest to the already-proven `signal_event_rate` pattern
- `signal_detect_measure` is the most immediate skeleton/CDG beneficiary after `signal_event_rate`

Parallelization:

- safe to run with up to `4` workers in parallel
- keep `signal_detect_measure` isolated because it is the likely first family to expose additional skeleton-binding wins
- one shared reseed after the wave, then one skeleton sync

Acceptance criteria:

- all `16` remaining signal-repo expansion helpers are registered
- `signal_detect_measure` stops relying on suffix-only fallback for its leaf set
- signal family skeletons and expansion assets resolve through real seeded atoms

## Phase 3: Rewrite Metadata Alignment

Scope:

- family assets under `data/expansions/*.json`
- family heuristics under `data/heuristics/families/*.json`
- any tests that assert rewrite step names or `matched_primitive` strings

Purpose:

- ensure every rewrite-described operation now points at a real registered atom FQDN
- eliminate drift between asset-level rewrite docs and actual seedable callable symbols

Work items:

1. Replace any stale short names with the registered function symbols where needed.
2. Normalize `matched_primitive` conventions so skeleton sync does not depend on ad hoc aliases.
3. Add an asset-consistency test that fails if a rewrite step names a non-registered helper.

Parallelization:

- can run in parallel with Phase 2 family work, but only after each family's registrations are merged
- best executed by one integration worker per repo, not per family

Acceptance criteria:

- asset-level rewrite references resolve to real registered atoms
- no expansion asset refers to a helper missing from the seed inventory

## Phase 4: Skeleton Binding Hardening

Scope:

- matcher skeleton sync and planner-facing artifact retrieval
- verification ingestion for skeleton-bound leafs

Purpose:

- once the expansion helpers exist as real atoms, make the skeleton/CDG side fully consume them
- reduce reliance on suffix-only fallback and transitional heuristics

Work items:

1. Rerun provider seed and file-backed backfills after each registration wave.
2. Rerun `scripts/sync_skeleton_artifacts.py`.
3. Capture per-skeleton binding counts, unresolved nodes, and verification coverage.
4. Tighten publishability rules so approved skeletons only advance when bindings and evidence exist.

Parallelization:

- not worth parallelizing internally
- run once after Phase 1 and once after Phase 2

Acceptance criteria:

- family skeletons gain live bound leaves in Supabase
- Memgraph projection contains those updated artifact bindings
- no current skeleton depends on declaration-only expansion helpers

## Phase 5: Manifest and Catalog Closure

Scope:

- `../sciona-atoms` seed/backfill replay
- matcher artifact sync
- SQLite manifest export

Purpose:

- close the loop so registered expansion helpers become normal catalog entries everywhere

Work items:

1. Run one clean local Supabase reset from `../sciona-infra`.
2. Reseed providers from the current repos.
3. Rerun file-backed backfills.
4. Rerun skeleton artifact sync.
5. Verify the new expansion helpers appear in the local artifact/atom catalog and downstream manifest export input.

Acceptance criteria:

- no remaining expansion helper needed by a published skeleton is declaration-only
- Supabase and Memgraph agree on bound skeleton leaf sets
- SQLite manifest generation sees the registered helpers through normal export paths

## Recommended Execution Waves

Wave A:

- Phase 1 family workers in parallel

Wave B:

- single integration pass: reseed + sync + binding report

Wave C:

- Phase 2 family workers in parallel

Wave D:

- single integration pass: reseed + sync + binding report

Wave E:

- Phase 3 rewrite metadata alignment
- Phase 4 skeleton binding hardening

Wave F:

- Phase 5 clean reset/reseed/manifest closure

## Deliverables Per Worker

Every worker handoff should include:

- changed file paths
- newly registered atom FQDNs
- focused tests run
- whether reseed was required for that slice
- whether any asset or skeleton references had to change

## Exit Condition

This plan is complete when expansion and rewrite-intent helpers no longer require probe-only or declaration-only handling anywhere in the publication path:

- provider repo source defines them as registered atoms
- Supabase seeds them directly
- skeleton/CDG sync binds them directly
- manifest export snapshots them without special logic
