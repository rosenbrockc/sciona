# SQL Manifest Phase B: Manifest Catalog Seeding

## Status

Drafted on April 14, 2026 as Phase B of
[SQL Manifest Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_IMPLEMENTATION_PLAN.md).

## Goal

Seed the architect `PrimitiveCatalog` from `manifest.sqlite` so runtime catalog
coverage is no longer limited to built-ins, saved catalogs, and locally
configured source repos.

## Purpose

This phase is the first runtime payoff from the manifest work. After it lands,
the architect can see approved atoms from provider repos that are not installed
locally.

That expands match space without changing the local-source precedence rules.

## Current Code Reality

Today [sciona/commands/runtime_helpers.py](/Users/conrad/personal/sciona-matcher/sciona/commands/runtime_helpers.py)
does this:

- create a new `PrimitiveCatalog`
- seed built-ins
- optionally load saved catalog JSONs
- derive primitives from configured sources
- attach local tunables from provider manifests

There is no manifest seeding path yet.

Also important:

- [sciona/architect/source_catalog.py](/Users/conrad/personal/sciona-matcher/sciona/architect/source_catalog.py)
  already contains catalog helpers, alias logic, and a keyword-based
  `_infer_concept_type()` helper.
- `_load_architect_catalog()` does not currently have a live `SkillIndex` in
  hand, so semantic dedup should be treated as optional rather than required
  for the first version.

## Scope

Phase B should do all of the following:

1. Add a manifest-to-primitive seeding function.
2. Reuse the Phase A `io_specs` table to build `IOSpec` objects.
3. Respect local-source precedence so manifest atoms fill gaps rather than
   override local installs.
4. Wire the seeding step into `_load_architect_catalog()`.
5. Add targeted tests for both the seeding function and runtime integration.

## Non-Goals

Phase B should not:

- build or merge semantic indexes
- change the manifest download flow
- add manifest freshness warnings
- solve perfect semantic dedup on day one

## Files In Scope

Primary files:

- [sciona/architect/source_catalog.py](/Users/conrad/personal/sciona-matcher/sciona/architect/source_catalog.py)
- [sciona/commands/runtime_helpers.py](/Users/conrad/personal/sciona-matcher/sciona/commands/runtime_helpers.py)

Primary tests:

- [tests/test_source_catalog.py](/Users/conrad/personal/sciona-matcher/tests/test_source_catalog.py)
- [tests/test_catalog_sync.py](/Users/conrad/personal/sciona-matcher/tests/test_catalog_sync.py)

## Implementation Steps

### Step 1: Add manifest seeding entry point

Create `seed_catalog_from_manifest_sqlite()` in `source_catalog.py`.

Responsibilities:

- open the manifest read-only
- query approved atoms
- load corresponding `io_specs`
- build `AlgorithmicPrimitive` instances
- add them to the provided catalog
- return the number of newly added primitives

This function should live near the existing source seeding logic so catalog
population policies remain in one module.

### Step 2: Reuse existing concept inference

Do not invent a second concept-classification subsystem just for manifest rows.

Preferred approach:

- use any explicit concept signal from manifest data if it is added later
- otherwise reuse the existing keyword-based `_infer_concept_type()` helper from
  `source_catalog.py`
- feed it atom fqdn, source/module hints, description, and domain tags

If the available metadata is weak, fall back to `ConceptType.CUSTOM`.

### Step 3: Build `IOSpec` values faithfully

Map manifest `io_specs` rows onto
[IOSpec](/Users/conrad/personal/sciona-matcher/sciona/architect/models.py):

- `port_name -> name`
- `type_desc -> type_desc`
- `constraints -> constraints`
- `data_kind -> data_kind`
- `required -> required`
- `default_value_repr -> default_value_repr`

Preserve port ordering through the `ordinal` field.

### Step 4: Decide dedup policy explicitly

The current runtime does not pass a `SkillIndex` into `_load_architect_catalog`.
That matters.

First-version dedup policy should be:

- exact-name precedence for anything already in the catalog
- local-source atoms win over manifest atoms
- optional semantic dedup only if a skill index can be threaded in cleanly
  without inflating this phase

This keeps the phase bounded and avoids forcing a new runtime dependency
through every CLI code path.

### Step 5: Wire into runtime catalog loading

Update `_load_architect_catalog()` so it:

1. seeds built-ins
2. loads saved catalogs if applicable
3. seeds configured local sources
4. if `~/.sciona/manifest.sqlite` exists, seeds manifest atoms that are still
   missing
5. continues attaching tunables as it does today

The manifest should be additive, not authoritative, at this stage.

## Testing Plan

Add or extend tests for:

- manifest seeding builds primitives with correct inputs, outputs, description,
  and category fallback
- locally seeded primitives prevent same-name manifest overrides
- `_load_architect_catalog()` uses manifest atoms when local sources are absent
- `_load_architect_catalog()` still behaves correctly when no manifest file is
  present
- broken or partial manifests degrade gracefully instead of aborting the whole
  catalog load

## Worker Breakdown

Recommended split:

- Worker B1: `source_catalog.py` plus `tests/test_source_catalog.py`
- Worker B2: `runtime_helpers.py` plus `tests/test_catalog_sync.py`

Constraint:

- B2 should start after B1 has defined the manifest seeding API, or the same
  worker should own both.

Do not run B2 concurrently with Phase C integration work in
`runtime_helpers.py`.

## Risks And Decisions

### Name shape

Manifest atoms should be registered under their stable FQDN, not only under a
leaf function name. Aliases can be added later if helpful.

### Category quality

Domain-tag-only inference will not always be strong. That is acceptable for the
first version as long as the fallback is conservative and deterministic.

### Tunables

Phase B should not block on attaching manifest hyperparameters to the catalog.
The runtime already has a separate hyperparameter loader path. Folding that into
catalog seeding can be a follow-up if needed.

## Exit Criteria

Phase B is complete when:

- the catalog can be seeded from `manifest.sqlite`
- local source entries still take precedence
- manifest-only atoms appear in the runtime catalog
- the new behavior is covered by source-catalog and runtime-catalog tests
