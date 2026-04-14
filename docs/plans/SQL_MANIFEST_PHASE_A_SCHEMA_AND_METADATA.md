# SQL Manifest Phase A: Schema And Metadata Foundation

## Status

Drafted on April 14, 2026 as Phase A of
[SQL Manifest Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_IMPLEMENTATION_PLAN.md).

## Goal

Extend `manifest.sqlite` so it can describe atom ports and carry self-describing
metadata without breaking existing readers.

This phase is the semantic foundation for the rest of the manifest work.

## Why This Phase Comes First

The runtime phases need two things that the current schema does not provide:

- typed port data so manifest atoms can be turned into `AlgorithmicPrimitive`
  and `Declaration` objects
- metadata so tier, freshness, and integrity can be reasoned about at load time

Without those tables, later phases either guess or duplicate state.

## Current Code Reality

Today [sciona/api/snapshot.py](/Users/conrad/personal/sciona-matcher/sciona/api/snapshot.py):

- fetches atoms, hyperparams, audit rollups, descriptions, and benchmarks
- writes five tables into SQLite
- accepts either the legacy positional call shape or the newer manifest-data
  mapping
- has no manifest metadata helper layer yet

The file is already a hotspot, so this phase should be treated as a single
owned change, not a broad parallel effort.

## Scope

Phase A should do all of the following:

1. Add an `io_specs` table to the generated SQLite schema.
2. Fetch `atom_io_specs` rows from Supabase for the fetched atom set.
3. Add a `manifest_metadata` table.
4. Write generated-at, generator-version, visibility-tier, and content-hash
   metadata.
5. Preserve backward compatibility for existing manifest readers and tests.

## Non-Goals

Phase A should not:

- wire manifest atoms into the architect catalog
- wire manifest declarations into the semantic index
- change the CLI sync contract
- enable CI export
- add freshness warnings in loaders

Those belong to later phases.

## Files In Scope

Primary files:

- [sciona/api/snapshot.py](/Users/conrad/personal/sciona-matcher/sciona/api/snapshot.py)

Primary tests:

- [tests/test_snapshot_generation.py](/Users/conrad/personal/sciona-matcher/tests/test_snapshot_generation.py)
- [tests/test_supabase_snapshot.py](/Users/conrad/personal/sciona-matcher/tests/test_supabase_snapshot.py)

## Implementation Steps

### Step 1: Extend the manifest data contract

Update `_coerce_manifest_data()` so the normalized mapping includes:

- `io_specs`
- `manifest_metadata` only if explicitly passed in later

The writer should still synthesize metadata itself by default. The normalized
shape matters because later phases rely on the mapping interface, not the
legacy positional one.

### Step 2: Add schema tables

Extend `_create_schema()` to add:

- `io_specs`
- `manifest_metadata`

Keep existing table names and column meanings stable so current readers remain
compatible.

The schema should remain drop-and-rebuild for now because
`generate_manifest_sqlite()` already behaves as a full snapshot writer.

### Step 3: Add row writers

Introduce dedicated helpers rather than inlining more SQL into
`generate_manifest_sqlite()`:

- `_insert_io_spec()`
- `_insert_manifest_metadata()`
- one small helper for content-hash generation if needed

This is important because D and F will extend metadata handling later.

### Step 4: Fetch io specs from Supabase

Extend `fetch_manifest_data()` to query `atom_io_specs` in atom-id batches,
similar to how hyperparams and descriptions are fetched.

Constraints for this step:

- query only rows for fetched atoms
- preserve deterministic ordering
- return a shape the SQLite writer can ingest directly

### Step 5: Write metadata deterministically

`generate_manifest_sqlite()` should compute and store:

- `generated_at`
- `generator_version`
- `visibility_tier`
- `content_hash`

The content hash should be deterministic from manifest content, not from row
insertion timing.

### Step 6: Keep existing readers working

Readers that only know about the old schema must continue to work against the
new manifest.

That means:

- no rename of existing tables
- no change to required columns used by current readers
- no new hard dependency on metadata for success

## Testing Plan

Add or extend tests for:

- schema contains `io_specs` and `manifest_metadata`
- io spec rows are written from the manifest-data mapping
- metadata rows are present and deterministic enough to validate shape
- legacy `generate_manifest_sqlite(atoms, hyperparams, benchmarks=...)` still
  works
- `load_hyperparams_manifest_sqlite()` and `load_benchmarks_sqlite()` still
  read a newly generated manifest
- `fetch_manifest_data()` includes `io_specs` when Supabase returns them

## Worker Breakdown

Recommended ownership:

- one worker owns `snapshot.py` and both snapshot-related test files

Not recommended:

- splitting implementation and tests across two workers
- parallel edits to `snapshot.py`

The write-set overlap is too high for this phase to benefit from multiple
workers.

## Risks And Decisions

### Metadata versioning

The generator version field needs a stable source. Do not hardcode an arbitrary
string in multiple places. Prefer one helper or module-level constant.

### Legacy manifests

Later readers will need to tolerate manifests that predate this table. Phase A
should not assume all manifests in the world are already upgraded.

### `visibility_tier` default

The metadata table should permit a sensible default even before Phase D lands.
`all` is the safest placeholder for a non-tiered export path.

## Exit Criteria

Phase A is complete when:

- `fetch_manifest_data()` returns `io_specs`
- generated manifests contain `io_specs` and `manifest_metadata`
- old manifest readers still pass their tests
- the new schema is stable enough for Phase B, C, D, and F to build on
