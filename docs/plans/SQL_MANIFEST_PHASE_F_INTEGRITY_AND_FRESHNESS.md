# SQL Manifest Phase F: Integrity And Freshness

## Status

Drafted on April 14, 2026 as Phase F of
[SQL Manifest Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_IMPLEMENTATION_PLAN.md).

## Goal

Add lightweight reader-side checks so stale or obviously inconsistent manifests
are surfaced early without breaking backward compatibility.

## Purpose

This is the hardening phase for the manifest path. It should improve operator
clarity without turning the manifest into a brittle source of runtime failures.

## Current Code Reality

Today:

- `load_hyperparams_manifest_sqlite()` silently reads whatever file exists
- `load_benchmarks_sqlite()` silently reads whatever file exists
- there is no freshness warning path
- there is no content-hash verification path
- legacy manifests without metadata are still likely to exist

## Scope

Phase F should do all of the following:

1. Add a shared freshness check helper.
2. Invoke it from manifest readers.
3. Add optional content-hash validation.
4. Keep legacy manifests readable.

## Non-Goals

Phase F should not:

- make stale manifests fatal by default
- block runtime just because metadata is missing
- redesign manifest generation now that Phase A already owns the base schema

## Files In Scope

Primary files:

- [sciona/architect/hyperparams.py](/Users/conrad/personal/sciona-matcher/sciona/architect/hyperparams.py)
- [sciona/ecosystem/benchmarks.py](/Users/conrad/personal/sciona-matcher/sciona/ecosystem/benchmarks.py)
- [sciona/api/snapshot.py](/Users/conrad/personal/sciona-matcher/sciona/api/snapshot.py)

Primary tests:

- [tests/test_hyperparams.py](/Users/conrad/personal/sciona-matcher/tests/test_hyperparams.py)
- [tests/test_benchmarks.py](/Users/conrad/personal/sciona-matcher/tests/test_benchmarks.py)
- snapshot tests if content-hash generation details need coverage

## Implementation Steps

### Step 1: Add a shared metadata reader helper

Avoid duplicating SQLite metadata parsing in multiple loaders.

Preferred direction:

- one small helper to read `manifest_metadata`
- one freshness-check helper built on top of it

This can live in `hyperparams.py` initially or in a small shared utility module
if that keeps the duplication cleaner.

### Step 2: Warn on stale manifests

Add a non-fatal warning when:

- metadata exists
- `generated_at` parses successfully
- the artifact is older than the chosen threshold

The user-facing action should be clear: run `sciona catalog sync`.

### Step 3: Tolerate legacy manifests

If the metadata table or key is missing:

- return normally
- do not warn just because the manifest predates the feature

That keeps backward compatibility intact.

### Step 4: Validate content hash conservatively

Add helper logic that can recompute the hash from manifest content and compare
it to `manifest_metadata.content_hash`.

This should be:

- warning-only by default
- isolated from the core row-loading logic
- easy to disable or skip for legacy manifests

### Step 5: Call checks from both readers

At minimum:

- `load_hyperparams_manifest_sqlite()`
- `load_benchmarks_sqlite()`

Both should share the same threshold and warning behavior where practical.

## Testing Plan

Add or extend tests for:

- freshness warning on old manifests
- no warning on recent manifests
- no crash on missing metadata table
- content-hash mismatch warning path
- existing loader behavior remaining unchanged for valid manifests

## Worker Breakdown

Recommended ownership:

- one worker owns both reader modules and their test files

Optional split only if needed:

- Worker F1: reader-side warnings in `hyperparams.py` and `benchmarks.py`
- Worker F2: snapshot-side hash generation refinements

This split is only safe if Phase A and D are already stable. Otherwise F should
be one small cleanup task.

## Risks And Decisions

### Warning channel

Use Python warnings for stale or mismatched metadata rather than `print()`.
That keeps the behavior testable and non-fatal.

### Threshold choice

The first threshold should be conservative and simple. Thirty days is a good
default until there is stronger operational guidance.

### Hash meaning

The content hash should be documented as a lightweight integrity signal over
manifest content, not as a cryptographic proof of distribution authenticity.

## Exit Criteria

Phase F is complete when:

- stale manifests produce a warning
- mismatched content hashes can be detected
- legacy manifests still load
- the warning behavior is covered by loader tests
