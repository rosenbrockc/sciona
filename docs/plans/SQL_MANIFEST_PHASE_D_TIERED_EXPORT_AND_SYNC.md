# SQL Manifest Phase D: Tiered Export And Sync

## Status

Drafted on April 14, 2026 as Phase D of
[SQL Manifest Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_IMPLEMENTATION_PLAN.md).

## Goal

Add visibility-tier-aware manifest export and download so different user tiers
can receive different published catalogs while still using the same local
`manifest.sqlite` path.

## Purpose

The manifest is intended to be a distributable offline catalog, not merely a
local development artifact. That requires tier-aware export naming and tier-
aware sync behavior.

## Current Code Reality

Today:

- `fetch_manifest_data()` always fetches the full approved and publishable set
- `generate_manifest_sqlite()` writes one manifest with no tier metadata beyond
  per-atom columns
- [sciona/commands/catalog_cmds.py](/Users/conrad/personal/sciona-matcher/sciona/commands/catalog_cmds.py)
  resolves exactly one object key: `manifests/manifest.sqlite`
- [sciona/cli.py](/Users/conrad/personal/sciona-matcher/sciona/cli.py)
  exposes no `--tier` flag

This phase is mostly independent from the runtime catalog and index phases, but
it shares `snapshot.py` with Phase A and Phase F.

## Scope

Phase D should do all of the following:

1. Parameterize manifest fetching by visibility tier set.
2. Define tier presets in one canonical location.
3. Export one manifest per tier.
4. Add `--tier` support to `sciona catalog sync`.
5. Persist the chosen tier in `manifest_metadata`.

## Non-Goals

Phase D should not:

- enable CI scheduling
- change reader-side freshness behavior
- redesign Supabase entitlements or RLS policies

## Files In Scope

Primary files:

- [sciona/api/snapshot.py](/Users/conrad/personal/sciona-matcher/sciona/api/snapshot.py)
- [sciona/commands/catalog_cmds.py](/Users/conrad/personal/sciona-matcher/sciona/commands/catalog_cmds.py)
- [sciona/cli.py](/Users/conrad/personal/sciona-matcher/sciona/cli.py)

Primary tests:

- [tests/test_supabase_snapshot.py](/Users/conrad/personal/sciona-matcher/tests/test_supabase_snapshot.py)

## Implementation Steps

### Step 1: Add tier presets

Define a single `MANIFEST_TIERS` mapping in `snapshot.py`.

Requirements:

- one canonical source of truth
- stable keys used in both export and sync
- values are the Supabase visibility tiers to include

### Step 2: Add tier filtering to fetch

Extend `fetch_manifest_data()` with a `visibility_tiers` keyword parameter.

The filter should apply to the atom query first so all downstream batch queries
inherit the filtered atom id set.

### Step 3: Add multi-manifest export helper

Introduce `export_tiered_manifests()` in `snapshot.py`.

Responsibilities:

- iterate through `MANIFEST_TIERS`
- fetch data for each tier
- write `manifest-{tier}.sqlite`
- populate `manifest_metadata.visibility_tier` correctly
- return a tier-to-path mapping for callers

### Step 4: Make sync tier-aware

Update `catalog_cmds.py` and `cli.py` so `sciona catalog sync` accepts:

- `--tier`, defaulting to the lowest public tier

Resolution behavior should be deterministic:

- artifact key becomes `manifests/manifest-{tier}.sqlite`
- local output path remains `~/.sciona/manifest.sqlite` unless overridden

### Step 5: Preserve explicit overrides

If `--manifest-url` is passed, that explicit URL should still win.

Tier logic should affect default URL resolution, not break direct override use
cases or tests.

## Testing Plan

Add or extend tests for:

- atom fetch filtering by tier set
- multi-manifest export writes one file per tier
- metadata in each output records the chosen tier
- default sync URL includes the selected tier
- explicit `--manifest-url` still overrides tier-based resolution
- CLI parser accepts `--tier`

## Worker Breakdown

Recommended split:

- Worker D1: `snapshot.py` tiering and snapshot tests
- Worker D2: `catalog_cmds.py`, `cli.py`, and sync-related tests

Coordination point:

- D1 should define final artifact naming before D2 finalizes URL resolution

This phase parallelizes well because the write sets are mostly disjoint after
the naming contract is set.

## Risks And Decisions

### Tier naming

Do not expose one naming scheme in export and a different one in sync. The
object key convention must be centralized and boring.

### Default tier

The default should be the most permissive public tier that anonymous or basic
users can always fetch. Do not default to a contributor-only tier.

### Metadata authority

The local filename is intentionally non-tiered. The loaded tier must therefore
be inspectable from `manifest_metadata`, not inferred from the path.

## Exit Criteria

Phase D is complete when:

- tier presets exist
- tiered manifest export works
- `sciona catalog sync --tier ...` resolves the correct artifact
- the written manifest records its tier in metadata
