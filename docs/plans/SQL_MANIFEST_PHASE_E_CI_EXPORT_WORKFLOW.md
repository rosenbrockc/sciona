# SQL Manifest Phase E: CI Export Workflow

## Status

Drafted on April 14, 2026 as Phase E of
[SQL Manifest Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_IMPLEMENTATION_PLAN.md).

## Goal

Operationalize tiered manifest export through a script and a disabled or
manual-only GitHub Actions workflow.

## Purpose

The earlier phases make local export possible. This phase turns that capability
into a repeatable operational path without forcing automatic rollout before the
infrastructure is ready.

## Current Code Reality

The repo already contains GitHub Actions workflows under
[.github/workflows](/Users/conrad/personal/sciona-matcher/.github/workflows),
but there is no manifest export workflow yet and no export script for tiered
publish.

## Scope

Phase E should do all of the following:

1. Add an export script for local or CI use.
2. Optionally upload per-tier manifests to S3.
3. Publish a small `latest.json` metadata file.
4. Add a manual-only GitHub Actions workflow for the export path.

## Non-Goals

Phase E should not:

- turn on scheduled production export automatically
- redesign AWS auth or secret management
- replace local verification with CI-only validation

## Files In Scope

Primary files:

- [scripts/export_manifest.py](/Users/conrad/personal/sciona-matcher/scripts/export_manifest.py)
- [.github/workflows/export-manifest.yml](/Users/conrad/personal/sciona-matcher/.github/workflows/export-manifest.yml)

Likely new tests:

- `tests/test_export_manifest.py`

## Implementation Steps

### Step 1: Add the export script

Create `scripts/export_manifest.py` with a minimal CLI:

- `--output-dir` required
- `--upload` optional

Responsibilities:

- read Supabase credentials from env
- call `export_tiered_manifests()`
- if uploading, push each tiered manifest to S3
- emit `latest.json` with generation timestamp and hashes

Keep script logic thin. Reuse snapshot helpers rather than duplicating export
logic.

### Step 2: Define the upload contract

Artifact layout should be:

- `manifests/manifest-{tier}.sqlite`
- `manifests/latest.json`

`latest.json` should be boring and machine-readable. It only needs enough data
for verification and rollback tooling.

### Step 3: Add workflow file

Create a workflow that is:

- `workflow_dispatch` only
- clearly marked as manual-only until secrets and infrastructure are ready

The workflow should:

- check out the repo
- set up Python
- install the package or the manifest-specific extras needed
- run the export script

### Step 4: Keep enablement explicit

Do not silently imply this is production-ready.

The workflow file should include clear comments about what still needs to exist
before cron or automatic publish is enabled.

## Testing Plan

Add or extend tests for:

- export script CLI argument handling
- environment validation failures being readable
- `latest.json` structure
- upload function call shape via mocking

The workflow file itself does not need unit tests, but it should be inspected
for syntax and path correctness.

## Worker Breakdown

Recommended split:

- Worker E1: `scripts/export_manifest.py` plus unit tests
- Worker E2: workflow YAML once the script interface is stable

This phase parallelizes well after the export script CLI contract is fixed.

## Risks And Decisions

### Install surface in CI

The workflow should install only what the export path needs. Do not assume the
full development environment is necessary.

### Upload safety

If `--upload` is omitted, the script should still fully support local export and
verification. Local dry runs are important before enabling automation.

### Latest metadata

`latest.json` should not become a second schema registry. Keep it compact:

- generated timestamp
- per-tier file names
- per-tier content hashes

## Exit Criteria

Phase E is complete when:

- the repo has a usable export script
- a manual workflow can invoke it
- tiered manifests and `latest.json` can be produced consistently
