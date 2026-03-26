# REFINE_INGEST Phase 15 Implementation Plan

## Phase Goal

Phase 15 wires the curated ingest regression corpus into CI deliberately,
rather than allowing it to remain a purely local/manual protected suite.

The objective is:

- make the corpus a real gate in a practical way
- separate fast always-on coverage from broader or less-frequent coverage
- define a clear workflow for intentional golden updates

Key rule:

- CI wiring should reflect the stabilized runtime and test slices, not become a
  forcing function for premature suite expansion

## Scope Boundaries

In scope:

- GitHub Actions or existing CI workflow changes
- fast and broader suite split
- explicit golden-update guidance in developer-facing docs or workflow comments

Out of scope:

- hosted dashboards
- broad product/CLI changes
- large test refactors unrelated to CI execution

## Current Code Touchpoints

- `.github/workflows/release-validation.yml`
- possibly a new dedicated ingest workflow file under `.github/workflows/`
- `tests/test_ingest_regression_harness.py`
- `tests/golden/ingest_regression/`
- `sciona/ingester/regression_harness.py`

## Why This Matters

The curated corpus and golden snapshots are now valuable enough to protect
regressions centrally, but they should be introduced carefully:

- always-on CI should stay fast and reliable
- broader coverage can run less frequently
- golden updates should be intentional rather than surprising

## Deliverables

### 1. Fast Always-On Slice

A lightweight slice that runs on normal PR validation, likely centered on:

- harness logic
- a representative subset of regression/golden checks
- adjacent chunker/emitter/stateful/procedural coverage if needed

### 2. Broader Scheduled Or Manual Slice

A second slice that runs:

- on schedule
- on manual dispatch
- or on selected branches/events

This can cover a wider protected-family set without slowing normal iteration.

### 3. Golden Update Workflow

Document or encode:

- when goldens are expected to change
- how maintainers should review those changes
- which suite should catch accidental drift

## Dependencies

Phase 15 should follow:

- phase 13 cleanup enough to settle transitional code churn
- phase 14 cache-enabled integration test so the protected CI surface reflects
  the more complete runtime confidence story

## Deterministic vs LLM Responsibilities

Deterministic:

- CI workflow edits
- suite selection
- documentation/comments for golden updates

LLM:

- none

## Rollout Plan

### Step 0. Choose The Stable Default Slice

- identify the subset safe for always-on execution
- keep runtime and flakiness practical

### Step 1. Add Broader Triggered Coverage

- add scheduled/manual/broader workflow coverage

### Step 2. Document Golden Expectations

- make the golden-review workflow explicit in CI comments, docs, or both

## Concrete File Plan

- `.github/workflows/release-validation.yml` or a new workflow file
- possibly a short supporting doc if needed

## Regression Risks

- always-on CI becomes too slow
- golden churn becomes noisy
- broader slice is rarely used and silently rots

## Test Plan

- validate workflow syntax
- run the intended fast local test slice
- if possible, dry-run or at least inspect workflow commands directly

## Acceptance Criteria

- the corpus is represented in CI deliberately
- there is a fast always-on slice
- there is a broader triggered slice
- golden update expectations are explicit

## Deferred

- dashboards
- broad infra redesign
- further suite expansion beyond the curated corpus
