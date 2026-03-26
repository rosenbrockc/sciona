# REFINE_INGEST Phase 17 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 17 defines the stable public contract for post-phase-12 ingest tooling
surfaces.

The objective is:

- be explicit about which schemas, artifacts, and exports are intended to stay
  stable
- separate true contracts from best-effort implementation detail
- reduce future ambiguity for tooling, tests, and maintainers

Key rule:

- this is a contract-definition phase, not a semantic redesign phase

## Scope Boundaries

In scope:

- documenting stable contract surfaces
- identifying best-effort or intentionally unstable fields
- tightening code comments or docs where needed to match that contract

Out of scope:

- major runtime changes
- broad product-facing API design
- hosted tooling or dashboard work

## Current Code Touchpoints

Likely primary surfaces:

- `sciona/ingester/cache.py`
- `sciona/ingester/monitor.py`
- `sciona/ingester/regression_harness.py`
- `sciona/ingester/models.py`
- documentation under `docs/`

Likely outputs:

- a stable-contract doc under `docs/`
- possibly small inline comments or schema notes in code

## Why This Matters

Phase 12 made operational surfaces more stable, but teams still need an
explicit answer to:

- which cache fields are contractually stable
- which monitor marker/status/surface fields are stable
- which golden artifact formats are stable
- which canonical runtime exports other tooling may rely on

Without that, future changes risk accidental contract drift.

## Deliverables

### 1. Contract Inventory

Inventory the key tooling surfaces:

- cache envelopes
- monitor status/marker/surface outputs
- regression golden artifact formats
- canonical runtime export surfaces, if any are relied on externally

### 2. Stability Classification

For each surface, classify fields as:

- stable contract
- best-effort informational
- internal implementation detail

### 3. Documentation And Code Alignment

Where helpful, add small code comments or doc notes so the intended contract is
not only captured in prose.

## Dependencies

This can run in parallel with phase 13, 14, and 16. It should take the current
phase-12 operational surfaces as its baseline.

## Deterministic vs LLM Responsibilities

Deterministic:

- contract inventory
- field classification
- documentation/code-note alignment

LLM:

- none required beyond ordinary writing assistance

## Rollout Plan

### Step 0. Inventory Current Surfaces

- list every relevant schema/artifact boundary

### Step 1. Classify Stability

- define what is stable and what is not

### Step 2. Align Docs And Code Notes

- add explicit notes where ambiguity currently exists

## Concrete File Plan

- likely one or more docs under `docs/`
- implemented contract note:
  - `docs/INGEST_TOOLING_CONTRACT.md`
- possibly small comments in:
  - `sciona/ingester/cache.py`
  - `sciona/ingester/monitor.py`
  - `sciona/ingester/regression_harness.py`

## Regression Risks

- contract claims overreach what the code can actually support
- best-effort details get accidentally frozen as contracts

## Test / Verification Plan

- mainly doc and code-review verification
- if comments/docs imply schema shapes, cross-check against existing tests

## Acceptance Criteria

- stable public contract surfaces are explicitly documented
- intentionally unstable details are clearly called out
- future maintainers/tooling authors can tell what they may rely on

## Deferred

- broader public API design
- product-facing support guarantees beyond the current tooling surfaces
