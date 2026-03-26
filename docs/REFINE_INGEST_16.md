# REFINE_INGEST Phase 16 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 16 produces a maintainer-oriented architecture note for the canonical
ingest system.

The objective is:

- document the canonical ingest data flow end-to-end
- explain where canonical truth lives and where compatibility exports begin
- explain how verification, regression, cache, and monitor surfaces fit
  together
- make future maintenance and onboarding materially easier

Key rule:

- this is an architecture communication phase, not a runtime refactor phase

## Scope Boundaries

In scope:

- a maintainer-facing architecture doc
- concrete file/module touchpoint references
- operational guidance for regression, cache, and monitor surfaces

Out of scope:

- changing runtime behavior
- adding new semantic machinery
- broad documentation overhaul unrelated to ingest

## Current Code Touchpoints

Primary modules the doc should explain:

- `sciona/ingester/extractor.py`
- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/graph.py`
- `sciona/ingester/regression_harness.py`
- `sciona/ingester/cache.py`
- `sciona/ingester/monitor.py`

Likely output location:

- a new maintainer doc under `docs/`, for example
  `docs/INGEST_ARCHITECTURE.md`

## Why This Matters

The architecture is much cleaner after phases 1-12, but the conceptual model is
spread across multiple modules. A clear maintainer note would reduce:

- re-discovery cost
- accidental architectural regressions
- confusion around compatibility/export boundaries

## Deliverables

### 1. Canonical Data-Flow Description

Explain:

- extraction facts
- canonical IR lowering
- planning/decomposition
- emission
- verification and repair boundaries

### 2. Compatibility Boundary Description

Explain:

- what legacy compatibility exports still exist
- what they are for
- what is no longer authoritative

### 3. Operational Surface Description

Explain:

- regression harness
- golden snapshots
- cache envelopes
- monitor status/marker/surface schemas

### 4. Maintainer Change Guidance

Explain where future changes should go and what must stay true:

- what may evolve semantically
- what is operationally stable
- what tests should be touched when changing a given layer

## Dependencies

This can run in parallel with phase 13, 14, and 17. It should simply reflect
the current architecture accurately by the time it lands.

## Deterministic vs LLM Responsibilities

Deterministic:

- documentation writing
- architecture synthesis from current code and tests

LLM:

- none required beyond ordinary writing assistance

## Rollout Plan

### Step 0. Read The Current Canonical Path

- verify the architecture against code, not historical assumptions

### Step 1. Draft The Architecture Note

- describe the layers in execution order

### Step 2. Add Maintainer Guidance

- include “where to change what” guidance and protected invariants

## Concrete File Plan

- likely a new doc such as `docs/INGEST_ARCHITECTURE.md`

## Regression Risks

- the doc becomes stale immediately if it is too implementation-specific
- the doc glosses over remaining compatibility realities

## Test / Verification Plan

- no code tests required unless the doc references commands that should be
  sanity-checked
- verify references and statements against current modules/tests

## Acceptance Criteria

- one maintainer architecture note exists
- it accurately describes canonical ingest flow
- it explains compatibility and operational surfaces
- it is useful for future maintainers without requiring them to read twelve old
  phase plans

## Deferred

- broad docsite restructuring
- user-facing documentation unrelated to maintainers
