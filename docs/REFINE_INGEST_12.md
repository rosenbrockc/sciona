# REFINE_INGEST Phase 12 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 12 hardens the operational surfaces around the canonical-first ingest
runtime: cache payloads, serialization contracts, monitor/artifact schemas, and
lightweight performance measurement over the protected regression corpus.

After phase 11, semantic correctness and regression visibility are materially
better, but operational stability is still softer than it should be:

- cache payloads still only store `IngestionBundle`, not the broader
  canonical-first runtime context that now drives execution
- monitor and artifact files are useful, but their schema is only partially
  stabilized for long-term harness/tool consumers
- the harness measures runtime duration, but there is no explicit phase-12
  baseline for cache behavior, artifact stability, or representative runtime
  costs

The objective is:

- make cache behavior explicit and versioned for canonical-first runtime use
- define which monitor/artifact fields are stable tooling surfaces
- add lightweight, deterministic performance summaries over representative
  slices
- protect these operational surfaces with focused tests

Key rule:

- phase 12 is a stability and measurement phase, not a semantic redesign; it
  must not change canonical ingest meaning or reintroduce legacy-first runtime
  assumptions

## Scope Boundaries

In scope:

- cache-format hardening and versioning for canonical-first runtime artifacts
- review and tightening of cache serialization/deserialization behavior
- monitor and artifact schema stabilization for harness/tool consumers
- regression-harness summaries for cache usage and representative runtime
  numbers
- focused compatibility tests for cache, monitor, and artifact schemas

Out of scope:

- changing the core semantic model or canonical planning/emission rules
- large hosted telemetry or observability systems
- broad CLI/product redesign
- aggressive performance optimization work unrelated to measured bottlenecks

## Current Code Touchpoints

Primary implementation surfaces:

- `sciona/ingester/cache.py`
- `sciona/ingester/graph.py`
- `sciona/ingester/monitor.py`
- `sciona/ingester/regression_harness.py`

Supporting surface if needed:

- `sciona/ingester/models.py`

Existing tests and regression surfaces that should anchor the work:

- `tests/test_ingest_monitor.py`
- `tests/test_ingest_regression_harness.py`
- the phase-11 protected-family harness corpus under
  `tests/fixtures/ingest_regression/` and `tests/golden/ingest_regression/`
- representative ingest slices such as:
  - `tests/test_ingest_stateful.py`
  - `tests/test_ingest_procedural.py`
  - `tests/test_ingester_emitter.py`
  - non-Python slices as needed

Likely new or expanded tests:

- a new cache-focused test file such as `tests/test_ingester_cache.py`
- additive monitor schema tests in `tests/test_ingest_monitor.py`
- additive harness metric/stability tests in `tests/test_ingest_regression_harness.py`

## Current Gaps

### 1. Cache Format Is Too Narrow and Implicit

Current cache behavior in `sciona/ingester/cache.py`:

- keys are versioned, but the payload only stores `IngestionBundle`
- cache data is not clearly separated into stable schema vs incidental
  implementation detail
- canonical IR/planning context is not explicitly represented in cache payloads,
  even though runtime is now canonical-first

This is workable, but it leaves too much ambiguity around what a cache hit is
allowed to guarantee.

### 2. Artifact and Monitor Schemas Are Useful but Soft

Current monitor/artifact behavior in `sciona/ingester/monitor.py` and the
harness:

- status, completed, failed, and trace files are emitted deterministically
- phase 11 normalized them for golden comparison
- there is still no clearly enforced boundary between stable tooling fields and
  best-effort runtime detail

That increases the risk of accidental schema churn breaking tooling or forcing
frequent golden updates for non-semantic reasons.

### 3. Runtime Cost Visibility Is Limited

The harness records case runtime and prompt counts, but phase 12 still lacks:

- stable benchmark summaries for the curated corpus
- explicit cache-hit/cache-miss visibility
- a compact way to compare operational behavior across representative families

The next stability phase should make those signals first-class without turning
the harness into a heavyweight benchmark service.

## Phase 12 Deliverables

### 1. Versioned Canonical-Aware Cache Payloads

Phase 12 should make cache behavior more explicit and future-proof.

Recommended direction:

- introduce an explicit cache envelope with schema version metadata
- separate stable top-level cache fields from the serialized payload content
- preserve existing bundle restore behavior while clarifying what is guaranteed
  on load
- include enough metadata for tooling/tests to assert compatibility without
  depending on unstable internals

If canonical runtime context is added to the cache, it should be done
conservatively:

- additive only
- version-gated
- not required to change current semantic outputs

### 2. Stable Artifact/Monitor Schema Boundaries

Define which fields in status/trace/marker/artifact outputs are intended to be
stable for harness/tool consumers.

Good targets:

- explicit monitor schema helpers or normalized read APIs
- more stable marker payload shape in `COMPLETED.json` / `FAILED.json`
- stable artifact-summary surfaces used by the harness

The important constraint is that future tooling should be able to rely on these
fields without scraping ad hoc payload structures.

### 3. Lightweight Performance and Cache Metrics in the Harness

Extend the phase-11 harness so it can summarize operational stability, not just
semantic correctness.

Recommended additions:

- cache-hit/cache-miss visibility where the harness exercises cached paths
- compact runtime summaries per family and across the corpus
- benchmark-friendly exported metrics that do not depend on wall-clock noise
  being exact

This should stay lightweight:

- no hosted dashboard
- no flaky performance thresholds
- no broad benchmarking framework

### 4. Compatibility and Migration Tests

Phase 12 should add tests that make schema and cache drift visible early.

Priority coverage:

- cache save/load round-trip compatibility
- tolerant loading of older or partial cache payloads where intentionally
  preserved
- monitor marker/status payload shape checks
- harness summary checks for cache/perf metadata

### 5. Protected-Family Operational Baseline

Use the phase-11 regression corpus as the representative operational baseline.

The worker should ensure:

- protected-family cases remain runnable under the updated cache/monitor rules
- benchmark and cache summaries can be computed from the same curated corpus
- no family requires bespoke tooling behavior just to participate in the stable
  phase-12 surfaces

## Required Interfaces With Prior Phases

Interface from phases 9 through 11:

- canonical IR/planning remains the runtime source of truth
- compatibility exports stay narrow and explicit
- phase 11 already established a curated protected-family corpus and normalized
  artifact comparisons

Phase 12 should build on those surfaces rather than inventing a separate
benchmark or schema path.

Interface to later work:

- any future cache or tooling consumer should be able to depend on the phase-12
  schema/version boundaries
- future performance work should start from phase-12 measured slices and stable
  harness summaries rather than ad hoc one-off timings

## Deterministic vs LLM Responsibilities

Deterministic in phase 12:

- cache envelope/schema design
- serialization/deserialization tightening
- monitor schema normalization/stabilization
- harness metric aggregation
- compatibility tests and benchmark summaries

LLM responsibilities in phase 12:

- none for the core work
- no new prompt dependency should be introduced

## Data Model Changes

Expected additive work:

- a versioned cache envelope or manifest in `cache.py`
- optional additive cache metadata fields for canonical-runtime provenance
- additive monitor summary helpers or normalized schema readers
- richer harness result/summary fields for cache/performance metrics

Avoid:

- changing core canonical semantic models just to simplify caching
- duplicating large canonical runtime objects unless there is a clear stability
  benefit
- serializing unstable internal state that the runtime does not actually need

## Rollout Plan

### Step 0. Lock the Existing Regression and Monitor Slices

Before changing cache or schema behavior, preserve:

- phase-11 harness golden coverage
- monitor lifecycle tests
- representative stateful/procedural slices

### Step 1. Introduce a Stable Cache Envelope

- add an explicit schema/version wrapper around cache payloads
- keep loads tolerant enough for the currently supported cache shape
- add tests proving versioned round-trip compatibility

### Step 2. Stabilize Monitor and Marker Read Surfaces

- identify stable fields in status/completed/failed/trace artifacts
- add helper readers or schema normalizers if needed
- test the intended payload contract directly

### Step 3. Add Harness Cache/Perf Summaries

- extend harness results to record cache usage and lightweight runtime metrics
- aggregate them by family and overall suite
- keep metrics descriptive, not threshold-enforcing

### Step 4. Exercise Representative Cached Paths

- add focused tests proving save/load behavior works with the current graph path
- ensure representative protected-family cases still run under the updated
  surfaces

### Step 5. Keep the Surface Reviewable

- keep schema and cache metadata small and explicit
- avoid adding large unstable payloads just because they are available
- document what is intentionally stable versus best-effort

## Concrete File Plan

Expected implementation edits:

- `sciona/ingester/cache.py`
- `sciona/ingester/monitor.py`
- `sciona/ingester/regression_harness.py`
- possibly `sciona/ingester/graph.py`

Likely tests:

- `tests/test_ingest_monitor.py`
- `tests/test_ingest_regression_harness.py`
- new `tests/test_ingester_cache.py`
- small representative regression slice tests if graph/cache integration needs
  direct coverage

## Regression Risks

Primary risks:

- cache versioning breaks existing load paths instead of degrading safely
- operational metadata gets conflated with semantic artifacts and causes noisy
  regression churn
- runtime measurements become too flaky to be useful in CI/local runs
- graph integration accidentally changes ingest semantics while refactoring
  cache boundaries

Mitigations:

- make new cache fields additive and versioned
- keep performance summaries descriptive rather than brittle threshold gates
- test both fresh and legacy-like payload loading behavior
- keep semantic regression coverage running alongside the new operational tests

## Test Plan

Minimum coverage for phase 12 should include:

- cache envelope round-trip tests
- tolerant load tests for older/minimal payloads
- monitor status/marker schema tests
- harness summary tests for cache/perf metadata
- representative integration tests touching cached ingest paths

Recommended local regression slice:

- `pytest -q tests/test_ingest_monitor.py tests/test_ingest_regression_harness.py`
- `pytest -q tests/test_ingester_cache.py`
- `pytest -q tests/test_ingest_stateful.py tests/test_ingest_procedural.py`

If graph/cache integration requires it, add one focused end-to-end cache test
rather than broadening the entire ingest suite.

## Acceptance Criteria

Phase 12 is complete when:

- cache behavior is explicitly versioned and stable enough for canonical-first
  runtime use
- monitor/artifact schemas expose a clear stable surface for harness/tool
  consumers
- representative runtime/cache metrics are available from the curated corpus
- focused compatibility tests cover cache and schema drift
- no protected-family regressions are introduced

## Deferred To Later Work

Explicitly defer:

- broad runtime optimization campaigns
- hosted dashboards or persistent benchmarking infrastructure
- changing canonical semantic content to make caching easier
- unrelated CLI/product work
