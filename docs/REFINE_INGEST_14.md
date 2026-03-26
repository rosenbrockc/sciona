# REFINE_INGEST Phase 14 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 14 adds one real cache-enabled end-to-end ingest test that exercises the
actual `IngesterAgent` cache path across a miss followed by a hit.

The objective is:

- prove that the live cache path in `sciona/ingester/graph.py` behaves
  correctly, not just that cache helper functions round-trip
- verify semantic outputs remain identical across miss and hit
- verify monitor/cache surfaces expose the expected operational state

Key rule:

- this is a narrow integration-confidence phase, not another cache architecture
  phase

## Scope Boundaries

In scope:

- one focused cache-enabled `IngesterAgent` integration test
- explicit assertions about cache miss then hit behavior
- additive monitor/cache assertions if required by the test

Out of scope:

- broad benchmark work
- large cache redesign
- CI wiring

## Current Code Touchpoints

- `sciona/ingester/graph.py`
- `sciona/ingester/cache.py`
- `sciona/ingester/monitor.py`
- likely a new focused integration test file, for example:
  - `tests/test_ingester_graph_cache_integration.py`

## Why This Matters

Phase 12 stabilized:

- cache envelopes
- monitor schemas
- harness cache summaries

But the remaining gap is the real runtime path:

- cache key computed by `IngesterAgent`
- cache miss leading to full ingest and cache write
- cache hit loading a cached bundle and short-circuiting work

Without this test, the system still lacks proof that the end-to-end cached path
works as intended.

## Deliverables

### 1. Miss-Then-Hit Integration Test

The test should:

- construct a small deterministic ingest target
- run with `enable_cache=True`
- confirm the first run is a miss and writes cache state
- confirm the second run is a hit and reuses the cached bundle

### 2. Semantic Equality Assertions

The test should prove that miss and hit produce equivalent semantic output,
such as:

- generated atom source
- witness source
- state-model source if relevant
- CDG or match-result shape

### 3. Operational Assertions

The test should assert whichever operational signal is already available or can
be exposed with a small additive change:

- cache file written
- marker/status cache state
- trace/monitor evidence if appropriate

## Required Interfaces

This phase builds directly on phase 12:

- keep the new cache envelope as-is unless a tiny additive hook is needed
- keep monitor/status schema stable

## Deterministic vs LLM Responsibilities

Deterministic:

- test setup
- miss/hit execution
- semantic equality assertions
- cache/monitor checks

LLM:

- none

## Rollout Plan

### Step 0. Pick the Narrowest Stable Target

- use a small deterministic fixture already friendly to ingest
- avoid introducing external dependency noise

### Step 1. Exercise Real Cache Miss

- run a real ingest through `IngesterAgent(enable_cache=True)`
- assert cache artifact creation

### Step 2. Exercise Real Cache Hit

- rerun the same ingest request
- assert the bundle is restored from cache and semantically equivalent

### Step 3. Assert Operational Surface

- verify the intended cache-state signal is exposed
- keep any new hook additive and small

## Concrete File Plan

- `sciona/ingester/graph.py` only if a tiny additive cache-state hook is needed
- likely a new test such as `tests/test_ingester_graph_cache_integration.py`
- possibly small additive assertions in `tests/test_ingest_monitor.py`

## Regression Risks

- integration setup becomes flaky or too slow
- the test asserts an unstable incidental detail rather than actual behavior
- cache-state signaling leaks semantic assumptions into monitor surfaces

## Test Plan

Minimum local slice:

- the new cache-enabled integration test
- `pytest -q tests/test_ingester_cache.py tests/test_ingest_monitor.py`

## Acceptance Criteria

- one focused end-to-end cached ingest test exists
- it proves miss then hit behavior
- it proves semantic output equality across the two paths
- it does not require broad cache redesign or flaky infrastructure

## Deferred

- broader cache benchmarks
- broader CI wiring
- multi-family cached integration suites
