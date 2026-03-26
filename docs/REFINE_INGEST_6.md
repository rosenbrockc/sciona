# REFINE_INGEST Phase 6 Implementation Plan

## Phase Goal

Phase 6 makes the refined ingest pipeline observable across representative atom
families so future semantic improvements can be measured instead of guessed.

The objective is not to add a heavyweight benchmark framework. It is to:

- lock a curated regression matrix spanning protected ingest families
- run those cases through the existing ingest pipeline and monitor artifacts
- summarize completion, verification, timeout/fallback, and runtime behavior
- make sklearn-focused improvements provably non-regressive for the rest of the
  repository

This phase should land as a lightweight harness first. It is the safety rail
for phases 1 through 5, not a redesign of them.

## Scope Boundaries

In scope:

- a deterministic regression harness over curated ingest cases
- a case/result schema capturing ingest outcomes and key metrics
- fixture-backed coverage for the required family slices
- aggregation of existing monitor/verification/fallback signals
- golden assertions for curated semantic expectations on selected cases
- focused tests for harness collection and summary behavior

Out of scope:

- a large standalone benchmarking service or dashboard
- new semantic planner/emitter behavior beyond small metric hooks
- live external dependency downloads just to exercise sklearn
- broad CLI redesign outside a small optional entrypoint
- replacing the existing unit/integration tests with the harness

Key rule:

- phase 6 should measure the current pipeline with representative cases, not
  invent a second ingest stack

## Current Code Touchpoints

Primary runtime surfaces:

- `sciona/ingester/graph.py`
- `sciona/ingester/monitor.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/verification_classifier.py`

Existing tests and fixtures that already cover protected families:

- `tests/test_ingester_extractor.py`
- `tests/test_ingester_chunker.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_biosppy_ecg.py`
- `tests/test_bayesian_ingester.py`
- `tests/test_message_passing.py`
- `tests/test_ingest_dl_boundary.py`
- `tests/test_ingest_procedural.py`
- `tests/test_treesitter_rust.py`
- `tests/test_treesitter_cpp.py`
- `tests/test_treesitter_julia.py`
- `tests/test_ffi_emitter.py`
- `tests/test_ingester_e2e.py`
- `tests/test_ingest_monitor.py`

Recommended new implementation surface:

- `sciona/ingester/regression_harness.py`
- optionally `tests/test_ingest_regression_harness.py`

Optional small touchpoints for metric hooks:

- `sciona/ingester/models.py`
- `sciona/ingester/cache.py`

## Current Gaps

The repository has strong targeted tests, but it still lacks one place that
answers:

- did the full ingest path complete for each protected family?
- how many LLM fallbacks were used?
- did mypy and ghost verification pass?
- did canonical-IR and fail-fast behavior reduce semantic drift without harming
  non-sklearn cases?
- did timeout or retry behavior regress?

Current problems:

- phase-specific tests prove local behavior, but not family-level regression
- monitor artifacts are written per run, but not summarized across a curated
  matrix
- fallback volume and runtime are observable in traces, but not normalized into
  harness metrics
- semantic quality on curated cases is spread across individual tests instead
  of one benchmark slice

## Required Regression Matrix

Phase 6 should lock a small curated matrix with at least these slices:

- one sklearn-style estimator class
  - use an in-repo fixture modeled on `CalibratedClassifierCV`
  - avoid a live sklearn dependency in the first cut
- one flat scientific function/module
  - use an existing NumPy/SciPy-style fixture or similar simple wrapper case
- one stateful rolling/windowed class
  - reuse the stateful ingest fixtures already covered in
    `tests/test_ingest_stateful.py`
- one Bayesian or message-passing example
  - include either the Bayesian estimator slice or message-passing slice
- one non-Python / FFI example
  - use a tree-sitter Rust/Julia/C++ or FFI-backed emitter case already in repo
- one procedural-ingest example
  - reuse `tests/test_ingest_procedural.py` style fixtures

Recommended first-cut case ids:

- `sklearn_style_estimator`
- `flat_scientific_function`
- `rolling_stateful_class`
- `bayesian_or_message_passing`
- `non_python_ffi`
- `procedural_ingest`

## Harness Data Model

Phase 6 should add a small additive result schema. It should be easy to dump as
JSON and stable enough for golden assertions.

Recommended models:

- `IngestRegressionCase`
  - `case_id`
  - `family`
  - `source_path`
  - `class_name`
  - `procedural`
  - `expected_language`
  - `semantic_expectations`
- `IngestRegressionResult`
  - `case_id`
  - `family`
  - `completed`
  - `failed_phase`
  - `timed_out_or_stalled`
  - `mypy_passed`
  - `ghost_passed`
  - `type_failure_reason`
  - `ghost_failure_reason`
  - `llm_call_count`
  - `llm_prompt_counts`
  - `runtime_ms`
  - `published_artifacts`
  - `semantic_checks`
- `IngestRegressionSummary`
  - `total_cases`
  - `completed_cases`
  - `completion_rate`
  - `mypy_pass_rate`
  - `ghost_pass_rate`
  - `timeout_or_stall_count`
  - `llm_call_total`
  - `family_breakdown`
  - `failures`

Keep this additive. Do not thread it through core ingest models unless a small
shared helper materially simplifies implementation.

## Metrics To Track

Phase 6 should report the metrics requested in the source plan:

- pipeline completion rate
- timeout/fallback behavior
- mypy pass rate
- ghost sim pass rate
- semantic review quality for curated examples
- number of LLM calls and total runtime

For the lightweight first cut, define them concretely as:

- completion rate
  - case ends with a bundle or explicit fail artifact, without hanging
- timeout/fallback behavior
  - monitor classified stall count
  - prompt-key LLM call counts
  - fallback prompt counts by key where trace/metadata exposes them
- mypy pass rate
  - `bundle.mypy_passed` or recorded verification result
- ghost sim pass rate
  - verification result after emission
- semantic review quality
  - deterministic case-specific assertions over canonical IR, emitted source, or
    failure classification
- total runtime
  - wall-clock duration per case from harness start/end, not inferred from only
    one phase

## Semantic Review Strategy

Phase 6 should not add an LLM grader.

Semantic quality for the curated matrix should be captured via deterministic
checks such as:

- sklearn-style estimator
  - fit/query/predict wrappers call real upstream methods
  - config slots and fitted slots remain distinct
  - prediction/query wrappers rehydrate required state
  - no invented output bindings or fitted attrs
- flat scientific function
  - no unnecessary state model introduced
  - wrapper remains direct and low-ceremony
- rolling/stateful class
  - mutable state threading still works
  - config flattening remains intact
- Bayesian/message-passing case
  - current stochastic/message specialization still passes
  - deterministic repair routing remains intact where applicable
- non-Python / FFI case
  - Python-centric semantic changes do not break extraction/emission path
- procedural case
  - procedural builder path remains functional and low-friction

These checks should be expressed as deterministic predicates or golden
assertions, not prose review.

## Required Interfaces With Other Phases

Interface from phases 1 and 2:

- snapshot semantic facts and canonical IR on curated cases where available
- confirm the legacy adapter path still works during transition

Interface from phase 3:

- capture planner decisions and LLM fallback counts on decomposition-heavy cases

Interface from phase 4:

- assert canonical emission uses exact signatures/output bindings
- treat fail-closed emission on underspecified canonical IR as an explicit,
  reportable outcome

Interface from phase 5:

- capture verification pass rate
- capture deterministic repair count vs fail-fast semantic failure count
- make LLM repair count visible and expected to remain near zero

Interface to later work:

- future semantic changes should add or update harness cases before broadening
  heuristics
- if a future phase changes family behavior intentionally, the harness should
  make that delta explicit in one place

## Deterministic vs LLM Responsibilities

Deterministic in phase 6:

- case definition
- harness execution
- artifact parsing
- trace/monitor summarization
- metric aggregation
- semantic expectation checks
- pass/fail reporting

LLM responsibilities in phase 6:

- none in the harness itself
- the harness may observe LLM calls made by the ingest pipeline, but it should
  never require an LLM to interpret benchmark outcomes

## Rollout Plan

### Step 0. Lock the Curated Case Matrix

- define the six required benchmark slices as explicit harness cases
- prefer existing in-repo fixtures and synthetic sklearn-style sources
- keep the first cut small enough to run in CI without becoming a new soak test

### Step 1. Add a Lightweight Harness Module

- add `sciona/ingester/regression_harness.py`
- implement case loading, one-case execution, and suite aggregation
- keep execution a thin wrapper around existing ingest entrypoints

Recommended entrypoints:

- `run_ingest_regression_case(...)`
- `run_ingest_regression_suite(...)`
- `summarize_ingest_regression_results(...)`

### Step 2. Collect Runtime and Verification Metrics

- use wall-clock timing in the harness
- read completion/failure status from returned bundle and monitor artifacts
- count LLM calls from monitor trace events when a monitor is attached
- normalize verification and failure-classification outputs into result records

### Step 3. Add Deterministic Semantic Checks

- encode curated expectations per case family
- start with a narrow set of high-signal checks
- fail clearly when emitted wrappers or failure routing contradict expectations

### Step 4. Add Focused Tests

- add harness unit tests for metric aggregation and trace parsing
- add integration-style tests running a small curated suite over existing
  fixtures
- assert family coverage and stable summary fields

### Step 5. Optional Thin CLI Surface

- only if useful, add a minimal command or developer script that runs the
  curated suite and writes a JSON report
- do not block phase 6 on a large CLI redesign

## Concrete File Plan

Expected edits:

- `sciona/ingester/regression_harness.py`
  - new harness models, execution helpers, summary logic
- `sciona/ingester/monitor.py`
  - only if a small helper is needed to summarize trace events cleanly
- `sciona/ingester/graph.py`
  - only if a small hook is needed to expose stable summary data
- tests
  - `tests/test_ingest_regression_harness.py`
  - optionally small additions to `tests/test_ingest_monitor.py`
  - optionally small additions to representative ingest-family tests to expose
    reusable fixtures

Possible optional additions:

- `scripts/run_ingest_regression.py`
- `benchmarks/` script glue only after the core harness works

## Regression Risks

Primary risks:

- the harness becomes too large or flaky to run regularly
- benchmark cases accidentally depend on live third-party libraries
- metric collection couples too tightly to monitor internals and becomes brittle
- semantic checks become so specific that harmless refactors look like failures
- one family dominates the harness and hides regressions in another

Mitigations:

- keep the curated matrix intentionally small
- prefer synthetic or existing local fixtures over networked/live dependencies
- treat monitor parsing as additive and tolerant of missing optional fields
- keep semantic checks focused on non-negotiable contract properties
- require at least one case per protected family slice

## Test and Benchmark Plan

Harness unit tests:

- trace parsing counts prompt-key LLM calls correctly
- suite summary computes completion and verification rates correctly
- stalled/failed monitor states map to harness result fields correctly
- missing optional artifacts degrade gracefully instead of crashing

Harness integration slice:

- run a curated suite with:
  - sklearn-style estimator fixture
  - rolling/stateful fixture
  - Bayesian or message-passing fixture
  - non-Python / FFI fixture
  - procedural fixture
- assert result records contain stable family tags and metrics

Protected regression checks to keep in the same phase-6 test run:

- `tests/test_ingester_extractor.py`
- `tests/test_ingester_chunker.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_bayesian_ingester.py`
- `tests/test_message_passing.py`
- `tests/test_ingest_dl_boundary.py`
- `tests/test_ingest_procedural.py`
- at least one tree-sitter or FFI test

## Acceptance Criteria

Phase 6 is complete when all of the following are true:

- a curated regression harness exists in-repo and runs the required family
  slices
- each result captures completion, verification, timeout/fallback, LLM-call,
  and runtime metrics
- semantic expectations are asserted deterministically on curated cases
- the harness reuses existing ingest entrypoints and monitor artifacts rather
  than inventing a parallel pipeline
- protected non-sklearn families are visible in the same summary as the
  sklearn-style case
- the harness is small and stable enough to run as a regular regression slice

## Deferred to Later Work

Not required in the first phase-6 implementation:

- a large benchmark dashboard
- historical trend storage across many runs
- live sklearn package benchmarking
- broad cross-repo benchmark orchestration
- exhaustive performance tuning of all ingest families

Those can follow once the lightweight curated harness is reliable.

## Recommended Execution Order

1. Lock the curated case matrix and result schema.
2. Add the lightweight harness module and summary helpers.
3. Wire in monitor/trace-derived metrics without changing ingest semantics.
4. Add deterministic semantic checks for the curated cases.
5. Add focused tests and keep the protected-family regression slice green.
