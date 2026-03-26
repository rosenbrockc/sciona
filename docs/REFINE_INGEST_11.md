# REFINE_INGEST Phase 11 Implementation Plan

## Phase Goal

Phase 11 broadens the ingest regression harness from a lightweight synthetic
smoke suite into a curated, reviewable corpus built from real repository
targets and stable golden semantic artifacts.

After phase 10, the runtime is canonical-first and the main remaining gap is
confidence breadth rather than another semantic redesign. The goal here is to
make protected-family regressions visible from one place, with artifact-level
evidence that reviewers can inspect when semantics change.

The objective is:

- expand the curated harness matrix from mostly inline synthetic examples to a
  small set of real repository targets
- snapshot canonical artifacts that matter for semantic review
- make harness results useful both as executable regression checks and as
  reviewable golden outputs
- keep the harness deterministic and cheap enough to run regularly

Key rule:

- phase 11 is a hardening and reviewability phase, not a new semantic-inference
  phase; it must consume the canonical-first runtime that already exists

## Scope Boundaries

In scope:

- expanding `sciona/ingester/regression_harness.py`
- defining a curated real-world regression corpus across protected families
- adding stable golden snapshot capture and comparison
- adding harness-side artifact normalization so snapshots stay reviewable
- adding test coverage for snapshot generation, comparison, and summary output
- small additive helpers in monitor/cache only if required for stable harness
  artifact publication

Out of scope:

- a hosted dashboard or benchmark service
- broad performance optimization work
- live dependency downloads for every harness run
- redesigning extraction, planning, emission, or verification semantics again
- large CLI/product changes unrelated to the harness

## Current Code Touchpoints

Primary implementation surfaces:

- `sciona/ingester/regression_harness.py`
- `tests/test_ingest_regression_harness.py`

Supporting surfaces that may need small additive work:

- `sciona/ingester/monitor.py`
- `sciona/ingester/graph.py`
- `sciona/ingester/cache.py`

Representative source/test families that should seed the corpus:

- `tests/test_ingest_stateful.py`
- `tests/test_bayesian_ingester.py`
- `tests/test_message_passing.py`
- `tests/test_ingest_biosppy_ecg.py`
- `tests/test_treesitter_rust.py`
- `tests/test_treesitter_cpp.py`
- `tests/test_treesitter_julia.py`
- `tests/test_ffi_emitter.py`
- `tests/test_ingest_procedural.py`

Likely new test fixture/golden surfaces:

- `tests/fixtures/ingest_regression/`
- `tests/golden/ingest_regression/`

## Current Gaps

Phase 6 introduced a useful harness, but it is still intentionally light.

Observed limitations:

- default cases are mostly inline source snippets instead of real repo targets
- semantic expectations are narrow booleans or simple string containment checks
- there is no durable golden baseline for canonical IR, planning graph, or
  emitted outputs
- reviewers still need to infer whether a semantic change is acceptable by
  reading unit tests instead of comparing normalized artifacts
- failure-artifact publication exists, but the harness does not yet snapshot it
  as part of a stable regression corpus

This means the architecture can still regress in real protected families while
the current harness stays green.

## Phase 11 Deliverables

### 1. Curated Real-World Corpus

Replace or augment the synthetic default matrix with a corpus built from small,
checked-in repository sources that represent the protected ingest families.

At minimum the corpus should contain:

- one sklearn-style estimator class
- one real rolling/stateful class
- one Bayesian or message-passing target
- one DSP or biosignal target
- one non-Python / FFI target
- one procedural target

Recommended selection rule:

- prefer sources already present in this repository or already used in tests
- keep each case small enough to run in regular CI/local workflows
- avoid cases whose correctness depends on large external downloads or flaky
  runtime environments

### 2. Golden Semantic Artifacts

The harness should produce normalized, reviewable artifact snapshots per case.

Priority snapshot set:

- canonical IR
- planning graph
- emitted wrapper source
- witness source or normalized witness signatures
- CDG metadata
- verification failure artifact payloads when the case is expected to fail

The snapshot format should be deterministic and stable under incidental noise.
That means the worker should normalize or omit:

- run-specific timestamps
- output-root-specific absolute paths
- ordering that is semantically irrelevant
- transient monitor-only fields that are not meaningful for semantic review

### 3. Golden Comparison Workflow

Phase 11 should make it obvious when a semantic change is intentional versus a
regression.

Recommended workflow:

- harness can materialize a snapshot bundle for each case
- tests compare observed normalized artifacts to checked-in goldens
- diffs should be attributable to canonical semantic changes, not incidental
  monitor noise
- summary output should name which artifact classes changed for a failing case

The worker does not need to build a full CLI, but the code should make it easy
to add snapshot-update tooling later.

### 4. Stronger Summary and Failure Reporting

Extend harness results and summaries so they are useful for one-shot review.

Useful additions:

- whether a case matched all goldens
- which artifact groups were compared
- which artifact groups mismatched
- whether a case produced a classified verification failure artifact
- family-level summary of golden mismatches

### 5. Testable Normalization Rules

Normalization logic must itself be under test. The important property is that
goldens capture semantics, not runtime accidentals.

The worker should add tests that prove normalization handles:

- absolute-path stripping
- deterministic ordering
- omitted timestamps / monitor transient fields
- missing optional artifacts without false mismatches

## Required Interfaces With Prior Phases

Interface from phases 6 through 10:

- the harness already runs curated cases through `IngesterAgent`
- canonical IR and planning graph already exist at runtime
- emission already publishes wrappers, witnesses, CDG, and match metadata
- verification failures already publish failure snapshots
- runtime is canonical-first and compatibility exports are now a narrow surface

Phase 11 should reuse those surfaces rather than rebuilding them.

Interface to later work:

- phase 12 should be able to consume the phase-11 corpus as the stable set of
  benchmark and cache/artifact-compatibility targets
- future planners should treat the phase-11 corpus as the main protected-family
  evidence base when making ingest runtime changes

## Deterministic vs LLM Responsibilities

Deterministic in phase 11:

- case selection and fixture materialization
- artifact staging and normalization
- golden snapshot serialization
- snapshot comparison
- summary aggregation and mismatch reporting
- semantic assertions over canonical artifacts

LLM responsibilities in phase 11:

- none in the harness itself
- the harness may still observe LLM usage inside `IngesterAgent`, but phase 11
  must not add new harness-level prompt dependence

## Data Model Changes

Expected additive work:

- richer harness case metadata to describe:
  - fixture source origin
  - expected artifact classes
  - optional expected-failure mode
  - golden snapshot identifier/path
- richer harness result metadata to describe:
  - golden match status
  - mismatched artifact groups
  - compared artifact groups
- helper models for normalized artifact bundles or snapshot manifests

Avoid:

- expanding core ingest semantic models just for the harness
- storing raw unstable monitor payloads as the golden format unless normalized

## Recommended Corpus Layout

Use checked-in repository fixtures rather than ad hoc inline strings wherever
practical.

Recommended layout:

- `tests/fixtures/ingest_regression/<case_id>/source.*`
- `tests/golden/ingest_regression/<case_id>/canonical_ir.json`
- `tests/golden/ingest_regression/<case_id>/planning_graph.json`
- `tests/golden/ingest_regression/<case_id>/atoms.py`
- `tests/golden/ingest_regression/<case_id>/witnesses.py`
- `tests/golden/ingest_regression/<case_id>/cdg.json`
- `tests/golden/ingest_regression/<case_id>/verification_failure.json`
- optional manifest file per case describing which artifacts are expected

If a full source file already exists elsewhere in the repo and is stable,
reusing that file path is preferable to copying it, as long as the harness can
materialize it reproducibly in tests.

## Rollout Plan

### Step 0. Lock the Existing Harness Slice

Before broadening the corpus, keep the existing synthetic coverage for:

- monitor trace summarization
- aggregation math
- minimal case execution

These tests should remain because they isolate harness logic from heavier
real-world fixtures.

### Step 1. Introduce Snapshot Models and Normalization

- add normalized artifact-bundle helpers
- define which artifact files participate in goldens
- normalize paths, timestamps, and unstable ordering
- add direct unit tests for normalization and comparison behavior

### Step 2. Introduce the Real-World Corpus

- add a curated checked-in corpus across the required families
- update the default suite or add a second “real corpus” suite entrypoint
- ensure each case records its expected language, class/procedural mode, and
  expected artifact classes

### Step 3. Add Golden Snapshot Comparison

- capture normalized artifacts for each case
- compare them against checked-in goldens
- expose case-level mismatch details in the result model
- make the failure output concise and reviewable

### Step 4. Add Protected-Family Regression Coverage

- add tests that run a representative subset of the real corpus
- add at least one expected-failure case if the repository already has a stable
  failure artifact worth protecting
- verify the summary groups results by family and mismatch type

### Step 5. Keep Runtime Practical

- avoid turning the harness into a long-running e2e suite
- keep fixture choice small and targeted
- keep snapshot generation deterministic enough for regular local use

## Concrete File Plan

Expected implementation edits:

- `sciona/ingester/regression_harness.py`

Likely new or updated tests:

- `tests/test_ingest_regression_harness.py`
- possibly small supporting assertions in:
  - `tests/test_ingest_stateful.py`
  - `tests/test_bayesian_ingester.py`
  - `tests/test_message_passing.py`
  - `tests/test_ingest_biosppy_ecg.py`
  - `tests/test_ffi_emitter.py`
  - `tests/test_ingest_procedural.py`

Likely new fixture/golden directories:

- `tests/fixtures/ingest_regression/`
- `tests/golden/ingest_regression/`

Only touch monitor/cache/graph files if snapshot stability or failure-artifact
publication genuinely requires it.

## Regression Risks

Primary risks:

- choosing cases that are too heavy or environment-sensitive
- golden files becoming noisy and hard to review
- snapshot normalization removing too much and hiding meaningful semantic drift
- non-Python or Bayesian cases needing special handling that the generic
  harness does not yet express cleanly
- failure-artifact coverage becoming flaky if expected-failure cases are not
  tightly controlled

Mitigations:

- prefer small checked-in fixtures already exercised elsewhere in tests
- normalize only clearly incidental fields
- keep artifact sets explicit per case instead of assuming one-size-fits-all
- add direct normalization tests before broad corpus tests
- keep default corpus size modest

## Test Plan

Minimum test coverage for phase 11 should include:

- unit tests for artifact normalization
- unit tests for golden comparison and mismatch reporting
- unit tests for case manifest/corpus construction
- execution tests for at least one stateful/procedural real fixture pair
- execution tests for at least one non-Python or FFI case
- summary tests proving family/mismatch aggregation is correct

Recommended local regression slice:

- `pytest -q tests/test_ingest_regression_harness.py`
- `pytest -q tests/test_ingest_regression_harness.py tests/test_ingest_stateful.py tests/test_ingest_procedural.py`
- targeted non-Python/FFI coverage as needed:
  - `pytest -q tests/test_treesitter_rust.py tests/test_treesitter_cpp.py tests/test_treesitter_julia.py tests/test_ffi_emitter.py`

If real-corpus execution proves too expensive for one test target, split:

- fast unit tests for normalization/comparison
- a smaller integration slice for representative corpus execution

## Acceptance Criteria

Phase 11 is complete when:

- the harness includes a curated real-world regression corpus across the
  required protected families
- normalized golden artifacts exist for the selected cases
- harness results identify artifact mismatches in a reviewable way
- canonical IR/planning/emission regressions are easier to detect from one
  harness surface
- runtime remains practical enough for regular local or CI usage
- no new LLM dependency is introduced in the harness layer

## Deferred To Later Work

Explicitly defer:

- large benchmark dashboards or hosted reporting
- broad cache-format stabilization work beyond what the harness needs
- artifact-update CLI/UX polish beyond minimal developer ergonomics
- expanding the corpus into a very large library zoo
- performance optimization or serialization redesign that belongs to phase 12
