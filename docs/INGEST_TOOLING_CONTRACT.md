# Ingest Tooling Contract

This document defines the stable public contract for post-phase-12 ingest
tooling surfaces.

The rule for consumers is simple:

- rely on the stable contract sections below
- treat informational fields as helpful but not guaranteed
- do not depend on internal implementation details

The contract is intentionally narrower than the runtime implementation. It
describes what tooling may rely on, not every field the runtime currently emits.

## 1. Cache Envelopes

The ingest cache is content-addressed and versioned. `save_ingest_cache()` writes
an envelope, and `load_ingest_cache()` accepts either that envelope or the
legacy top-level bundle shape for backward compatibility.

### Stable contract

- cache keys are deterministic for the same source content and request inputs
- cache entries are versioned by cache-key version and envelope schema version
- the envelope uses these top-level fields:
  - `schema`
  - `schema_version`
  - `cache_key`
  - `cache_key_version`
  - `runtime_mode`
  - `payload_kind`
  - `payload`
- `payload_kind` identifies the serialized bundle family
- `runtime_mode` identifies the canonical runtime mode used by the writer
- `payload` is the serialized runtime bundle used for restore

### Best-effort informational fields

- `payload_summary` is derived metadata for inspection and debugging
- summary counts may change in shape or breadth as the runtime evolves

### Internal details

- the key derivation uses a JSON payload and a hash implementation detail
- the temporary file naming used for atomic writes is internal
- readers should not depend on the raw on-disk formatting

### Runtime bundle surface

The cache payload currently serializes `IngestionBundle`. Tooling may rely on
that bundle shape at a high level:

- `cdg`
- `sub_graphs`
- `generated_atoms`
- `generated_state_models`
- `generated_witnesses`
- `match_results`
- `mypy_passed`
- `ghost_sim_passed`

`ghost_sim_report` is included, but it is free-form informational data and should
not be treated as a strict schema commitment.

## 2. Monitor Surfaces

The monitor writes three durable surfaces:

- `.ingest_status.json`
- `COMPLETED.json` or `FAILED.json`
- `trace.jsonl` when tracing is enabled

It also exposes `read_surface()` as a normalized tooling view over status and
marker files.

### Stable contract: status file

The normalized status surface has these stable fields:

- `schema`
- `schema_version`
- `run_id`
- `state`
- `phase`
- `current_step`
- `source_path`
- `class_name`
- `procedural`
- `llm_provider`
- `llm_model`
- `max_depth`
- `started_at`
- `ended_at`
- `last_heartbeat_at`
- `llm_call_inflight`
- `error`
- `summary`

Tooling may rely on the meaning of:

- `state` being one of the monitor lifecycle states
- `run_id` identifying the run
- `started_at`, `ended_at`, and `last_heartbeat_at` being timestamps
- `summary` being the completion/failure summary payload when present

### Stable contract: marker files

The normalized marker surface has these stable fields:

- `schema`
- `schema_version`
- `state`
- `run_id`
- `phase`
- `completed_at` or `failed_at`
- `error`
- `summary`

`read_surface()` returns:

- `schema`
- `schema_version`
- `derived_state`
- `status`
- `marker`

### Best-effort informational fields

- `phase` and `current_step` are useful progress markers, but they should be
  treated as operational hints rather than semantic guarantees
- `llm_call_inflight` is an operational detail for stall detection and live
  debugging
- `llm_provider` and `llm_model` are descriptive context, not semantic inputs
- `summary` is a convenience payload whose exact content may expand over time

### Stable contract: trace surface

When enabled, `trace.jsonl` is an append-only JSONL stream. Each line is a JSON
object with these top-level fields:

- `timestamp`
- `round`
- `phase`
- `event_type`
- `node_id`
- `payload`
- `duration_ms`

The existence of the trace file and the top-level event keys are stable. The
exact event payload schema is best-effort and event-specific.

### Internal details

- stdout mirroring from `monitor_stdout` is for human debugging only
- the atomic replace mechanism and file rewrite mechanics are internal
- consumer code should use the normalized readers instead of reimplementing
  parsing logic

## 3. Regression Golden Formats

The regression harness compares normalized artifacts against checked-in goldens.
Those goldens are the public artifact contract for regression review.

### Stable contract

The checked-in golden directory for one case is:

- `tests/golden/ingest_regression/<case_id>/`

The stable artifact file set is:

- `canonical_ir.json`
- `planning_graph.json`
- `atoms.py`
- `state_models.py`
- `witnesses.py`
- `cdg.json`
- `verification_failure.json`

The harness treats these as the canonical comparison targets. A case may omit an
artifact only when that omission is already part of the case definition.

### Stable contract: normalized comparison behavior

Before comparison, the harness normalizes snapshot payloads so goldens represent
semantics rather than runtime noise. Stable normalization behavior includes:

- removal of transient monitor/runtime keys
- normalization of path-like strings to relative placeholders
- stable ordering for lists that have obvious identity keys
- JSON artifact validation through the corresponding model where applicable

### Best-effort informational fields

- `matches.json` and `ingest_failure_state.json` are runtime artifacts, not
  checked-in golden formats
- textual formatting of the golden files is not itself part of the contract

### Maintainer review expectations

- the fast CI slice in `.github/workflows/ingest-regression.yml` is the
  always-on gate for the curated corpus and checked-in goldens
- the broader protected-family slice runs on schedule and manual dispatch so
  maintainers can recheck adjacent ingest surfaces without slowing normal PRs
- checked-in golden files should change only when canonical ingest semantics or
  the normalized golden contract changes intentionally
- when goldens change, review the runtime/code change and the corresponding
  `tests/golden/ingest_regression/<case_id>/` diff together in the same PR
- if goldens need to be refreshed intentionally, run the broader manual slice
  before merging so wider family coverage still passes with the new snapshots

### Internal details

- the specific ordering hints used by the normalizer are implementation detail
- the placeholder spellings used during path normalization are internal
- the harness may evolve its internal normalization as long as the semantic
  comparison contract stays intact

## 4. Canonical Runtime Export Surfaces

The canonical runtime source of truth is the canonical IR and planning graph.
Legacy exports remain available only as compatibility views.

### Stable contract

These runtime surfaces are the ones tooling may rely on:

- `ProposedMacroPlan.canonical_ir`
- `ProposedMacroPlan.planning_graph`
- `runtime_macro_atoms()`
- `runtime_state_models()`
- `runtime_edge_definitions()`
- `materialize_legacy_plan_views()`

Their contract is:

- canonical IR takes precedence whenever it exists and contains operations
- legacy macro atoms, state models, and edge definitions are compatibility
  exports, not authoritative state
- compatibility views should preserve the canonical runtime meaning rather than
  inventing new semantics

The canonical IR and planning graph themselves are the authoritative exports
for tooling that needs the true ingest plan.

### Best-effort informational fields

- legacy macro-atom and state-model fields are useful compatibility views, but
  they are downstream conveniences rather than authoritative inputs
- `generated_atoms`, `generated_state_models`, and `generated_witnesses` are
  runtime publication outputs, not canonical truth

### Internal details

- `canonical_operation_id()` is a normalization helper, not a public semantic
  boundary
- the compatibility view construction logic may evolve as long as canonical
  precedence stays intact

## 5. Regression Harness Result Formats

The harness result rows are stable JSON-friendly summaries for tooling and CI.

### Stable contract

`IngestRegressionResult` is the per-case result shape. Tooling may rely on the
presence of:

- case identity fields
- completion and failure classification fields
- cache-state and runtime-summary fields
- artifact comparison fields
- semantic-check fields
- canonical-IR / planning-graph presence flags

`IngestRegressionSummary` and `FamilyBreakdown` are the aggregate summary
surfaces. Tooling may rely on them as compact reporting formats.

### Best-effort informational fields

- `cache_state_source`, `type_failure_reason`, `ghost_failure_reason`, and
  `golden_mismatch_details` are explanation fields, not stable decision inputs
- `published_artifacts` and `output_dir` are operational conveniences
- the exact list of semantic checks in a run can grow as the suite expands

### Internal details

- the precise aggregation implementation is internal
- result ordering within lists is only guaranteed where the harness already
  defines a deterministic order

## 6. Practical Reading Guide

If you are writing tooling:

- read cache envelopes through the versioned envelope contract
- read monitor status and marker files through the normalized monitor APIs
- compare regression artifacts through the normalized golden contract
- treat canonical IR and planning graph as authoritative runtime exports
- avoid depending on free-form reports, transient counts, or file formatting

If you are changing runtime code:

- keep stable top-level keys intact unless a version bump accompanies the
  change
- add new fields as informational before promoting them to stable contract
- do not repurpose an existing stable field for a different meaning
