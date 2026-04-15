# Benchmark Manifest Schema And Seed Plan

## Goal

Define a provider-owned benchmark manifest format that can:

- represent cross-disciplinary atom and CDG evaluation without forcing one global metric space
- attach results to immutable artifact identity via `content_hash`
- seed the current flat `atom_benchmarks` table deterministically
- leave room for a richer future Supabase schema

This spec is intentionally **contract-centric** rather than library-centric.

## Principles

1. A benchmark suite defines a problem contract, protocol, and metric set.
2. An artifact result binds one `artifact_fqdn` and `content_hash` to one suite.
3. Comparability is explicit and local to a suite.
4. Atoms and CDGs can both participate in the same suite.
5. The manifest is the source of truth; current Supabase benchmark rows are a flattened projection.

## File Layout

Provider repos should converge on this layout:

```text
data/benchmarks/benchmark_suites.json
data/benchmarks/benchmark_results.json
```

In the short term, matcher can validate and prototype against the same shape.

## Suite Schema

Each suite entry should contain:

- `suite_id`: stable identifier
- `title`: human-readable name
- `artifact_scope`: `atom`, `cdg`, or `both`
- `contract_id`: stable contract identifier
- `contract_summary`: short contract description
- `domain_tags`: domain labels like `signal_processing`, `state_estimation`
- `family_tags`: family labels like `signal_detect_measure`, `kalman_filter`
- `modality_tags`: modality/data-shape labels like `ecg`, `waveform`, `time_series`
- `dataset_tag`: stable dataset identifier for the current suite
- `dataset_version`: version/date/hash of the dataset slice
- `protocol_version`: benchmark protocol version
- `comparability_class`: usually `within_suite_only`
- `metrics`: metric definitions
- `slices`: optional sub-cohort labels
- `owner_repo`: provenance
- `status`: `draft`, `active`, or `retired`
- `notes`: optional free-text notes

Each metric definition should contain:

- `metric_name`
- `direction`: `higher_is_better` or `lower_is_better`
- `unit`
- `primary`: boolean
- `aggregation`: optional, e.g. `mean`, `median`, `p95`
- `notes`: optional

## Result Schema

Each result entry should contain:

- `suite_id`
- `artifact_fqdn`
- `artifact_kind`: `atom` or `cdg`
- `content_hash`
- `semver`: optional but recommended
- `metric_name`
- `metric_value`
- `slice_key`: optional slice label, empty means whole-suite aggregate
- `measured_at`
- `runner`
- `run_config_hash`
- `status`: `completed`, `failed`, or `partial`
- `evidence_uri`: optional pointer to a richer report
- `notes`: optional

## Deterministic Seed Mapping

### Current Supabase tables

Current runtime consumption still depends on:

- `benchmark_suites`
- `atom_benchmarks`

### Suite flattening

Map one suite manifest row to `benchmark_suites`:

- `benchmark_id <- suite_id`
- `domain_tags <- domain_tags + family_tags + modality_tags`
- `description <- contract_summary`
- `dataset_s3_key <- ""` for now, or artifact-relative pointer later
- `metric_names <- [metric.metric_name ...]`
- `curation_source <- "foundation"` for checked-in manifests
- `status <- active/proposed/retired` mapped from suite status

### Result flattening

Map one result manifest row to `atom_benchmarks` only when:

- `artifact_kind == "atom"`
- the target `content_hash` resolves to a seeded `atom_versions.version_id`

Then:

- `version_id <- resolved from (artifact_fqdn, content_hash)`
- `benchmark_name <- suite_id`
- `metric_name <- metric_name`
- `metric_value <- metric_value`
- `dataset_tag <- dataset_tag` from the suite
- `measured_at <- measured_at`

CDG results should not be thrown away. In the short term they should be retained in the manifest file and skipped for `atom_benchmarks` projection. In the medium term they should land in a new `artifact_benchmarks` table.

## Validation Rules

The seeder should fail closed when:

- `suite_id` is duplicated
- a suite has zero metrics
- more than one metric is marked `primary`
- a result references an unknown `suite_id`
- a result metric is not declared by its suite
- a result references an unknown artifact
- an atom result references a known artifact but unresolved `content_hash`

The seeder should warn, not fail, when:

- a suite is `draft`
- a suite has no results yet
- a result exists for a `cdg` artifact but there is no `artifact_benchmarks` target table yet

## Recommended Rollout

### Phase 1

- adopt the manifest format in matcher docs
- add validation in the future provider-owned seed path
- seed `benchmark_suites`
- seed `atom_benchmarks` from atom-only results

### Phase 2

- add `artifact_benchmarks` to Supabase for CDGs and future non-atom artifacts
- expose `get_artifact_benchmarks(...)`
- include artifact benchmarks in the unified catalog document

### Phase 3

- include benchmark suite metadata and result rows in the SQLite manifest export
- allow planner/template retrieval to consider benchmark priors

## First Draft Suites

The first benchmark suites should cover:

- `signal_detect_measure`
- `kalman_filter`
- `particle_filter`

The canonical draft lives in:

- [benchmark_suites.draft.json](/Users/conrad/personal/sciona-matcher/docs/benchmarks/benchmark_suites.draft.json)

An accompanying result-shape example lives in:

- [benchmark_results.example.json](/Users/conrad/personal/sciona-matcher/docs/benchmarks/benchmark_results.example.json)

## Notes On Scope

This format is intentionally broad enough to support:

- domain-specific suites like ECG rate estimation
- family-specific suites like Kalman tracking
- cross-family contract suites where both atoms and CDGs compete on the same top-level task

It does **not** assume that metric values are comparable across suites.
