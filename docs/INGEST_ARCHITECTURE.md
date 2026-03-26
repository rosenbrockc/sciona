# Canonical Ingest Architecture

This note is for maintainers. It describes the current canonical ingest path,
where compatibility exports begin, and which operational surfaces are stable
enough for regression and tooling.

The runtime is canonical-first. The semantic source of truth is the canonical
IR and planning data, not the legacy macro-atom/state-model projections.

## Primary Touchpoints

- `sciona/ingester/extractor.py`
- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/graph.py`
- `sciona/ingester/regression_harness.py`
- `sciona/ingester/cache.py`
- `sciona/ingester/monitor.py`

## Canonical Data Flow

The top-level runtime in `sciona/ingester/graph.py` runs these phases in order:

1. `phase1_extract`
   - `extractor.py` builds `RawDataFlowGraph` and `MethodFact` data
   - this is deterministic AST / tree-sitter / extractor output
   - ambiguous facts are recorded as unknowns instead of guessed

2. `phase2_chunk`
   - `chunker.py` lowers the extracted graph into `ValidatedMacroPlan`
   - the canonical payload lives in `ValidatedMacroPlan.plan.canonical_ir`
   - `IngestIRPlan` contains the canonical operations, state slots, outputs,
     edges, and unknowns
   - `IngestPlanGraph` is the deterministic planning view over that IR
   - `phase2_conjugate_heuristics` may refine conjugate-update classification,
     but it still stays within the canonical plan

3. `phase3_emit`
   - `emitter.py` consumes the validated plan and canonical IR to generate the
     `IngestionBundle`
   - canonical IR drives wrapper generation, witness generation, CDG export,
     and the derived match results
   - non-Python sources can add FFI binding stubs, but they still flow through
     the same canonical-first emission path

4. Verification
   - `verify_types` runs mypy / proof-environment checks
   - `verify_ghost` runs ghost-simulation checks
   - verification may emit failure artifacts, but it does not redefine the
     canonical IR

Procedural ingest is a narrower path. `ingest_procedural` bypasses chunking and
the verification loops, but it still materializes canonical IR through
`build_procedural_plan`.

## Compatibility Boundary

`sciona/ingester/models.py` still defines the legacy projection types:

- `MacroAtomSpec`
- `StateModelSpec`
- `ProposedMacroPlan`
- `ValidatedMacroPlan`

Those types remain because downstream code still expects them, but they are not
authoritative anymore. The canonical source of truth is:

- `IngestIRPlan`
- `OperationSpec`
- `StateSlotSpec`
- `OutputBindingSpec`
- `OperationEdge`
- `IngestPlanGraph`

The explicit bridge functions are the compatibility boundary:

- `legacy_macro_atoms_from_ir`
- `legacy_state_models_from_ir`
- `runtime_macro_atoms`
- `runtime_state_models`
- `runtime_edge_definitions`
- `materialize_legacy_plan_views`

Current runtime behavior follows these rules:

- canonical IR is built first
- legacy macro-atom and state-model views are derived from the canonical plan
- `build_cdg_export()` prefers canonical IR edges when they exist
- compatibility exports are only for adapters, harness consumers, and any
  remaining downstream expectations

Future work should not introduce new runtime semantics into the legacy view.
If a change needs new semantic truth, it belongs in canonical IR first.

## Verification And Repair Boundaries

Verification is intentionally narrow.

Type verification:

- `verify_types` writes the generated bundle files to the proof environment
- failures are classified by `classify_type_failure`
- only repairable mechanical failures route to `repair_types`
- `build_deterministic_type_fixes` is the only deterministic repair path

Ghost verification:

- `verify_ghost` runs `run_ghost_simulation`
- failures are classified by `classify_ghost_failure`
- repairable witness failures route to `repair_ghost`
- deadlock-style message-passing failures can route to
  `repair_message_cycle`
- `repair_message_cycle` is the only LLM-assisted repair path, and it is still
  constrained to witness edits rather than semantic redesign

The boundary is deliberate:

- repairs may fix syntax, typing, or witness mechanics
- repairs must not invent missing canonical semantics
- if a failure is semantic, the run should fail with artifacts rather than be
  papered over

## Regression Harness

`sciona/ingester/regression_harness.py` is the maintainer-facing regression
surface.

It covers:

- curated family cases from `default_ingest_regression_cases()`
- `IngestMonitor` status / marker / trace summaries
- normalized artifact bundles for goldens
- semantic expectations such as canonical IR presence, planning graph presence,
  source language, and minimum CDG size

Important harness behavior:

- `capture_normalized_artifact_bundle()` snapshots the canonical artifacts
- transient monitor fields and unstable paths are stripped during normalization
- golden comparisons are semantic, not byte-for-byte runtime noise
- `verification_failure.json` is treated as a surfaced artifact when present

The harness is the right place to add assertions when a change affects:

- canonical IR shape
- planning graph shape
- artifact publication
- monitor / cache interpretation

## Cache

`sciona/ingester/cache.py` stores content-addressed ingestion bundles.

The cache key is derived from:

- source file SHA-256
- class name
- `max_depth`
- `line_threshold`
- cache format version

The envelope is versioned and explicit:

- schema: `sciona.ingester.cache-envelope`
- schema version: `1`
- runtime mode: `canonical-first`
- payload kind: `ingestion_bundle`

Load behavior is intentionally tolerant:

- current envelopes are validated before payload extraction
- legacy top-level payloads are still accepted on read
- corrupt or mismatched cache entries fall back to a miss

The cache is an execution optimization, not a semantic authority. If cache
behavior changes, keep the canonical bundle format stable and update harness
expectations only when the payload meaning changes.

## Monitor Surfaces

`sciona/ingester/monitor.py` provides the runtime observability surface.

Stable file / artifact surfaces:

- `.ingest_status.json`
- `COMPLETED.json`
- `FAILED.json`
- `trace.jsonl`
- `.partial/`

Stable schemas:

- status schema: `sciona.ingester.monitor.status`
- marker schema: `sciona.ingester.monitor.marker`
- surface schema: `sciona.ingester.monitor.surface`
- schema version: `1`

What the monitor surface means:

- status records live progress, heartbeat, and run metadata
- markers record completion or failure
- `read_surface()` combines status and marker data into a derived run view
- stale running runs can be classified as stalled
- the regression harness consumes the normalized surface rather than raw file
  layout details

The monitor is intentionally operational. It should remain useful for local and
CI runs without becoming a second semantic model.

## Where To Change What

Use this as the maintainer rule of thumb:

- extraction changes belong in `extractor.py` and the extracted fact models
- canonical semantic shape changes belong in `models.py` and the chunker
- compatibility-only changes should stay inside the legacy projection helpers
- emission / artifact changes belong in `emitter.py`
- verification policy changes belong in `graph.py`
- regression expectations belong in `regression_harness.py`
- cache envelope or key changes belong in `cache.py`
- status / marker / surface changes belong in `monitor.py`

## Protected Invariants

Keep these true unless a future phase explicitly changes them:

- canonical IR stays the source of truth
- legacy exports stay derived and non-authoritative
- verification repairs stay narrow and mechanical
- monitor / cache surfaces stay versioned and readable by the harness
- regression goldens should reflect semantics, not transient runtime noise

