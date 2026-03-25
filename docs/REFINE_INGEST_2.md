# REFINE_INGEST Phase 2 Implementation Plan

## Phase Goal

Phase 2 redesigns the ingest intermediate representation so it can express
object-oriented library semantics faithfully without pushing guesswork into the
emitter.

The new IR must be able to represent, explicitly and deterministically:

- config state vs fitted state vs derived/transient artifacts
- method role and invocation shape
- required pre-state before a method may run
- post-call state updates
- exact source of each emitted output
- the difference between `return self`, `return self.attr`, `return value`,
  and metadata/query returns

Phase 2 is successful when phase-3 planning and phase-4 emission can operate
over the IR without needing to rediscover semantics from raw source.

## Scope Boundaries

In scope:

- define and land a new OO-aware ingest IR
- add deterministic lowering from phase-1 semantic facts into that IR
- add compatibility adapters so current plan/emitter code can coexist
- add tests and fixtures proving the IR captures sklearn-style and existing
  stateful cases

Out of scope:

- replacing the entire chunker/decomposer flow in the same change
- rewriting wrapper emission to use the new IR directly
- broad LLM prompt redesign beyond small adapter payloads
- verification/repair-loop changes
- full tree-sitter parity beyond keeping interfaces open for later adoption

Non-negotiable design rule:

- if the emitter must infer where an output came from or which state must be
  rehydrated, the IR is underspecified

## Current Code Touchpoints

Current plan and state models:

- `sciona/ingester/models.py`
  - `MacroAtomSpec`
  - `StateModelSpec`
  - `ProposedMacroPlan`
  - `ValidatedMacroPlan`

Current plan construction:

- `sciona/ingester/chunker.py`
  - deterministic simple-class path
  - LLM macro-atom parsing
  - config flattening
  - state hoisting
  - critic coverage checks
  - recursive decomposition hooks

Current IR consumers that reveal model gaps:

- `sciona/ingester/emitter.py`
  - `generate_atom_wrappers`
  - `generate_stateful_wrappers`
  - `build_cdg_export`
  - `emit_ingestion_bundle`

Pipeline orchestration and artifacts:

- `sciona/ingester/graph.py`
- `sciona/ingester/cache.py`
- `sciona/ingester/prompts.py`

Tests that should drive phase 2:

- `tests/test_ingester_chunker.py`
- `tests/test_ingester_emitter.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_config_flatten.py`
- `tests/test_ingest_biosppy_ecg.py`
- `tests/test_ingest_dl_boundary.py`
- `tests/test_bayesian_ingester.py`
- `tests/test_message_passing.py`
- `tests/test_ast_state_hoister.py`
- `tests/test_ingest_procedural.py`

## Current Gaps

The existing `MacroAtomSpec` and `StateModelSpec` flatten too many different
concepts into the same shape.

Specific limitations today:

- `inputs` and `outputs` mix constructor config, fitted state, and direct
  method I/O
- outputs do not record whether they come from a return value, an attribute
  read after mutation, or a metadata/query method
- `state_models` do not distinguish config state from learned/fitted state
- there is no representation of pre-state requirements per method/unit
- there is no first-class notion of method role in the plan layer
- `method_names` can group methods, but the grouping has no semantics
- `return self` and metadata/query returns cannot be represented cleanly
- the emitter currently compensates with assumptions, especially in stateful
  wrapper generation

Phase 1 now exposes richer semantic facts. Phase 2 must consume them instead of
reverting to the older read/write-only view.

## Required Inputs From Phase 1

The phase-2 IR should lower from these phase-1 facts:

- `MethodFact.signature`
- `MethodFact.return_facts`
- `MethodFact.call_facts`
- `MethodFact.semantic_role`
- `MethodFact.config_attributes`
- `MethodFact.fitted_attributes`
- `RawDataFlowGraph.attribute_facts`
- `RawDataFlowGraph.config_attributes`
- `RawDataFlowGraph.fitted_attributes`
- `RawDataFlowGraph.derived_attributes`
- `RawDataFlowGraph.semantic_unknowns`

Important boundary:

- phase 2 must treat phase-1 unknowns as unknowns
- it must not turn them into invented state fields or outputs

## Proposed IR

Introduce a new additive IR in `sciona/ingester/models.py` or a new
`sciona/ingester/ir.py` module. The exact file layout is flexible, but the
schema should be explicit and stable.

### Top-Level Container

Recommended top-level model:

- `IngestIRPlan`

Core fields:

- `subject_name`
- `source_language`
- `operations`
- `state_slots`
- `artifacts`
- `edges`
- `unknowns`
- `lowering_version`

### Operation Unit

Recommended operation model:

- `OperationSpec`

Fields:

- `operation_id`
- `display_name`
- `role`
  - `constructor`
  - `state_transition`
  - `query`
  - `metadata`
  - `predict`
  - `transform`
  - `score`
  - `helper`
  - `unknown`
- `method_bindings`
- `direct_inputs`
- `required_state_slots`
- `emitted_outputs`
- `state_effects`
- `concept_type`
- `is_optional`
- `is_opaque`
- `is_external`
- `provenance`

Design rule:

- an operation may bind one or more methods
- grouping multiple methods is allowed only when the lowering can explain why
  the group is semantically one unit
- otherwise one method becomes one operation

### Method Binding

Recommended method binding model:

- `MethodBinding`

Fields:

- `method_name`
- `signature`
- `call_style`
  - positional/keyword structure from phase 1
- `return_behavior`
- `requires_instance_state`
- `provenance`

This keeps exact source fidelity attached to the operation instead of relying on
free-form method names.

### State Slot

Recommended state model:

- `StateSlotSpec`

Fields:

- `slot_name`
- `state_kind`
  - `config`
  - `fitted`
  - `derived`
  - `transient`
  - `stochastic`
- `type_desc`
- `required_before`
- `written_by`
- `read_by`
- `source_attr`
- `provenance`

This replaces the current monolithic `StateModelSpec` view as the canonical IR.

`StateModelSpec` can remain as a legacy adapter product for the current emitter.

### Output Binding

Recommended output model:

- `OutputBindingSpec`

Fields:

- `output_name`
- `type_desc`
- `binding_kind`
  - `return_value`
  - `attribute_read`
  - `tuple_element`
  - `self_return`
  - `metadata_object`
  - `constant`
  - `unknown`
- `source_method`
- `source_attr`
- `tuple_index`
- `provenance`

This is the key phase-2 addition. The emitter cannot be faithful without it.

### State Effect

Recommended state mutation model:

- `StateEffectSpec`

Fields:

- `slot_name`
- `effect_kind`
  - `initialize`
  - `update`
  - `read_only`
  - `clear`
  - `unknown`
- `source_method`
- `provenance`

### Edge Model

Recommended edge model:

- `OperationEdge`

Fields:

- `source_operation_id`
- `target_operation_id`
- `edge_kind`
  - `data`
  - `state`
  - `metadata`
  - `control`
- `artifact_or_slot_name`
- `provenance`

## Lowering Rules From Phase 1 Facts

Phase 2 should define deterministic lowering rules from semantic facts into the
new IR.

Minimum deterministic lowering behavior:

- constructor-bound config attrs become `config` state slots
- attrs classified as fitted become `fitted` state slots
- attrs classified as derived become `derived` state slots unless the method is
  clearly query-only
- methods with `fit_or_update` role lower to `state_transition` operations
- methods with `predict_or_transform` lower to `predict` or `transform`
  operations
- methods with `query_or_metadata` lower to `query` or `metadata` operations
- `return self` yields no data artifact output; it records state continuity
- `return self.attr` yields `attribute_read` output binding
- direct call returns yield `return_value` or `tuple_element` bindings

Unknown handling:

- if phase 1 cannot prove an output source, the IR stores `binding_kind=unknown`
- if phase 1 cannot prove required state, the IR stores an explicit unknown
  dependency
- phase 2 must not invent a new slot or output to make the plan look cleaner

## Compatibility Strategy

Phase 2 should be additive first.

Recommended rollout:

1. Add the new IR models.
2. Lower semantic facts into the new IR.
3. Add a legacy adapter:
   - `ir_to_proposed_macro_plan(...)`
   - `ir_to_state_models(...)`
4. Keep current `ValidatedMacroPlan` and emitter paths working.
5. Only after that, let phase 3 and phase 4 consume the new IR directly.

Important rule:

- the legacy adapter may lose detail
- the canonical source of truth must become the new IR, not the adapter output

## Required Interfaces With Other Phases

Interface from phase 1:

- deterministic semantic facts are the only source of truth for signatures,
  return behavior, config/fitted state, and unknowns

Interface to phase 3:

- deterministic planning should consume `IngestIRPlan`
- phase 3 must not reconstruct state roles from `MacroAtomSpec`

Interface to phase 4:

- the emitter must be able to determine:
  - exact method to call
  - exact arguments to supply
  - exact state to inject before call
  - exact state to extract after call
  - exact output source after call

Interface to phase 6:

- regression fixtures should snapshot both the new IR and the legacy adapter
  output during the transition

## Deterministic vs LLM Responsibilities

Deterministic in phase 2:

- IR schema
- lowering from phase-1 facts
- state slot classification from phase-1 inventories
- output binding from phase-1 return facts
- state/data/metadata edge construction when evidence is sufficient
- legacy adapter generation

LLM responsibilities in phase 2:

- none for the core IR schema
- at most, later prompt adapters may consume IR summaries for naming or
  grouping, but that should not block phase-2 implementation

## Rollout Plan

### Step 0. Lock the Regression Harness Slice

Before changing plan structures, ensure the lightweight harness from phase 6 is
available for:

- sklearn-style estimator fixture
- rolling/windowed stateful class
- biosignal wrapper
- Bayesian/message-passing example
- opaque DL boundary case
- procedural ingest case

Phase 2 should add IR snapshots to that harness.

### Step 1. Land Additive IR Models

- add `IngestIRPlan` and supporting models
- keep `MacroAtomSpec`, `StateModelSpec`, and `ValidatedMacroPlan` intact
- default all new fields so existing tests and tree-sitter paths do not break

### Step 2. Build Deterministic Lowering

- add lowering helpers that consume phase-1 semantic facts
- begin with one-method-per-operation lowering
- only group methods when deterministic evidence is strong

Suggested placement:

- new `sciona/ingester/ir_lowering.py`, or
- a clearly separated phase-2 section in `chunker.py`

### Step 3. Add Legacy Adapters

- add `ir_to_proposed_macro_plan(...)`
- add `ir_to_validated_macro_plan(...)`
- add `ir_to_state_model_specs(...)`

This allows current chunker/emitter code to continue while phase-3 and phase-4
work is staged.

### Step 4. Thread IR Through the Pipeline

- `graph.py` should preserve the richer IR in monitor/debug artifacts
- cache payloads should be able to store the IR or an IR snapshot
- the current pipeline state should carry both:
  - canonical IR
  - legacy adapted plan

### Step 5. Update Validation Logic

Current coverage checks in `chunker.py` are attribute-name based.
Phase 2 should add IR-aware validation:

- every state slot has a provenance-backed source
- every output binding has a source method and source kind
- every operation with required state references existing slots
- metadata/query operations do not pretend to mutate learned state
- `return self` operations do not become fake output artifacts

### Step 6. Expand Tests and Golden Cases

- add model-level tests for IR construction
- add lowering tests from semantic facts to IR
- add adapter tests from IR to legacy plans
- add emitter-facing tests that confirm the adapter preserves current behavior

## Concrete File Plan

Expected phase-2 edits:

- `sciona/ingester/models.py`
  - add new IR models
- `sciona/ingester/chunker.py`
  - canonicalize on IR construction
  - preserve legacy plan adapter output
- `sciona/ingester/graph.py`
  - carry IR through pipeline state and artifacts
- `sciona/ingester/cache.py`
  - store/load IR snapshots if needed
- `sciona/ingester/prompts.py`
  - only minimal summary changes if prompt inputs need IR-aware context
- tests
  - chunker, emitter, stateful, biosppy, Bayesian, and new IR-focused cases

Likely new modules:

- `sciona/ingester/ir_lowering.py`
- optionally `sciona/ingester/ir_validation.py`

## Regression Risks

Primary risks:

- phase 2 breaks current emitter assumptions before phase 4 is ready
- the IR becomes too abstract and loses direct binding to source facts
- sklearn-driven state distinctions degrade Bayesian or message-passing cases
- the adapter back to legacy plans becomes lossy in ways that hide semantic bugs

Mitigations:

- keep phase-1 provenance on every IR node
- maintain a canonical IR plus explicit legacy adapter during rollout
- require golden comparisons on sklearn-style and protected non-sklearn cases
- keep stochastic/Bayesian fields as first-class state slots, not ad hoc flags

## Test and Benchmark Plan

New direct tests:

- IR model construction from a semantic-fact fixture
- lowering of:
  - `fit` method writing fitted attrs
  - `predict` method reading fitted attrs
  - metadata/query method returning router/tags
  - `return self`
  - tuple-return methods
- adapter tests proving:
  - outputs retain source bindings
  - config state is not collapsed into fitted state
  - metadata/query methods do not become mutators

Protected integration cases:

- sklearn-style estimator fixture modeled on `CalibratedClassifierCV`
- `RollingAverager`
- BioSPPy ECG wrapper
- one Bayesian/message-passing case
- one opaque DL boundary case
- one procedural ingest case

Metrics to record:

- IR completeness
- number of unknown bindings
- legacy adapter parity for existing passing tests
- pipeline completion and verification rates

## Acceptance Criteria

Phase 2 is complete when all of the following are true:

- a canonical OO-aware ingest IR exists in code
- semantic facts lower deterministically into that IR
- the IR explicitly distinguishes config, fitted, derived, and transient state
- output bindings explicitly identify their source kind
- `return self` and metadata/query outputs are represented without invention
- the current ingest pipeline can still run via a legacy adapter
- curated sklearn-style and non-sklearn stateful fixtures demonstrate improved
  semantic fidelity
- phase-3 work can plan over the IR without inspecting raw source
- phase-4 work can emit faithful wrappers from IR without adding semantic
  guesses

## Deferred to Later Phases

Phase 3:

- deterministic grouping/planning policies over the new IR

Phase 4:

- emitter rewrite to consume the IR directly

Phase 5:

- narrowing verification and repair behavior

## Recommended Execution Order

1. Lock regression fixtures and IR snapshots.
2. Add additive IR models.
3. Implement deterministic lowering from phase-1 facts.
4. Add legacy adapters and keep current plan/emitter code working.
5. Move validation onto the IR.
6. Only then start phase-3 deterministic planning changes.
