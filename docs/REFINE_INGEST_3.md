# REFINE_INGEST Phase 3 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 3 replaces LLM-heavy ingest decomposition with deterministic planning
rules over the canonical ingest IR introduced in phase 2.

The planner should answer:

- which operations should remain atomic
- which operations should be decomposed into smaller operations
- when recursion is justified
- what evidence is required before introducing a child operation, edge, output,
  or state dependency

The canonical planning unit is no longer a raw `MacroAtomSpec`. It is the
phase-2 `IngestIRPlan` and its `OperationSpec` nodes.

## Scope Boundaries

In scope:

- deterministic planning rules over `IngestIRPlan`
- deterministic-first decomposition for common OO patterns
- explicit planner eligibility rules for LLM fallback
- recursion policy over IR operations
- compatibility adapters back to legacy `MacroAtomSpec.children` and `sub_edges`
- tests for deterministic and fallback planning behavior

Out of scope:

- direct emitter rewrite over IR
- verification/repair policy changes
- broad prompt redesign unrelated to decomposition inputs
- tree-sitter parity beyond preserving compatibility

Key rule:

- the planner may group or split operations, but it may not invent state slots
  or outputs not already supported by phase-1 facts and phase-2 IR bindings

## Current Code Touchpoints

Primary implementation surface:

- `sciona/ingester/chunker.py`

Relevant existing decomposition code:

- `is_atom_complex(...)`
- `_gather_source_for_atom(...)`
- `_parse_sub_atoms(...)`
- `_decompose_single_atom(...)`
- `decompose_complex_atoms(...)`

Deterministic fallback already present:

- `sciona/ingester/control_flow_decomposer.py`

Supporting prompt/route surface:

- `sciona/ingester/prompts.py`
- `sciona/ingester/graph.py`

Primary regression tests:

- `tests/test_chunker_depth.py`
- `tests/test_ingester_chunker.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_biosppy_ecg.py`
- `tests/test_ingest_config_flatten.py`
- `tests/test_ingest_procedural.py`
- `tests/test_ingest_dl_boundary.py`

## Current Gaps

The current decomposition path still has legacy behavior that phase 3 should
replace or contain:

- decomposition targets `MacroAtomSpec` instead of canonical IR operations
- complexity heuristics are line-count/call-count based, not semantic-role based
- LLM decomposition can still invent conceptual substeps and invented outputs
- deterministic CFG decomposition is purely structural and not IR-aware
- child atoms do not carry explicit evidence for why they exist
- recursion eligibility is tied to legacy tree depth rather than operation type

Phase 3 should convert decomposition from “large source block splitting” into
“evidence-backed planning over operation semantics”.

## Inputs From Prior Phases

Phase 3 should consume:

- phase-1 semantic facts
- phase-2 canonical IR:
  - `IngestIRPlan`
  - `OperationSpec`
  - `StateSlotSpec`
  - `OutputBindingSpec`
  - `OperationEdge`

Required inherited guarantees:

- output bindings already identify return-value vs attribute-read vs metadata
- state slots already identify config/fitted/derived/transient roles
- unknowns are explicit and must remain explicit

## Planning Model

### Canonical Planner Output

Add an additive planning layer for decomposition, either in
`sciona/ingester/models.py` or a new `sciona/ingester/planning.py`.

Recommended models:

- `DecompositionDecision`
  - `operation_id`
  - `decision`
    - `keep_atomic`
    - `decompose_deterministic`
    - `decompose_llm`
    - `blocked_unknown`
  - `reason`
  - `evidence`
  - `child_operation_ids`

- `PlannedOperationGroup`
  - deterministic grouping of one or more `OperationSpec`s into an output atom
  - `group_id`
  - `member_operation_ids`
  - `group_role`
  - `required_state_slots`
  - `emitted_outputs`
  - `planner_source`
    - `deterministic`
    - `control_flow`
    - `llm`

- `IngestPlanGraph`
  - planned operation groups
  - inter-group edges
  - blocked/unknown nodes
  - planner metadata

The exact names can vary, but the planner output must be explicit and separate
from raw IR construction.

### Legacy Adapter

As with phase 2, keep a compatibility adapter:

- `planned_ir_to_macro_atoms(...)`
- `planned_ir_to_sub_edges(...)`

This allows current CDG/emitter paths to keep functioning until phase 4.

## Deterministic Planning Rules

Phase 3 should add deterministic-first planning rules for common OO patterns.

### Rule Family 1: Keep Atomic

An operation should stay atomic when any of the following is true:

- role is `metadata` or `query`
- role is `predict`, `transform`, or `score` and it already has one clear
  method binding plus explicit output bindings
- role is `state_transition` and it corresponds to one real upstream method
  with clear state effects
- operation is opaque, external, or procedural legacy output
- decomposition would only restate return-value extraction as fake substeps

### Rule Family 2: Deterministically Decompose

Allow deterministic decomposition when evidence is strong:

- one operation binds multiple upstream methods that form a real sequence
- state-transition pipelines contain a clear ordered call chain
- CFG decomposition yields child units that can be mapped back to existing IR
  inputs/outputs without invention
- sequential children can inherit real state/output bindings from parent facts

Deterministic patterns to support first:

- constructor/config initialization split from fit/update transition
- fit/update method with clear internal stages
  - validate input
  - compute sufficient statistics
  - update fitted state
- prediction pipeline with clear staged helpers
  - preprocess
  - core infer/predict
  - postprocess
- signal pipeline wrappers like BioSPPy where the wrapper is already a staged
  orchestration over real methods

### Rule Family 3: Block on Unknown

Do not decompose when decomposition would require guessing:

- operation depends on unknown output bindings
- operation depends on unknown state requirements
- CFG decomposition yields children that cannot be bound to known outputs/state
- dynamic call behavior prevents faithful stage boundaries

In these cases:

- either keep the operation atomic
- or mark it `blocked_unknown` and surface it in artifacts

### Rule Family 4: LLM Fallback Eligibility

LLM decomposition should only be allowed when all of the following are true:

- operation exceeds deterministic complexity thresholds
- deterministic IR-aware planning cannot derive faithful children
- the parent IR still provides enough factual anchors to constrain the prompt
- the prompt can forbid introducing unbound outputs or state slots

LLM fallback should be forbidden when:

- the operation is metadata/query-only
- the operation already has faithful one-method semantics
- unknowns in the parent IR are too large to constrain safe subdivision

## Evidence Requirements

Before phase 3 may introduce a child operation, it must have evidence for:

- source method or source region backing the child
- direct inputs backed by existing parent inputs or bound artifacts
- emitted outputs backed by existing `OutputBindingSpec`s or CFG-observed
  intermediate values
- required state backed by existing `StateSlotSpec`s

Before phase 3 may introduce an inter-child edge, it must have evidence for:

- a matching output/input binding
- or a state slot written by one child and read by another
- or a deterministic CFG data dependency

Prohibited:

- inventing names for new fitted attrs
- inventing “result” outputs when the parent does not emit one
- inventing metadata or router outputs not present in the IR

## Complexity and Recursion Policy

Replace the current source-length/call-count heuristic with an IR-aware policy.

Suggested decomposition score inputs:

- number of method bindings in the operation
- number of distinct state effects
- number of emitted outputs
- size/shape of internal call graph
- CFG decomposition confidence
- unknown count

Suggested recursion rules:

- metadata/query operations never recurse
- single-method predict/query operations usually do not recurse
- multi-stage state-transition operations may recurse
- recursion stops when:
  - deterministic child operations are already single-purpose
  - max depth reached
  - any child would require unknown invention

Preserve the current `ingester_max_depth` behavior as an upper bound, but make
semantic eligibility the primary gate.

## Required Interfaces With Other Phases

Interface from phase 2:

- canonical IR is the planner input
- planner output must preserve a compatibility adapter for current emitter flow

Interface to phase 4:

- planned groups must carry enough information for direct wrapper emission later
- child operations should already know their real inputs, state requirements,
  and output sources

Interface to phase 5:

- planner should surface blocked/unknown decisions early so semantic issues fail
  before repair loops

Interface to phase 6:

- regression harness should snapshot planner decisions and LLM fallback counts

## Deterministic vs LLM Responsibilities

Deterministic:

- keep/decompose/block decision
- decomposition for common OO patterns
- CFG/IR merge when control-flow evidence is strong
- recursion stopping rules
- compatibility adapter back to legacy macro-atom tree

LLM:

- only for genuinely ambiguous multi-method orchestration
- only under explicit fallback eligibility rules
- only with prompts constrained by canonical IR bindings

## Rollout Plan

### Step 0. Lock Regression Cases

Add or confirm fixtures for:

- sklearn-style estimator wrapper with fit/predict/query methods
- rolling/stateful class
- BioSPPy staged wrapper
- opaque DL boundary
- procedural ingest example

Phase 3 should record whether each case:

- stayed atomic deterministically
- decomposed deterministically
- required LLM fallback

### Step 1. Add Planning Models

- add planner decision/group models
- keep them additive
- do not replace `MacroAtomSpec` storage yet

### Step 2. Build IR-Aware Deterministic Planner

- operate over `IngestIRPlan.operations`
- classify each operation into keep/decompose/block
- prefer one-operation-to-one-atom when semantics are already faithful

### Step 3. Integrate CFG Decomposer With IR

- use `control_flow_decomposer.py` only after deterministic semantic planning
  says decomposition is allowed
- map CFG children back to existing IR bindings
- discard CFG suggestions that require invented outputs/state

### Step 4. Restrict LLM Fallback

- update decomposition prompts to include canonical IR facts
- add hard validation that LLM-produced children reference only known outputs,
  state slots, and operation inputs
- reject invalid LLM decompositions instead of silently accepting them

### Step 5. Adapt Planned Groups Back to Legacy Macro-Atoms

- preserve current `children` and `sub_edges` structure
- annotate source of each decomposition:
  - deterministic planner
  - CFG decomposition
  - LLM fallback

### Step 6. Expand Tests

- add deterministic planner tests for keep/decompose/block decisions
- add fallback-eligibility tests
- update depth tests to reflect IR-aware eligibility
- add assertions that metadata/query operations stay atomic
- add assertions that no invented outputs/state appear in children

## Concrete File Plan

Expected edits:

- `sciona/ingester/models.py`
  - additive planning models if kept in-model
- `sciona/ingester/chunker.py`
  - canonical deterministic planner
  - IR-aware decomposition decisions
  - strict LLM fallback gating and validation
- `sciona/ingester/control_flow_decomposer.py`
  - only if needed to expose stronger evidence for IR mapping
- `sciona/ingester/prompts.py`
  - constrain LLM decomposition input/output around canonical IR
- tests
  - `tests/test_chunker_depth.py`
  - `tests/test_ingester_chunker.py`
  - possibly new focused planning tests

## Regression Risks

Primary risks:

- planner becomes too conservative and stops useful decomposition
- CFG decomposition is accepted without faithful IR mapping
- legacy child-atom tree diverges from canonical planning decisions
- sklearn-focused rules harm existing DSP/stateful wrappers

Mitigations:

- keep deterministic keep-atomic as the safe default
- treat blocked ambiguity as a valid outcome
- require evidence-backed child/output/state creation
- benchmark deterministic-vs-LLM fallback counts across protected families

## Test and Benchmark Plan

Direct planner tests:

- metadata/query operation remains atomic
- single-method predict operation remains atomic
- multi-method state-transition operation decomposes deterministically
- operation with unknown bindings becomes `blocked_unknown`
- LLM fallback is skipped when eligibility rules fail

Depth/recursion tests:

- semantic eligibility overrides naive line-count heuristics
- `max_depth=1` remains backward compatible
- recursion stops when children are semantically atomic

Compatibility tests:

- planned groups adapt back to valid `MacroAtomSpec.children`
- no child output is introduced without parent IR evidence
- no child state dependency appears outside known state slots

Curated acceptance cases:

- sklearn-style estimator fixture
- BioSPPy ECG wrapper
- rolling/windowed stateful class

## Acceptance Criteria

Phase 3 is complete when all of the following are true:

- deterministic planning runs over canonical IR, not raw macro-atoms
- common OO patterns are decomposed or kept atomic without LLM help
- metadata/query operations are not decomposed into fake computational stages
- no child operation introduces invented outputs or state fields
- LLM fallback is gated by explicit eligibility rules
- blocked ambiguity surfaces as artifacts instead of semantic invention
- legacy child-atom adapters still keep the existing pipeline working
- protected regression cases pass with lower LLM dependence

## Deferred to Later Phases

Phase 4:

- emitter rewrite over canonical IR/planned groups

Phase 5:

- verification and repair narrowing after semantic planning is stabilized

## Recommended Execution Order

1. Add planning models and regression fixtures.
2. Implement deterministic keep/decompose/block rules over IR.
3. Integrate CFG decomposition as a constrained subroutine.
4. Restrict and validate LLM fallback.
5. Adapt planned groups back to legacy macro-atom trees.
6. Only then begin phase-4 emitter work.
