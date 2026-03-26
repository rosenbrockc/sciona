# REFINE_INGEST Phase 7 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 7 extends the refined ingest semantics beyond Python so protected
non-Python and FFI-backed families participate in the same canonical path.

Phases 1 through 5 made semantic extraction, IR, planning, emission, and
verification much more faithful for Python. Phase 6 added the harness proving
that non-Python families are protected. The remaining gap is that Rust, Julia,
C++, and FFI-backed cases still mostly survive through compatibility behavior
instead of the canonical semantic models.

The objective is:

- preserve current non-Python ingest success
- add deterministic semantic facts for tree-sitter languages where source can
  prove them
- lower those facts into canonical IR and planning metadata
- let emission and verification consume that canonical information without
  assuming Python-only object semantics

Key rule:

- non-Python ingestion should gain semantic structure without pretending Rust,
  Julia, or C++ behave like Python classes

## Scope Boundaries

In scope:

- additive semantic-fact enrichment for tree-sitter extractors
- canonical IR lowering for non-Python function/module cases
- deterministic planning parity for flat and stateful non-Python cases where
  evidence is sufficient
- non-Python emission parity where canonical IR can already express the needed
  behavior
- focused regression coverage in the phase-6 harness and existing tree-sitter /
  FFI tests

Out of scope:

- a universal cross-language alias analysis engine
- deep control-flow or borrow-checker-level reasoning for Rust/C++
- redesigning the FFI emitter from scratch
- broad language support beyond the existing tree-sitter languages already in
  repo
- forcing Python OO state-slot assumptions onto procedural/module-style sources

## Current Code Touchpoints

Primary implementation surfaces:

- `sciona/ingester/treesitter_extractor.py`
- `sciona/ingester/models.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/ffi_emitter.py`

Regression and harness surfaces:

- `sciona/ingester/regression_harness.py`
- `tests/test_treesitter_rust.py`
- `tests/test_treesitter_cpp.py`
- `tests/test_treesitter_julia.py`
- `tests/test_ffi_emitter.py`
- `tests/test_ingest_regression_harness.py`
- `tests/test_ingest_procedural.py`
- `tests/test_message_passing.py`
- `tests/test_bayesian_ingester.py`

Related existing canonical-path surfaces:

- `sciona/ingester/extractor.py`
- `sciona/ingester/python_extractor.py`
- `sciona/ingester/graph.py`

## Current Gaps

The current system now has a good Python semantic path and a good regression
story, but non-Python sources still have notable asymmetry.

Observed issues:

- tree-sitter extraction still returns mostly legacy `MethodFact` shape without
  the richer provenance-backed semantic facts used by the Python path
- canonical IR lowering mostly reflects Python-oriented state/method semantics
- phase-3 planning is canonical-first, but many non-Python cases reach it with
  thin semantic evidence
- phase-4 emission prefers canonical IR only for Python, while non-Python
  wrappers mostly fall back to generic FFI generation
- the phase-6 harness can detect regressions in non-Python cases, but it cannot
  yet assert much canonical semantic improvement there

This leaves an undesirable state:

- Python benefits from the refined semantic middle
- non-Python is protected mainly by compatibility and tests, not by equivalent
  semantic modeling

## Phase 7 Deliverables

### 1. Tree-Sitter Semantic Fact Parity

Extend tree-sitter extraction so it can populate the additive semantic-fact
fields introduced in phase 1 where the source language supports them.

Target facts:

- richer parameter/signature facts for extracted functions/methods
- call-site facts, not only flattened callee names
- explicit return-behavior facts where syntactically visible
- provenance / source-span data for extracted facts
- explicit unknown markers when the extractor cannot safely classify a fact
- deterministic role labels for common non-Python patterns:
  - state update / mutator
  - query / pure view
  - helper / internal function
  - oracle / stochastic interface where existing tree-sitter logic already
    proves it

The goal is parity of *shape* and *truthfulness*, not forcing all languages to
yield the same density of facts.

### 2. Canonical IR Lowering for Non-Python Cases

Upgrade phase-2 lowering so canonical IR can be built from non-Python semantic
facts without assuming Python instance attributes.

Required capabilities:

- procedural/module-level operations can lower into `OperationSpec`
- output bindings can reference direct return values and explicit derived
  artifacts from non-Python extraction
- state slots are optional and evidence-driven, not assumed
- oracle/message/stochastic subgraphs already surfaced by tree-sitter extractors
  can survive into canonical IR metadata

Important constraint:

- if a non-Python case does not have enough evidence for a canonical stateful
  interpretation, lower it as a flat or procedural operation instead of
  inventing pseudo-object state

### 3. Planning Parity for Non-Python Canonical IR

Upgrade phase-3 planning so non-Python canonical IR cases receive deterministic
keep/decompose/block decisions using the same evidence-first philosophy.

Priority behaviors:

- keep simple module-level functions atomic
- decompose only when canonical call graph evidence justifies it
- preserve oracle/message/stochastic boundaries as explicit operations
- block ambiguous stateful decomposition rather than inventing language-specific
  object semantics

### 4. Canonical Emission Parity for Non-Python

Phase 4 intentionally left canonical emission Python-only. Phase 7 should close
part of that gap without replacing the current FFI emitter wholesale.

Required changes:

- allow `emit_ingestion_bundle(...)` to consume canonical IR metadata for
  non-Python sources where it improves wrapper fidelity
- keep using `ffi_emitter.py` for import/binding scaffolding where appropriate
- source wrapper outputs from canonical output bindings when known
- keep fail-closed behavior on underspecified outputs instead of guessing

The emitter may still rely on FFI scaffolding, but it should stop ignoring
available canonical evidence just because the source language is not Python.

## Required Interfaces With Prior Phases

Interface from phase 1:

- non-Python extractors should populate the same additive fact containers where
  possible, including explicit unknowns

Interface from phase 2:

- canonical IR must remain additive and backward-compatible for current
  consumers
- legacy adapter paths must continue to work during the transition

Interface from phase 3:

- planner fallbacks and blocked ambiguity must remain explicit artifacts

Interface from phase 4:

- non-Python emission should use canonical bindings when available, but must
  preserve current FFI/procedural behavior when canonical evidence is absent

Interface from phase 5:

- verification still fails fast on semantic mismatches
- non-Python canonical underspecification should surface as design failure, not
  as a generic repair opportunity

Interface from phase 6:

- extend the regression harness so the non-Python / FFI case can assert more
  than “did not regress”
- track whether canonical IR is now present for selected tree-sitter cases

## Deterministic vs LLM Responsibilities

Deterministic in phase 7:

- tree-sitter semantic fact extraction
- non-Python canonical IR lowering
- evidence-gated decomposition planning
- canonical output/state binding for non-Python emission where supported
- regression harness assertions

LLM responsibilities in phase 7:

- none for the new cross-language semantic extraction and lowering itself
- at most, existing bounded fallback paths may still appear for unrelated
  decomposition/naming cases, but phase 7 should not introduce new LLM reliance

## Data Model Changes

Expected additive model work:

- extend tree-sitter-generated `MethodFact` population to fill existing
  phase-1 semantic fields
- add any small canonical metadata needed to distinguish procedural/module-level
  non-Python operations from Python OO operations
- preserve defaults so older tests and compatibility paths do not break

Avoid:

- introducing language-specific one-off models unless the existing generic
  schema cannot express the fact cleanly

## Rollout Plan

### Step 0. Lock Non-Python Regression Targets

Before semantic changes, confirm the phase-6 harness and direct tests cover:

- one Rust tree-sitter case
- one Julia or C++ tree-sitter case
- one FFI-emitter case
- one procedural fallback case

Phase 7 should expand expectations on those cases rather than invent new broad
coverage first.

### Step 1. Enrich Tree-Sitter Semantic Facts

- populate provenance/span data where already available from tree-sitter nodes
- populate parameter facts and return facts for Rust/Julia/C++ functions
- add explicit unknown markers for dynamic or unsupported constructs
- preserve current oracle/message/stochastic extraction behavior

### Step 2. Lower Non-Python Facts Into Canonical IR

- extend canonical lowering helpers in `chunker.py`
- ensure selected tree-sitter cases now produce `canonical_ir`
- keep legacy macro-atom/state-model adapters working

### Step 3. Add Deterministic Planning Parity

- run existing keep/decompose/block logic over the richer non-Python IR
- keep atomic defaults strong for procedural/module-level cases
- block ambiguous object-like decomposition in non-Python languages unless
  evidence is explicit

### Step 4. Consume Canonical Metadata in Emission

- route non-Python emission through canonical output/state metadata where
  possible
- retain `ffi_emitter.py` as the binding/backend layer, not the semantic source
  of truth
- fail closed on underspecified canonical outputs

### Step 5. Expand Tests and Harness Expectations

- add tree-sitter tests proving semantic-fact population
- add chunker/emitter tests proving non-Python canonical lowering/emission
- extend regression harness expectations for at least one non-Python case to
  assert canonical IR presence and stable semantic checks

## Concrete File Plan

Expected edits:

- `sciona/ingester/treesitter_extractor.py`
- `sciona/ingester/chunker.py`
- `sciona/ingester/emitter.py`
- `sciona/ingester/models.py`
- `sciona/ingester/regression_harness.py`
- tests:
  - `tests/test_treesitter_rust.py`
  - `tests/test_treesitter_cpp.py`
  - `tests/test_treesitter_julia.py`
  - `tests/test_ffi_emitter.py`
  - `tests/test_ingester_chunker.py`
  - `tests/test_ingester_emitter.py`
  - `tests/test_ingest_regression_harness.py`

Prefer keeping the write scope narrow:

- extraction facts
- canonical lowering
- emission consumption of canonical bindings
- direct tests

## Regression Risks

Primary risks:

- Python-centric canonical assumptions leak into non-Python cases and break
  existing FFI behavior
- tree-sitter enrichments overclaim facts that are not actually proven
- canonical emission for non-Python becomes too clever and diverges from current
  reliable scaffolding
- planner rules tuned for Python OO over-decompose procedural Rust/Julia/C++
  code

Mitigations:

- default to explicit unknowns when extraction cannot prove a fact
- prefer flat/procedural lowering over invented state
- keep non-Python emission fail-closed on missing bindings
- require tree-sitter and FFI tests to stay green alongside the phase-6 harness

## Test and Benchmark Plan

Direct extraction tests:

- selected Rust/Julia/C++ fixtures assert signature facts, return facts, call
  facts, and unknown handling where applicable

Canonical-path tests:

- selected non-Python case now lowers to canonical IR
- planner keeps atomic procedural cases atomic
- emitter uses canonical output bindings for supported non-Python cases

Protected-family regression slice:

- `tests/test_treesitter_rust.py`
- `tests/test_treesitter_cpp.py`
- `tests/test_treesitter_julia.py`
- `tests/test_ffi_emitter.py`
- `tests/test_ingest_regression_harness.py`
- `tests/test_ingest_procedural.py`
- any adjacent Bayesian/message-passing tests affected by oracle handling

Phase-6 harness expectation:

- at least one non-Python case should move from language-only checks to
  canonical-semantic checks, such as `has_canonical_ir`

## Acceptance Criteria

Phase 7 is complete when all of the following are true:

- selected tree-sitter languages populate additive semantic facts with explicit
  unknowns rather than only legacy flattened facts
- canonical IR is produced for representative non-Python cases where evidence
  is sufficient
- deterministic planning remains conservative and does not invent Python-style
  object semantics for procedural/module code
- non-Python emission consumes canonical bindings when available and otherwise
  preserves current FFI/procedural behavior
- the phase-6 harness can assert canonical semantic improvement on at least one
  non-Python / FFI case
- existing tree-sitter, FFI, procedural, and protected-family regressions stay
  green

## Deferred to Later Work

Not required in phase 7:

- fully feature-complete cross-language semantic parity with Python OO cases
- deeper Rust borrow/type reasoning
- generalized alias analysis for C++
- major FFI backend redesign
- adding entirely new languages beyond current tree-sitter support

Phase 7 should establish the canonical cross-language path, not exhaust it.
