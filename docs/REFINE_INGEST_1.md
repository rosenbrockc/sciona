# REFINE_INGEST Phase 1 Implementation Plan

> Archived: this planning document is historical. See `docs/REFINE_INGEST_STATUS.md` for the active references and `../ageo-atoms/REFINE_INGEST.md` for the current shared brief.


## Phase Goal

Phase 1 delivers a deterministic semantic fact extraction layer for ingest.
Its job is to recover source-of-truth facts from code before chunking,
decomposition, hoisting, or emission make grouping decisions.

The output of this phase is not a new emitter and not a new macro-atom IR.
It is a provenance-backed fact model that can answer, with explicit unknowns:

- what callable signatures are real
- which methods call which other methods
- which `self.*` attributes are read, written, or returned
- which attributes are config state vs learned/fitted state
- whether a method is a mutator, predictor, query, helper, or unknown
- what return behavior is actually visible in source

## Scope Boundaries

In scope:

- deterministic extraction for Python class/function ingestion
- provenance for each extracted fact
- explicit unknown markers where static analysis cannot prove something
- a compatibility path so current chunker/hoister code keeps working
- benchmark fixtures for one sklearn-style class and protected existing families

Out of scope:

- redesigning `MacroAtomSpec`
- rewriting chunking/decomposition prompts
- changing wrapper emission semantics
- broad repair-loop changes
- full multi-language semantic parity beyond keeping interfaces compatible

Required prerequisite:

- add a lightweight regression harness slice first, as suggested in
  `../ageo-atoms/REFINE_INGEST.md`, so phase-1 changes can be measured before
  phase-2+ work starts

## Current Code Touchpoints

Primary modules:

- `sciona/ingester/models.py`
- `sciona/ingester/extractor.py`
- `sciona/ingester/python_extractor.py`
- `sciona/ingester/graph.py`

Compatibility-sensitive consumers:

- `sciona/ingester/chunker.py`
- `sciona/ingester/ast_state_hoister.py`
- `sciona/ingester/prompts.py`

Cross-language parity surfaces:

- `sciona/ingester/treesitter_extractor.py`
- `sciona/ingester/base_extractor.py`

Current tests that define protected behavior:

- `tests/test_ingester_extractor.py`
- `tests/test_ingest_config_flatten.py`
- `tests/test_ingest_stateful.py`
- `tests/test_ingest_biosppy_ecg.py`
- `tests/test_ingest_dl_boundary.py`
- `tests/test_ingest_procedural.py`
- tree-sitter and FFI coverage under `tests/test_treesitter_*.py` and
  `tests/test_ffi_emitter.py`

## Current Gaps

The existing `RawDataFlowGraph` is useful but too weak for the phase target.

Observed limitations in the current extractor stack:

- method signatures are reduced to plain parameter names
- defaults, keyword-only args, varargs, decorators, and exact annotations are
  not modeled strongly enough
- attribute facts are flattened into name lists without provenance or role
- return behavior is reduced to a string annotation, not actual return sources
- call graph extraction records only callee names, not call-site facts
- config detection is mostly a container-name heuristic from `__init__`
- learned/fitted attributes are not separated from generic mutable state
- role classification needed by later phases does not exist yet
- consumers such as `chunker.py` directly depend on the simplified shape

## Phase 1 Deliverables

### 1. Semantic Fact Model

Extend `sciona/ingester/models.py` with a deterministic fact schema that sits
under or alongside `RawDataFlowGraph`.

Planned additions:

- `SourceSpan`
  - file path, line/column start/end
- `FactProvenance`
  - source span
  - extraction rule id
  - evidence text or normalized AST snippet
- `UnknownFact`
  - reason code such as `dynamic_getattr`, `dynamic_setattr`,
    `unresolved_indirect_call`, `mixed_return_paths`
- `ParameterFact`
  - name
  - kind: positional-only, positional-or-keyword, vararg, kw-only, kwarg
  - annotation text
  - default expression text
  - has_default
- `ReturnFact`
  - kind: `none`, `self`, `attribute`, `call_result`, `tuple`, `constant`,
    `parameter`, `unknown`, `mixed`
  - referenced attrs
  - referenced callees
- `CallFact`
  - callee expression text
  - resolved local method target when known
  - line number / span
  - arg expression summaries
- `AttributeFact`
  - attribute name
  - declared or first-seen location
  - written in `__init__`
  - written outside `__init__`
  - read outside `__init__`
  - config-origin flag
  - learned/fitted flag
  - query-only flag when applicable
- `MethodSemanticFact`
  - exact signature
  - decorators
  - reads/writes with provenance
  - internal/external calls
  - return facts
  - role classification
  - explicit unknowns
- `SemanticFactGraph`
  - class/function/module level fact container
  - provenance-indexed attribute inventory
  - internal call graph
  - config inventory
  - learned/fitted inventory
  - compatibility projection into current `RawDataFlowGraph`

Design rule:

- keep `RawDataFlowGraph` available during rollout
- add a deterministic projection layer rather than forcing phase-2 consumers to
  understand the richer schema immediately

### 2. Python Extractor V2

Refactor `sciona/ingester/extractor.py` into a richer semantic extractor while
keeping its existing API stable.

Planned extraction responsibilities:

- exact method signature extraction from `ast.arguments`
- constructor parameter inventory and parameter-to-attribute binding
- precise `self.*` read/write tracking with spans and statement provenance
- direct internal call facts with call-site metadata
- return source analysis from `return` statements
- detection of tuple returns and returned attributes
- detection of `return self`
- recognition of dynamic-analysis escape hatches:
  - `getattr`
  - `setattr`
  - `hasattr`
  - delegated calls through aliases
  - star-arg forwarding that prevents exact binding
- unknown-fact emission instead of silent guessing

Analysis boundary:

- intraprocedural plus simple inter-method propagation only
- no symbolic execution
- no LLM inference

### 3. Deterministic Role Classification

Add a small rule engine in phase 1 rather than waiting for phase 3.
Later phases need better facts about method intent, and some of that intent is
visible deterministically.

Initial role labels:

- `constructor`
- `config_setter`
- `fit_or_update`
- `predict_or_transform`
- `score_or_evaluate`
- `query_or_metadata`
- `helper`
- `unknown`

Rules should use only extracted evidence:

- name patterns such as `fit`, `partial_fit`, `update`, `predict`, `transform`,
  `score`, `get_`, `is_`, `has_`, `__sklearn_tags__`,
  `get_metadata_routing`
- presence or absence of writes
- return behavior
- whether learned attrs are written
- visibility (`_private` helper vs public API)

Important rule:

- if evidence conflicts, emit `unknown`; do not guess

### 4. Config vs Learned/Fitted State Inventory

The extractor should separate at least three attribute categories:

- config state
  - values copied from constructor params or config containers
- learned/fitted state
  - attributes first written by `fit`/`update`-style methods or by methods
    classified as state transitions
- derived/transient artifacts
  - attributes written and consumed as intermediate outputs

Minimum deterministic heuristics for phase 1:

- `self.attr = param` in `__init__` marks config-origin
- writes outside `__init__` in `fit`/`partial_fit`/`update`-style methods mark
  learned/fitted candidates
- attrs read by predict/query methods and written only by mutator methods are
  strong fitted-state candidates
- attrs only read inside a method that also writes them remain mutable/transient
  unless later evidence upgrades them

### 5. Compatibility Adapter

Phase 1 should not force a same-turn rewrite of `chunker.py` or
`ast_state_hoister.py`.

Deliver a projection layer that derives the current fields from the new fact
model:

- `methods`
- `all_attributes`
- `config_branches`
- `init_chain`
- `cross_window_attrs`
- `internal_call_graph`

This keeps the pipeline running while later phases adopt richer semantics.

## Required Interfaces With Other Phases

Interface to phase 0 / harness work:

- benchmark fixtures and golden fact assertions must land first or alongside
  the first extractor refactor

Interface to phase 2:

- `SemanticFactGraph` must expose enough structure for a future OO-aware ingest
  IR without forcing the phase-2 planner to recover facts from raw source again

Interface to phase 3:

- method role labels
- real call signatures
- explicit unknown markers
- provenance on introduced config/fitted fields

Interface to phase 4:

- return facts must distinguish:
  - method return value
  - returned `self`
  - returned attribute
  - tuple returns

## Deterministic vs LLM Responsibilities

Deterministic in phase 1:

- signatures
- decorators
- constructor/config inventory
- `self.*` reads and writes
- internal call graph
- return behavior visible in source
- config/fitted/transient candidate labeling
- method role classification when rule evidence is sufficient
- unknown markers when rule evidence is insufficient

Explicitly not LLM work in phase 1:

- inferring missing signatures
- inventing attributes or outputs
- deciding whether dynamic code "probably" mutates some state
- classifying ambiguous methods without deterministic evidence

## Rollout Plan

### Step 0. Add the Lightweight Regression Slice

Before changing extractor behavior, add a small benchmark matrix dedicated to
semantic fact extraction.

Required slices:

- sklearn-style estimator fixture
  - prefer a checked-in source fixture modeled on `CalibratedClassifierCV`
    rather than importing a live sklearn dependency
- flat functional/procedural fixture
- existing stateful rolling/windowed fixture
- existing biosignal wrapper fixture
- one FFI/tree-sitter fixture
- opaque DL boundary fixture

Outputs to snapshot:

- extracted signatures
- role labels
- config attrs
- learned/fitted attrs
- internal call graph
- return facts
- unknown-fact inventory

### Step 1. Introduce New Models Without Breaking Callers

- add the new fact models to `sciona/ingester/models.py`
- keep `RawDataFlowGraph` stable
- add conversion helpers such as `semantic_to_raw_dfg(...)`
- update `graph.py` state payloads only enough to carry the richer object where
  useful for debugging and monitor snapshots

### Step 2. Refactor Python Extraction Around Provenance

- split `extractor.py` visitors into focused passes:
  - signature extraction
  - attribute access extraction
  - call extraction
  - return extraction
  - constructor/config binding extraction
  - role classification
- remove places where facts are immediately collapsed into plain strings if that
  would discard provenance needed by later phases

### Step 3. Compute Attribute Inventories Deterministically

- build per-attribute facts from per-method evidence
- distinguish constructor-only, mutator-written, and query-read attrs
- compute learned/fitted candidates from write/read topology and role labels
- preserve current `cross_window_attrs` as a compatibility projection, not the
  source of truth

### Step 4. Wire Through the Ingest Entry Points

- `PythonASTExtractor.extract_class` should return the enriched fact model
  projected into `RawDataFlowGraph`
- `graph.py` should preserve the richer debug artifact where practical
- `chunker.py` should continue to function unchanged at first, but it may read
  optional richer facts if that is a low-risk improvement

### Step 5. Expand Test Coverage and Lock Golden Cases

- add direct extractor tests for signature fidelity and return facts
- add curated regression cases for query-vs-mutator classification
- add fixtures for dynamic escape hatches and ensure they produce unknowns
- add end-to-end ingest smoke tests proving no regression for protected families

## Concrete File Plan

Expected phase-1 edits:

- `sciona/ingester/models.py`
  - add new fact/provenance models
- `sciona/ingester/extractor.py`
  - implement extractor v2 and compatibility projection
- `sciona/ingester/python_extractor.py`
  - return/project enriched facts
- `sciona/ingester/graph.py`
  - preserve richer debug state and artifact publication
- `sciona/ingester/chunker.py`
  - only minimal compatibility changes if required
- `tests/test_ingester_extractor.py`
  - expand beyond reads/writes into signatures, returns, roles, unknowns
- new curated tests/fixtures
  - likely under `tests/fixtures/ingest_refine/`
  - plus new focused extractor tests

Possible but optional phase-1 edits:

- `sciona/ingester/treesitter_extractor.py`
  - only if adding optional fields requires parity stubs or shared helpers

## Regression Risks

Primary risks:

- phase-1 model changes silently break chunker assumptions
- sklearn-specific heuristics degrade current stateful/Bayesian/DSP flows
- Python-only assumptions leak into language-agnostic extractor interfaces
- richer extraction increases artifact size enough to hurt monitor/debug flows

Mitigations:

- keep current `RawDataFlowGraph` as an adapter surface during rollout
- prefer additive fields over replacing existing ones immediately
- benchmark against non-sklearn fixtures in the same test run
- treat dynamic cases as unknown, not as guessed facts
- gate optional richer-consumer usage behind local helpers until phase 2 begins

## Test and Benchmark Plan

Direct extractor unit tests:

- exact signature extraction including kw-only, defaults, `*args`, `**kwargs`
- return classification for:
  - `return self`
  - `return self.attr`
  - `return helper(...)`
  - tuple returns
  - mixed/ambiguous returns
- config-origin detection from constructor bindings
- learned/fitted detection from `fit` then `predict`
- query-vs-mutator classification for metadata methods
- unknown-fact emission for `getattr`/`setattr` patterns

Integration tests to preserve existing families:

- `RollingAverager` style stateful class
- BioSPPy ECG wrapper
- opaque DL boundary class
- procedural ingest fixture
- one tree-sitter/FFI case already covered in repo

Curated sklearn-style acceptance fixture:

- add a checked-in estimator-like class mirroring the failure modes from
  `CalibratedClassifierCV`
- assert:
  - true constructor signature
  - fit/update methods own learned attrs
  - metadata/query methods are not treated as mutators
  - prediction/query methods read learned state but do not invent outputs

## Acceptance Criteria

Phase 1 is complete when all of the following are true:

- deterministic extraction produces a provenance-backed semantic fact graph
- every extracted fact is either proven or explicitly marked unknown
- exact callable signatures are preserved for Python class methods
- method-to-method call graph includes direct call-site evidence
- config attrs and learned/fitted attrs are inventoried separately
- role classification exists and leaves ambiguous cases as unknown
- the current ingest pipeline still passes protected non-sklearn tests
- curated sklearn-style fixtures demonstrate objective improvement over the
  current read/write-only model
- phase-2 work can consume the new fact model without re-parsing source

## Deferred to Later Phases

Phase 2:

- OO-aware ingest IR redesign

Phase 3:

- deterministic decomposition policies based on the richer fact graph

Phase 4:

- faithful wrapper/state emission over the new IR

Phase 5:

- narrowing repair semantics around mechanical failures only

## Recommended Execution Order

1. Add the lightweight regression harness slice.
2. Land additive fact/provenance models.
3. Refactor Python extraction and keep the old projection surface.
4. Add sklearn-style and protected-family benchmarks.
5. Only then begin phase-2 IR planning/implementation.
