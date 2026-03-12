# Deterministic Tools — Implementation Plan

Implementation plan for the 10 missing deterministic tools identified in
[TOOLS.md](TOOLS.md). Each tool follows the standard `LLMClient` protocol
wrapper pattern: try deterministic analysis first, fall back to the wrapped
LLM on low confidence.

Tools are grouped into three implementation waves based on dependency ordering,
code reuse potential, and value-to-effort ratio.

---

## Wave 1 — High-value, low-risk (reuse existing infrastructure)

These three tools reuse infrastructure that already exists in the codebase.
They are the fastest to implement and cover the highest-traffic prompt keys
among the uncovered set.

### Tool 1: `ingester_fix_type` — DeterministicTypeFixer

**Prompt key**: `ingester_fix_type`
**File**: `ageom/ingester/deterministic_type_fixer.py`
**Wiring**: `ageom/commands/ingest_cmds.py` (override in LLM router)

**Rationale**: The ingester's `repair_types` node receives mypy errors and
generated source code, then asks the LLM for line-level patches. The
`classifier.py` already has `classify_error()` and `suggest_deterministic_fix()`
that handle the same error patterns — they just aren't wired into the ingester
path.

**Implementation**:

```
class DeterministicTypeFixer:
    """LLMClient wrapper for ingester_fix_type."""

    async def complete(self, system: str, user: str) -> str:
        mypy_errors, source_code = _parse_fix_type_prompt(user)
        fixes = []
        for error_line in mypy_errors.splitlines():
            category = classify_error(error_line)
            fix = suggest_deterministic_fix(category, error_line)
            if fix is not None:
                line_num = _extract_line_number(error_line)
                fixes.append({"line_start": line_num, "line_end": line_num,
                               "replacement": fix})
        if fixes:
            return json.dumps(fixes)
        return await self._fallback.complete(system, user)
```

**Steps**:

1. Create `ageom/ingester/deterministic_type_fixer.py`:
   - `_parse_fix_type_prompt(user)` — extract mypy errors and source code from
     the `FIX_TYPE_ERROR_USER` format (split on `"mypy errors:"` and
     `"Generated source:"` markers).
   - `_extract_line_number(error_line)` — regex to pull line number from
     standard mypy output format (`file.py:42: error: ...`).
   - For each error line: call `classify_error()` → `suggest_deterministic_fix()`.
   - For `MISSING_IMPORT` fixes: insert as new lines at top of file (line_start=1,
     line_end=1, replacement=fix).
   - For `TYPE_MISMATCH` fixes with known coercions: locate the expression on the
     error line and wrap it.
   - Return JSON array of patches matching the expected format.
   - Fall back to LLM if no deterministic fixes found or if any error is UNKNOWN.

2. Wire into `ageom/commands/ingest_cmds.py`:
   - Import `DeterministicTypeFixer` and `INGESTER_FIX_TYPE`.
   - After building the LLM router, wrap the `INGESTER_FIX_TYPE` client.

3. Tests:
   - Unit test: parse mypy output → correct fix patches.
   - Unit test: unknown errors → fallback called.
   - Integration: run through `repair_types` node with mock env.

**Estimated effort**: Small — mostly glue between existing `classifier.py` and
the ingester prompt format.

---

### Tool 2: `synthesizer_tactic` — DeterministicTacticSuggester

**Prompt key**: `synthesizer_tactic`
**File**: `ageom/synthesizer/tactic_suggester.py`
**Wiring**: `ageom/commands/synthesize_cmds.py` (override in LLM router)

**Rationale**: The `SorryElimination` node in the repair graph finds
sorry/Admitted/NotImplementedError locations, extracts the goal type from
surrounding context, and asks the LLM for a tactic or implementation. Many of
these cases are mechanical:

- Lean: `rfl`, `simp`, `omega`, `norm_num`, `decide` cover a large fraction
  of sorry stubs generated during assembly.
- Python: `raise NotImplementedError` on a function with a known matched
  library function → emit a delegation call to the matched function.
- Coq: `reflexivity`, `auto`, `omega`, `lia` cover simple Admitted stubs.

**Implementation**:

```
class DeterministicTacticSuggester:
    """LLMClient wrapper for synthesizer_tactic."""

    async def complete(self, system: str, user: str) -> str:
        goal_type, hypotheses, prover = _parse_tactic_prompt(system, user)
        tactic = _suggest_tactic(goal_type, hypotheses, prover)
        if tactic is not None:
            return tactic
        return await self._fallback.complete(system, user)
```

**Steps**:

1. Create `ageom/synthesizer/tactic_suggester.py`:
   - `_parse_tactic_prompt(system, user)` — extract goal type, hypotheses, and
     prover from the `GENERATE_TACTIC_USER` format. Detect Python vs Lean/Coq
     from the system prompt (check for `GENERATE_IMPLEMENTATION_SYSTEM_PYTHON`
     vs `GENERATE_TACTIC_SYSTEM`).
   - `_suggest_tactic(goal_type, hypotheses, prover)` — pattern-match on the
     goal type:

     **Lean 4 patterns** (checked in order):
     | Goal pattern | Tactic |
     |-------------|--------|
     | `_ = _` where both sides are identical | `rfl` |
     | `_ = _` with Nat/Int arithmetic | `omega` |
     | `_ = _` with numeric literals | `norm_num` |
     | `_ ∈ _` or `_ < _` or `_ ≤ _` with literals | `omega` or `norm_num` |
     | `Decidable _` | `infer_instance` |
     | `True` or `⊤` | `trivial` |
     | `_ ∧ _` where both sides are in hypotheses | `exact ⟨‹_›, ‹_›⟩` |
     | `_ → _` where conclusion is in hypotheses | `exact ‹_›` |
     | Anything else with `Nat` or `Int` | `omega` |
     | Generic fallback | `simp` |

     **Coq patterns**:
     | Goal pattern | Tactic |
     |-------------|--------|
     | `_ = _` where both sides are identical | `reflexivity.` |
     | `_ = _` with nat arithmetic | `omega.` or `lia.` |
     | Generic | `auto.` |

     **Python patterns**:
     | Context | Implementation |
     |---------|---------------|
     | Function with matched library name in context | `return matched_func(...)` |
     | Function with type hint `-> bool` | `return True  # TODO` |
     | Generic | Fall back to LLM |

   - Confidence gate: if the goal type is empty or contains complex
     quantifiers/induction, return `None` (fall back to LLM).
   - The suggested tactic must NOT contain `sorry`/`Admitted` (the repair
     node already checks this, but we enforce it here too).

2. Wire into `ageom/commands/synthesize_cmds.py`:
   - Import `DeterministicTacticSuggester` and `SYNTHESIZER_TACTIC`.
   - Wrap the `SYNTHESIZER_TACTIC` client in the LLM router.

3. Tests:
   - Unit tests for each pattern (rfl, omega, norm_num, simp, reflexivity).
   - Unit test: complex goal → fallback.
   - Integration: run `SorryElimination` with simple goals and verify
     deterministic resolution.

**Estimated effort**: Medium — the pattern matching is straightforward, but
there are many patterns to cover and test. The Lean tactic patterns can be
iterated on incrementally.

---

### Tool 3: `ingester_hoist_state` — ASTStateHoister

**Prompt key**: `ingester_hoist_state`
**File**: `ageom/ingester/ast_state_hoister.py`
**Wiring**: `ageom/commands/ingest_cmds.py` (override in LLM router)

**Rationale**: The `hoist_state` chunker node receives cross-window attributes
(instance variables shared across methods) and a macro-atom plan, and asks the
LLM to group them into `StateModelSpec`s. For Python classes with typed
`__init__`, this is entirely derivable from the AST:

- `self.x = value` in `__init__` → field with inferred type
- `self.x: Type = value` → field with annotated type
- `__init__` parameter types → initial field types
- Grouping: all fields on a single class → single StateModelSpec

**Implementation**:

```
class ASTStateHoister:
    """LLMClient wrapper for ingester_hoist_state."""

    async def complete(self, system: str, user: str) -> str:
        cross_window_attrs, macro_plan = _parse_hoist_prompt(user)
        result = _hoist_from_attrs(cross_window_attrs, macro_plan)
        if result is not None:
            return json.dumps(result)
        return await self._fallback.complete(system, user)
```

**Steps**:

1. Create `ageom/ingester/ast_state_hoister.py`:
   - `_parse_hoist_prompt(user)` — extract `cross_window_attrs` (the raw
     attribute list) and `macro_plan_json` from the `HOIST_STATE_USER` format.
   - `_hoist_from_attrs(cross_window_attrs, macro_plan)`:
     - Parse `cross_window_attrs` — this is a list of attribute dictionaries
       from the `RawDataFlowGraph.cross_window_attrs` field. Each contains
       at minimum the attribute name; some contain type annotations.
     - For each attribute, extract:
       - Field name: `attr_name` (strip `self.` prefix if present)
       - Type annotation: from the attribute's `type_annotation` field, or
         infer from default value (`int`, `float`, `str`, `list`, `dict`,
         `None` → `Optional[...]`).
       - Source attr: `self.{attr_name}`
     - Group fields into StateModelSpecs:
       - If macro_plan has a single class, emit one `{ClassName}State` model.
       - If macro_plan has multiple related atoms, group by which atoms
         read/write which attributes (using the atom's `method_names` to look
         up read/write sets from the DFG).
       - Fallback: one model per class with all attributes.
     - Generate `model_name` in PascalCase: `{class_name}State`.
     - Generate `docstring`: `"State for {class_name}: {field_count} fields
       hoisted from instance attributes."`.
   - Confidence gate: return `None` if:
     - `cross_window_attrs` is empty (the node already short-circuits, but
       be safe).
     - More than 50% of attributes have no type information and no inferrable
       default value.
     - Attributes use `**kwargs` unpacking or dynamic `setattr`.

2. Wire into `ageom/commands/ingest_cmds.py`:
   - Import and wrap `INGESTER_HOIST_STATE` in the router.

3. Tests:
   - Unit test: typed `__init__` with 5 attributes → correct StateModelSpec.
   - Unit test: untyped attributes → fallback.
   - Unit test: multiple classes → multiple StateModelSpecs.

**Estimated effort**: Medium — the attribute extraction is straightforward, but
the grouping heuristic needs care. The existing `_reads_writes` function from
`control_flow_decomposer.py` can be reused for read/write set computation.

---

## Wave 2 — Medium-value, structured patterns

These tools handle structured error/output patterns where the fix space is
bounded. They require new pattern libraries but follow the established tool
pattern exactly.

### Tool 4: `ingester_fix_ghost` — DeterministicGhostFixer

**Prompt key**: `ingester_fix_ghost`
**File**: `ageom/ingester/deterministic_ghost_fixer.py`
**Wiring**: `ageom/commands/ingest_cmds.py`

**Rationale**: The `repair_ghost` node receives a `GhostSimReport` error with
a node name, function name, and error message, then asks the LLM to fix the
witness function. Ghost simulation errors are highly structured — the error
types come from `ageoa.ghost.simulator` and follow predictable patterns.

**Implementation**:

**Steps**:

1. Create `ageom/ingester/deterministic_ghost_fixer.py`:
   - `_parse_ghost_prompt(user)` — extract `error_node`, `error_function`,
     `error_message`, and `witness_source` from the `FIX_GHOST_ERROR_USER`
     format.
   - `_fix_ghost_error(error_node, error_function, error_message, witness_source)`:
     - Pattern: `"Shape mismatch"` or `"shape"` in error →
       Find the witness function in `witness_source`, locate the return
       statement, and wrap the output in a reshape call:
       `AbstractSignal(shape=expected_shape, ...)`.
     - Pattern: `"domain mismatch"` → find where the witness sets `domain`
       metadata and change it to match the expected domain (time/frequency).
     - Pattern: `"TypeError"` or `"type"` in error → check if the witness
       returns the wrong abstract type (e.g., `AbstractArray` vs
       `AbstractSignal`) and fix the return type.
     - Pattern: `"KeyError"` or `"missing"` in error → add the missing field
       to the witness's return value with a sensible default.
     - Pattern: `"AttributeError"` → the witness accesses an attribute that
       doesn't exist on the abstract type. Replace with the correct attribute
       name based on known abstract type APIs.
     - For each matched pattern, emit a JSON fix with `witness_name`,
       `fix_description`, and `replacement` (the full fixed witness function).
   - Confidence gate: return `None` if the error message doesn't match any
     known pattern, or if the witness source can't be parsed.

2. Wire into ingest_cmds.

3. Tests:
   - Unit test per error pattern (shape, domain, type, key, attribute).
   - Unit test: unknown error → fallback.

**Estimated effort**: Medium — bounded fix space but requires understanding the
abstract type API from `ageoa.ghost.abstract`.

---

### Tool 5: `ingester_abstract` — TemplateAbstractor

**Prompt key**: `ingester_abstract`
**File**: `ageom/ingester/template_abstractor.py`
**Wiring**: `ageom/commands/ingest_cmds.py`

**Rationale**: The conceptual abstraction node generates domain-agnostic
descriptions of ingested atoms for future semantic search. The LLM prompt
receives the atom's name, concept type, inputs, outputs, and method names,
and returns a structured `ConceptualProfile`. For well-documented code, this
is derivable from the existing metadata.

**Implementation**:

**Steps**:

1. Create `ageom/ingester/template_abstractor.py`:
   - `_parse_abstract_prompt(user)` — extract atom name, concept type, inputs,
     outputs, and method names from the `CONCEPTUAL_ABSTRACT_USER` format.
   - `_generate_abstract(atom_name, concept_type, inputs, outputs, methods)`:
     - `abstract_name`: Convert atom name to domain-agnostic form:
       - Strip domain prefixes (`ecg_`, `audio_`, `financial_`, etc.)
       - Map to generic operation names using a lookup table:
         `bandpass_filter` → `band_selective_filter`,
         `moving_average` → `windowed_mean`, etc.
       - Fallback: `snake_to_title(atom_name)`.
     - `generic_description`: Template from concept type:
       `"A {concept_type} operation that transforms {input_types} into
       {output_types} via {method_count} processing steps."`
     - `mathematical_class`: Map concept type to math category:
       `SIGNAL_FILTER` → `linear_operator`,
       `OPTIMIZATION` → `iterative_solver`, etc.
     - `key_properties`: Derive from input/output types:
       - If inputs and outputs have same type → `idempotent_candidate`
       - If concept type is filter → `linear`, `causal`
       - If concept type is transform → `invertible_candidate`
     - `similar_operations`: Use the concept type to suggest related operations
       from a static lookup table.
   - Confidence gate: return `None` if:
     - Atom name is too generic (< 3 characters after stripping).
     - Concept type is `UNKNOWN` or missing.
     - No inputs or outputs defined.

2. Wire into ingest_cmds.

3. Tests:
   - Unit test: well-typed signal processing atom → correct profile.
   - Unit test: generic unnamed atom → fallback.

**Estimated effort**: Medium — the template tables need curation, but the
structure is mechanical. The conceptual profile format is already defined in
`chunker.py` as `ConceptualProfile`.

---

### Tool 6: `ingester_fix_message_cycle` — DeterministicCycleBreaker

**Prompt key**: `ingester_fix_message_cycle`
**File**: `ageom/ingester/deterministic_cycle_breaker.py`
**Wiring**: `ageom/commands/ingest_cmds.py`

**Rationale**: The `repair_message_cycle` node receives deadlocked nodes, cycle
edges, and witness source, then asks the LLM to break the cycle. This is a
graph algorithm problem with a bounded fix space: add damping, add convergence
epsilon, or add iteration cap.

**Implementation**:

**Steps**:

1. Create `ageom/ingester/deterministic_cycle_breaker.py`:
   - `_parse_cycle_prompt(user)` — extract `deadlock_nodes`, `cycle_edges`,
     and `witness_source` from the `FIX_MESSAGE_CYCLE_USER` format.
   - `_break_cycle(deadlock_nodes, cycle_edges, witness_source)`:
     - Parse the witness source to find the message-passing loop.
     - Strategy selection (in order):
       1. **Iteration cap**: If the witness has a `while True` or unbounded
          loop, insert `max_iter` parameter and counter check.
       2. **Convergence check**: If the witness computes messages, add
          `if np.allclose(old_msg, new_msg, atol=eps): break` after the
          message update.
       3. **Damping**: If neither applies, wrap the message update in
          `new_msg = alpha * new_msg + (1 - alpha) * old_msg` with
          `alpha=0.5`.
     - Emit JSON patches that modify the witness source at the correct line
       ranges.
   - Confidence gate: return `None` if:
     - Can't find a message-passing loop in the witness source.
     - The cycle involves more than 5 nodes (complex topology).

2. Wire into ingest_cmds.

3. Tests:
   - Unit test: simple 2-node cycle → iteration cap inserted.
   - Unit test: message-passing loop → convergence check inserted.
   - Unit test: complex cycle → fallback.

**Estimated effort**: Medium — the fix templates are simple, but parsing
arbitrary witness source to find the right insertion point requires care.

---

## Wave 3 — Higher-value, higher-complexity (extend existing tools)

These tools extend existing partially-covering tools to handle more cases. They
require more architectural care because they modify the existing tool chain
rather than adding new standalone wrappers.

### Tool 7: `architect_decompose` — Full Deterministic Decomposition

**Prompt key**: `architect_decompose`
**File**: `ageom/architect/deterministic_decompose.py` (extend existing)
**Wiring**: `ageom/commands/_helpers.py` or `ageom/commands/decompose_cmds.py`

**Rationale**: The existing `DeterministicDecompose` post-processes LLM output
but never replaces the LLM call. The `StrategyClassifier` already identifies
the paradigm with a confidence score. When that confidence is high (> 0.8),
the decomposition structure is predictable enough to emit directly from skeleton
templates.

**Implementation**:

**Steps**:

1. Extend `ageom/architect/deterministic_decompose.py`:
   - Add `class DeterministicDecomposer` implementing `LLMClient`:
     ```
     async def complete(self, system: str, user: str) -> str:
         goal, parent_node = _parse_decompose_prompt(user)
         strategy, confidence = self._strategy_classifier.classify(goal)
         if confidence < 0.8:
             return await self._fallback.complete(system, user)
         decomposition = _emit_from_skeleton(strategy, goal, parent_node)
         if decomposition is None:
             return await self._fallback.complete(system, user)
         return json.dumps(decomposition)
     ```
   - `_parse_decompose_prompt(user)` — extract goal text and parent node
     information from the `DECOMPOSE_USER` prompt format.
   - `_emit_from_skeleton(strategy, goal, parent_node)`:
     - Look up the skeleton template for the selected strategy (already in
       `skeletons.py`).
     - Instantiate the template: replace placeholder names with terms
       extracted from the goal text.
     - Generate IO specs by propagating the parent node's IO types through
       the skeleton edges.
     - Validate: each sub-node must have non-empty description and IO specs.
     - Return `None` if the skeleton can't be fully instantiated (goal text
       doesn't contain enough terms to fill the template).
   - Reuse the existing `_bind_primitives_by_token_overlap` and
     `_infer_concept_type` functions from the current module.

2. Wire as an `LLMClient` wrapper for `ARCHITECT_DECOMPOSE`:
   - In `_create_llm_router`, wrap the decompose client when the strategy
     classifier is active.

3. Tests:
   - Unit test: "merge sort" → divide-and-conquer skeleton emitted.
   - Unit test: "ecg bandpass filter" → signal filter skeleton emitted.
   - Unit test: ambiguous goal → fallback.
   - Regression: existing `DeterministicDecompose` post-processing still works.

**Estimated effort**: Large — the skeleton instantiation and IO propagation
require careful validation. Should be implemented incrementally, starting with
2-3 well-tested paradigms (divide-and-conquer, signal filter, optimization).

---

### Tool 8: `architect_critique` — Extended Deterministic Critique

**Prompt key**: `architect_critique`
**File**: `ageom/architect/structural_critic.py` (extend existing)
**Wiring**: `ageom/commands/_helpers.py` or `ageom/commands/decompose_cmds.py`

**Rationale**: The existing `structural_critique_issues` catches structural
problems but the LLM is always invoked for semantic evaluation. Several
semantic checks can be formalized as heuristics, allowing the LLM to be skipped
when all checks pass clearly.

**Implementation**:

**Steps**:

1. Add `class DeterministicCritic` implementing `LLMClient` in a new file
   `ageom/architect/deterministic_critic.py`:
   - `_parse_critique_prompt(user)` — extract the proposed decomposition
     (parent node + sub-nodes) from the `CRITIQUE_USER` prompt format.
   - `_semantic_critique(parent, sub_nodes)`:
     - **Completeness score**: Compute token overlap between parent's
       output names and the union of sub-node output names. If coverage
       > 0.8, mark as complete.
     - **Relevance score**: Compute token overlap between parent's
       description and each sub-node's description. If minimum relevance
       > 0.5, mark as relevant.
     - **Type consistency**: For each edge, check that the source output
       type string overlaps with the target input type string. Flag
       mismatches.
     - **Non-triviality**: Check that no sub-node's description has Jaccard
       similarity > 0.85 with the parent (already in structural_critic, but
       promote to a soft signal here).
     - **IO completeness**: All parent inputs must appear in at least one
       sub-node's inputs. All parent outputs must be produced by at least
       one sub-node.
   - Decision: If all semantic checks pass with high confidence AND
     `structural_critique_issues` returns no issues, return an approval
     JSON directly. Otherwise, fall back to LLM.
   - The approval format must match what the LLM would return (JSON with
     `approved: true` and `feedback` field).

2. Wire as `LLMClient` wrapper for `ARCHITECT_CRITIQUE`.

3. Tests:
   - Unit test: clean decomposition → deterministic approval.
   - Unit test: missing output coverage → fallback.
   - Unit test: trivial sub-node (duplicate of parent) → fallback.

**Estimated effort**: Medium — the heuristics are straightforward but the
confidence thresholds need tuning against the benchmark suite.

---

### Tool 9: `ingester_chunk` — Extended Deterministic Chunking

**Prompt key**: `ingester_chunk`
**File**: `ageom/ingester/chunker.py` (extend existing `propose_macro_atoms`)
**Wiring**: Already wired; extend the deterministic path in the chunker node.

**Rationale**: Currently only opaque DL classes get deterministic chunking.
For simple utility classes where each method is self-contained (no complex
cross-method control flow), chunking is mechanical: one macro-atom per public
method.

**Implementation**:

**Steps**:

1. Extend `propose_macro_atoms` in `chunker.py`:
   - After the existing `is_opaque` check, add a new heuristic:
     ```python
     if _is_simple_class(dfg):
         return _chunk_by_method(dfg)
     ```
   - `_is_simple_class(dfg)` returns `True` when:
     - All public methods have ≤ 30 lines.
     - No method calls another method on the same class (no internal dispatch).
     - Cross-window attributes ≤ 3 (simple shared state).
     - No inheritance beyond `object`.
   - `_chunk_by_method(dfg)`:
     - One `MacroAtomSpec` per public method (skip `__init__`, `__repr__`,
       `__str__`, etc.).
     - Inputs: method parameters + read attributes.
     - Outputs: return type + write attributes.
     - Concept type: infer from method name using the existing keyword table
       from `deterministic_decompose.py`.
     - Edges: sequential by method definition order + data-flow edges from
       shared state (reuse `_compute_state_edges`).

2. Tests:
   - Unit test: simple 3-method utility class → 3 atoms.
   - Unit test: class with internal dispatch → fallback to LLM.
   - Unit test: class with complex inheritance → fallback to LLM.

**Estimated effort**: Small-medium — the heuristic is simple, and
`_compute_state_edges` already handles the edge synthesis.

---

### Tool 10: `ingester_opaque_witness` — TemplateWitnessGenerator

**Prompt key**: `ingester_opaque_witness`
**File**: `ageom/ingester/template_witness_generator.py`
**Wiring**: `ageom/ingester/emitter.py` (where `INGESTER_OPAQUE_WITNESS` is
called)

**Rationale**: The emitter asks the LLM to draft ghost witness functions for
opaque DL modules. For common layer types (linear, conv, pooling, activation,
normalization), the shape transform is well-known and can be emitted from a
template.

**Implementation**:

**Steps**:

1. Create `ageom/ingester/template_witness_generator.py`:
   - `class TemplateWitnessGenerator` implementing `LLMClient`.
   - `_parse_opaque_prompt(user)` — extract the forward signature, module
     name, and input/output shapes from the prompt format.
   - Template library keyed by layer type pattern:

     | Pattern in name/signature | Shape transform |
     |--------------------------|-----------------|
     | `Linear`, `Dense`, `fc` | `(*, in_features)` → `(*, out_features)` |
     | `Conv1d` | `(N, C_in, L)` → `(N, C_out, L')` |
     | `Conv2d` | `(N, C_in, H, W)` → `(N, C_out, H', W')` |
     | `MaxPool`, `AvgPool` | Divide spatial dims by kernel_size |
     | `BatchNorm`, `LayerNorm` | Identity shape |
     | `ReLU`, `GELU`, `Sigmoid`, etc. | Identity shape |
     | `Dropout` | Identity shape |
     | `Flatten` | `(N, *dims)` → `(N, prod(dims))` |
     | `Embedding` | `(*, )` → `(*, embed_dim)` |

   - For each matched pattern, emit a Python function:
     ```python
     def witness_{name}(x: AbstractArray) -> AbstractArray:
         return AbstractArray(shape={output_shape}, dtype=x.dtype)
     ```
   - Confidence gate: return `None` if the layer type doesn't match any
     template, or if the input/output shapes can't be inferred from the
     signature.

2. Wire into `ageom/ingester/emitter.py` as an `LLMClient` wrapper for the
   `INGESTER_OPAQUE_WITNESS` call.

3. Tests:
   - Unit test per layer type template.
   - Unit test: unknown layer → fallback.

**Estimated effort**: Medium — the template library is bounded but requires
knowledge of common DL layer APIs.

---

## Wiring Pattern

All tools follow the same wiring pattern. In the relevant command module
(e.g., `ingest_cmds.py`, `synthesize_cmds.py`):

```python
# After building the LLM router
from ageom.ingester.deterministic_type_fixer import DeterministicTypeFixer
from ageom.llm_router import INGESTER_FIX_TYPE

# In the prompt-key override section:
if INGESTER_FIX_TYPE in prompt_keys:
    fallback = overrides.get(INGESTER_FIX_TYPE, default)
    overrides[INGESTER_FIX_TYPE] = DeterministicTypeFixer(fallback)
```

For tools in the existing `_create_llm_router` helper (used by run_cmds and
match_cmds), add to the existing deterministic-tool wiring block in
`ageom/commands/_helpers.py`.

---

## Testing Strategy

Each tool needs three test tiers:

1. **Unit tests** (`tests/test_{tool_name}.py`):
   - Parse the prompt format correctly.
   - Each pattern produces the expected output.
   - Unknown patterns return `None` (trigger fallback).
   - Edge cases: empty input, malformed prompt, unicode.

2. **Integration tests** (extend existing test files):
   - Wire the tool into the pipeline node and verify end-to-end behavior.
   - Mock the fallback LLM and verify it's called when expected.
   - Verify telemetry metadata records `"source": "deterministic"` vs
     `"source": "fallback"`.

3. **Benchmark validation** (extend `e2e_benchmark.sh`):
   - After all tools are implemented, re-run the benchmark and compare:
     - Total LLM calls per mode.
     - Latency per mode.
     - Verification success rate.
   - The benchmark should show a measurable reduction in LLM calls,
     especially in `single_agent` and `structured` modes.

---

## Dependency Order

```
Wave 1 (no dependencies between tools):
  Tool 1: ingester_fix_type     ← reuses classifier.py
  Tool 2: synthesizer_tactic    ← standalone
  Tool 3: ingester_hoist_state  ← standalone

Wave 2 (no dependencies on Wave 1):
  Tool 4: ingester_fix_ghost    ← standalone
  Tool 5: ingester_abstract     ← standalone
  Tool 6: ingester_fix_message_cycle ← standalone

Wave 3 (benefits from Wave 1-2 learnings):
  Tool 7: architect_decompose   ← extends existing + uses strategy_classifier
  Tool 8: architect_critique    ← extends existing structural_critic
  Tool 9: ingester_chunk        ← extends existing chunker
  Tool 10: ingester_opaque_witness ← standalone but informs Tool 4
```

Wave 1 tools can be implemented in parallel. Wave 2 tools can be implemented in
parallel. Wave 3 tools should be implemented sequentially to avoid conflicts in
shared files.

---

## Expected Impact

| Metric | Before | After Wave 1 | After All |
|--------|--------|-------------|-----------|
| Prompt keys with deterministic coverage | 11/17 (65%) | 14/17 (82%) | 17/17 (100%) |
| Estimated LLM calls eliminated (single_agent) | ~40% | ~55% | ~70% |
| Estimated LLM calls eliminated (verified) | ~30% | ~40% | ~55% |
| New files | 0 | 3 | 9 |
| Modified files | 0 | 3 | 6 |

These estimates assume typical workloads. Actual coverage depends on how often
each prompt key is invoked and what fraction of invocations match deterministic
patterns. The benchmark suite should be used to validate after each wave.
