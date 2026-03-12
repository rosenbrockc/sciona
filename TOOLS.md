# Deterministic Tools

AGEO-Matcher follows the **deterministic-first, LLM-fallback** principle: every
operation that can be handled by a regex, AST walk, embedding lookup, or type
check is handled deterministically. The LLM is only invoked when the
deterministic tool's confidence is too low or the pattern is unrecognized.

This document catalogs every deterministic tool in the system, maps them to the
LLM prompt keys they replace, and identifies gaps where new tools would be
useful.

---

## Prompt Key Coverage

The system has **17 LLM prompt keys**. The table below shows the deterministic
coverage status of each.

| Prompt Key | Phase | Deterministic Tool | Coverage |
|-----------|-------|-------------------|----------|
| `architect_strategy` | Architect | `StrategyClassifier` | **Covered** |
| `architect_decompose` | Architect | `DeterministicDecompose` (post-processor) | Partial |
| `architect_critique` | Architect | `structural_critique_issues` (pre-filter) | Partial |
| `hunter_score` | Hunter | `EmbeddingReranker` + `HeuristicCandidateRanker` | **Covered** |
| `hunter_reformulate` | Hunter | `HeuristicQueryReformulator` | **Covered** |
| `hunter_analyze_failure` | Hunter | `DeterministicFailureAnalyzer` | **Covered** |
| `synthesizer_repair` | Synthesizer | `DeterministicFix` + `classify_error` | Partial |
| `synthesizer_tactic` | Synthesizer | — | **None** |
| `orchestrator_refine` | Orchestrator | `_deterministic_split_subnodes` + `_split_on_connectors` | Partial |
| `ingester_chunk` | Ingester | opaque DL fallback (`propose_macro_atoms`) | Partial |
| `ingester_decompose` | Ingester | `decompose_function` (AST control-flow) | **Covered** |
| `ingester_hoist_state` | Ingester | — | **None** |
| `ingester_abstract` | Ingester | — | **None** |
| `ingester_fix_type` | Ingester | — | **None** |
| `ingester_fix_ghost` | Ingester | — | **None** |
| `ingester_opaque_witness` | Ingester | — | **None** |
| `ingester_fix_message_cycle` | Ingester | — | **None** |

**Summary**: 6 fully covered, 5 partially covered, 6 with no deterministic tool.

---

## Existing Tools

### Hunter Phase

#### EmbeddingReranker

- **File**: `ageom/hunter/embedding_reranker.py`
- **Replaces**: `hunter_score`
- **What it does**: Ranks candidate functions by cosine similarity between
  embedded query and candidate text, with a type-signature token overlap bonus.
- **Fallback trigger**: Top-2 margin < 0.05 (configurable).
- **Patterns handled**: Any query/candidate pair where embedding distance is
  discriminating.

#### HeuristicCandidateRanker

- **File**: `ageom/hunter/candidate_ranker.py`
- **Replaces**: `hunter_score` (fallback tier between EmbeddingReranker and LLM)
- **What it does**: Token-overlap scoring with domain-specific bonus rules
  (~15 domain patterns: filtering, shortest path, matrix factorization, etc.).
- **Fallback trigger**: Min score < 1.0 or margin < 0.2.
- **Patterns handled**: Name and type-signature token overlap; hardcoded domain
  bonuses.

#### DeterministicFailureAnalyzer

- **File**: `ageom/hunter/failure_analyzer.py`
- **Replaces**: `hunter_analyze_failure`
- **What it does**: Regex pattern matching on compiler output to extract
  CAUSE/TARGET/NEXT analysis for known error classes.
- **Fallback trigger**: No regex pattern matches.
- **Patterns handled**: ~20 patterns covering unknown identifier, arity mismatch,
  type mismatch, syntax errors, unsolved goals, universe mismatch (Lean 4, Coq,
  Python/mypy).

#### HeuristicQueryReformulator

- **File**: `ageom/hunter/query_reformulator.py`
- **Replaces**: `hunter_reformulate`
- **What it does**: Generates alternative search queries using phrase-rule
  matching, keyword variants, and catalog-hint variants.
- **Fallback trigger**: < min_queries (default 3) generated.
- **Patterns handled**: ~12 domain-specific phrase rules; tokenized domain
  anchors; namespace/declaration hints from index.

### Architect Phase

#### StrategyClassifier

- **File**: `ageom/architect/strategy_classifier.py`
- **Replaces**: `architect_strategy`
- **What it does**: Phrase-rule matching (~79 rules) over goal text to rank
  ConceptType categories (DP, graph optimization, signal filter, etc.).
- **Fallback trigger**: Confidence < 0.55 or margin < 0.15.
- **Patterns handled**: Weighted phrase rules mapping to paradigm strategies;
  skeleton-based lexical overlap.

#### DeterministicDecompose (post-processor)

- **File**: `ageom/architect/deterministic_decompose.py`
- **Replaces**: Post-processes `architect_decompose` output
- **What it does**: Synthesizes operational details from LLM-proposed nodes:
  infers concept type from keywords, binds primitives via token overlap,
  auto-generates IO specs, constructs type-propagation edges.
- **Fallback trigger**: Uses hardcoded templates when LLM output is sparse.
- **Patterns handled**: ~20 keyword-to-ConceptType rules; domain-specific
  fallback templates (signal filter, signal transform); generic 3-step fallback.

#### structural_critique_issues (pre-filter)

- **File**: `ageom/architect/structural_critic.py`
- **Replaces**: Pre-filters for `architect_critique`
- **What it does**: Validates decompositions deterministically: minimum 2
  sub-nodes, depth constraints, no self-loops, edge I/O coverage, duplicate
  detection (Jaccard > 0.85), type coverage, reachability.
- **Fallback trigger**: Issues found trigger the LLM critique for semantic
  evaluation.
- **Patterns handled**: Structural graph validation; type-theoretic I/O checks;
  primitive catalog lookup.

### Ingester Phase

#### decompose_function (Control-Flow Decomposer)

- **File**: `ageom/ingester/control_flow_decomposer.py`
- **Replaces**: `ingester_decompose`
- **What it does**: Python AST-based function splitting at structural boundaries
  (loops, conditionals, significant function calls); extracts variable
  reads/writes; builds data-flow edges.
- **Fallback trigger**: Returns `None` if function is too simple or result doesn't
  meet min_sub_atoms (default 2).
- **Patterns handled**: For/while loops, if-branches, call assignments; min/max
  sub-atom constraints (2-6); tiny block merging.

#### propose_macro_atoms — opaque DL fallback

- **File**: `ageom/ingester/chunker.py`
- **Replaces**: `ingester_chunk` (for opaque dataflow-opaque classes only)
- **What it does**: Deterministically emits a single macro-atom wrapping the
  entire class for neural-network/DL layers.
- **Fallback trigger**: Only fires when `dfg.is_opaque == True`.
- **Patterns handled**: TensorFlow/JAX/PyTorch layer classes.

#### _compute_state_edges (post-processor)

- **File**: `ageom/ingester/chunker.py`
- **What it does**: Builds state-typed edges from method read/write sets without
  LLM involvement.
- **Patterns handled**: Parses DataFlow metadata; maps methods to macro-atoms;
  builds transitive reader/writer relationships.

#### flatten_config (post-processor)

- **File**: `ageom/ingester/chunker.py`
- **What it does**: Flattens config-gated method branches into optional variant
  atoms; marks atoms with `is_optional` flag.
- **Patterns handled**: Config branches from method metadata.

### Synthesizer Phase

#### classify_error + suggest_deterministic_fix

- **File**: `ageom/synthesizer/classifier.py`
- **Replaces**: Pre-filters for `synthesizer_repair`
- **What it does**: Regex-based error categorization (~19 patterns) into
  MISSING_IMPORT, TYPE_MISMATCH, UNSOLVED_GOAL, UNIVERSE_MISMATCH, SYNTAX,
  UNKNOWN; suggests fixes for known patterns.
- **Fallback trigger**: Returns `None` if no deterministic fix pattern matches.
- **Fix database**:
  - **Imports**: Lean open/import, Python module imports, common name imports
    (numpy, pandas, scipy, typing)
  - **Type coercions**: ~12 Python coercion rules (int↔float, str→int,
    list↔ndarray, etc.); ~6 Lean coercions (Nat↔Int, Fin→Nat, List↔Array)
  - **Syntax**: Missing delimiters, indentation, annotation errors

#### DeterministicFix (repair graph node)

- **File**: `ageom/synthesizer/repair.py`
- **Replaces**: Batches fixes before `synthesizer_repair` LLM calls
- **What it does**: Applies all deterministic fixes in one pass, inserts combined
  fix text at top of file.
- **Fallback trigger**: No fixes available → routes to LLMRepair.

### Orchestrator Phase

#### _deterministic_split_subnodes

- **File**: `ageom/orchestrator.py`
- **Replaces**: `orchestrator_refine` (for known domain patterns)
- **What it does**: Pattern-matched sub-node splitting for known algorithmic
  domains.
- **Fallback trigger**: No domain pattern matches.
- **Patterns handled** (~20 domains):
  - ECG bandpass filter (Design/Apply)
  - Shortest path (Initialize/Relax)
  - SPD matrix (Cholesky/Solve)
  - LCS / edit distance (DP Table/Backtrack)
  - Matrix factorization (Factorize/Reconstruct)
  - Eigenvalue (Compute/Transform)
  - Optimization (Initialize/Iterate/Extract)
  - FFT / spectral (Transform/Analyze)
  - Sorting (Partition/Merge/Sort)
  - String matching (Build Index/Search)
  - Interpolation (Fit/Evaluate)
  - Clustering (Initialize/Assign/Update)
  - Statistical inference (Prior/Likelihood/Posterior)
  - Tree/graph traversal (Initialize/Visit/Collect)
  - Convolution / correlation (Prepare/Convolve)
  - Normalization (Compute Stats/Scale)

#### _split_on_connectors

- **File**: `ageom/orchestrator.py`
- **Replaces**: `orchestrator_refine` (second-tier fallback)
- **What it does**: Splits goal description on linguistic connectors ("and then",
  "with", "before") to create 2-3 sub-nodes.
- **Fallback trigger**: Split count invalid or insufficient content tokens.

### Verification Oracles

These are deterministic by nature — they use compilers/interpreters as ground
truth, never LLMs.

| Oracle | File | Method |
|--------|------|--------|
| `LeanEnvironment` | `ageom/judge/lean_env.py` | `example : {type} := @{term}` via lean-interact |
| `CoqEnvironment` | `ageom/judge/coq_env.py` | `Definition _check : {type} := {term}.` via coqpyt |
| `PythonEnvironment` | `ageom/judge/python_env.py` | Import-based: importable + callable + arity check |

### Catalog & Search Oracles

| Oracle | File | Method |
|--------|------|--------|
| `PrimitiveCatalog` | `ageom/architect/catalog.py` | Exact name/alias lookup + token overlap confidence |
| `SkillIndex` | `ageom/architect/embedder.py` | FAISS cosine similarity over skill primitives |
| `SemanticIndexImpl` | `ageom/indexer/builder.py` | FAISS cosine similarity over library declarations |

### Planner Policy

| Tool | File | Method |
|------|------|--------|
| `_select_policy` | `ageom/services/planner_service.py` | Compound goal marker detection + token count |
| `_is_compound_goal` | `ageom/services/planner_service.py` | Phrase marker matching (10 markers) |

---

## Missing Tools

The following prompt keys have **no deterministic coverage** today. Each entry
describes the LLM call, why a deterministic tool is feasible, and what it would
do.

### High Value

#### 1. `synthesizer_tactic` — Deterministic Tactic Suggester

**Current LLM use**: Generates proof tactics or Python implementations to replace
`sorry` / `Admitted` / `NotImplementedError` stubs.

**Why feasible**: Many sorry-elimination cases follow mechanical patterns:
- `sorry` on a `rfl`-solvable goal → emit `rfl`
- `sorry` on a `simp`/`omega`/`norm_num`-solvable goal → try standard closers
- `sorry` on a `decide`-solvable goal → emit `decide`
- `NotImplementedError` on a function with a known match → emit delegation call
- `Admitted` on a `Prop` that appears in Mathlib → emit `exact <lemma>`

**Proposed tool**: `DeterministicTacticSuggester` — pattern-match on the proof
goal and try a ranked list of standard tactics (`rfl`, `simp`, `omega`,
`norm_num`, `decide`, `exact?`, `apply?`). For Python, emit delegation to the
matched library function. Fall back to LLM for complex proof obligations.

**Estimated coverage**: 30-50% of sorry/Admitted stubs; higher for Python
`NotImplementedError`.

#### 2. `ingester_hoist_state` — Deterministic State Hoister

**Current LLM use**: Identifies cross-method shared state (instance variables,
class attributes) and hoists them into `StateModelSpec`s with typed fields.

**Why feasible**: Python AST can extract:
- All `self.x` assignments across methods → field names
- Type annotations from `__init__` signature and body
- Read/write sets per method (already computed by `_reads_writes`)
- Default values from `__init__` assignments

**Proposed tool**: `ASTStateHoister` — walk the class AST, collect all
`self.attr` writes with their inferred types, build a `StateModelSpec` with
typed fields and default values. Fall back to LLM only when types can't be
inferred (dynamic attribute creation, `**kwargs` unpacking).

**Estimated coverage**: 60-80% of Python classes with typed `__init__`.

#### 3. `ingester_fix_type` — Deterministic Type Fixer

**Current LLM use**: Fixes mypy type errors in generated atom wrappers.

**Why feasible**: The error patterns are the same as `synthesizer_repair` —
missing imports, type mismatches, incompatible signatures. The existing
`classifier.py` fix database already handles these patterns but isn't wired
into the ingester.

**Proposed tool**: Reuse `classify_error` + `suggest_deterministic_fix` from
`classifier.py` directly. Parse mypy output, classify each error, apply
deterministic fixes. Fall back to LLM for errors not in the fix database.

**Estimated coverage**: 40-60% of mypy errors (same as synthesizer_repair
coverage).

### Medium Value

#### 4. `ingester_fix_ghost` — Deterministic Ghost Fix

**Current LLM use**: Fixes ghost witness function errors (simulation failures,
abstract type mismatches).

**Why feasible**: Ghost simulation errors are highly structured:
- Shape mismatch → insert reshape/transpose call
- Domain mismatch (time vs frequency) → insert domain conversion
- Missing witness → emit pass-through witness (identity function)
- Type mismatch in abstract values → adjust AbstractSignal metadata

**Proposed tool**: `DeterministicGhostFixer` — pattern-match on ghost simulation
error types (`ShapeMismatch`, `DomainMismatch`, `MissingWitness`). Apply
known fixes from a lookup table keyed by (atom_name, error_type). Fall back
to LLM for novel error combinations.

**Estimated coverage**: 40-60% of ghost failures (most are shape/domain
mismatches).

#### 5. `architect_decompose` — Deeper Deterministic Decomposition

**Current state**: `DeterministicDecompose` post-processes LLM output but doesn't
replace the LLM call. The LLM is still invoked for every decomposition.

**Why feasible**: For goals that match known algorithmic templates (sorting,
searching, optimization, signal processing), the decomposition structure is
well-known. The `StrategyClassifier` already identifies the paradigm — the
next step is to emit the full decomposition directly when confidence is high.

**Proposed tool**: Extend `DeterministicDecompose` to be a full LLM replacement
(not just a post-processor) when the `StrategyClassifier` confidence is above
a threshold (e.g., 0.8). Use skeleton templates to generate the complete CDG
sub-tree with typed IO specs. Fall back to LLM when confidence is below
threshold or the skeleton doesn't cover the goal structure.

**Estimated coverage**: 20-30% of decompositions (those with clear paradigm
matches).

#### 6. `architect_critique` — Deeper Deterministic Critique

**Current state**: `structural_critique_issues` catches structural problems but
the LLM is always called for semantic evaluation.

**Why feasible**: Many semantic critique checks can be formalized:
- Sub-nodes collectively cover parent's output names → completeness check
- Sub-node descriptions contain parent's key terms → relevance check
- Type signatures are compatible along edges → type consistency
- No sub-node is a near-duplicate of its parent → non-trivial decomposition

**Proposed tool**: Extend `structural_critique_issues` with semantic heuristics:
term-overlap completeness score, type-propagation validation, and parent-child
similarity check. If all deterministic checks pass with high confidence,
approve without LLM. Fall back to LLM when any check is ambiguous.

**Estimated coverage**: 30-40% of critique passes (those where structural
validation is unambiguous).

#### 7. `ingester_abstract` — Deterministic Abstraction

**Current LLM use**: Generates abstract descriptions of code chunks for embedding
and matching.

**Why feasible**: Docstrings, function signatures, and variable names already
contain most of the semantic information. A template-based approach can produce
adequate abstractions:
- Function name → title (snake_to_title)
- Parameter names + types → input description
- Return type → output description
- Docstring first line → summary

**Proposed tool**: `TemplateAbstractor` — extract function signature, docstring,
and key variable names. Format into a structured abstract description. Fall
back to LLM when docstring is missing and variable names are uninformative.

**Estimated coverage**: 50-70% of well-documented code.

### Lower Value

#### 8. `ingester_chunk` — Deterministic Chunking (beyond opaque DL)

**Current state**: Only fires for opaque DL classes. Regular classes go to LLM.

**Why feasible**: For classes with clear method boundaries (each method is
self-contained, no deep cross-method state), chunking is mechanical:
one macro-atom per public method, with edges from shared state.

**Proposed tool**: Extend the opaque fallback to a general heuristic: if each
method has ≤ N lines and method-to-method data flow is simple (only through
`self.x` attributes), emit one atom per method. Fall back to LLM for complex
cross-method control flow.

**Estimated coverage**: 20-30% of classes (utility classes, data processors).

#### 9. `ingester_opaque_witness` — Deterministic Witness Generation

**Current LLM use**: Generates ghost witness functions for opaque atoms (those
without existing witnesses in the registry).

**Why feasible**: For pass-through atoms and simple transformations, witnesses
are mechanical:
- Identity: `lambda x: x` (shape/type unchanged)
- Reshape: `lambda x: AbstractSignal(shape=new_shape, ...)`
- Type cast: propagate input metadata with new dtype

**Proposed tool**: `TemplateWitnessGenerator` — look up atom category, emit
template witness. Fall back to LLM for atoms with complex domain semantics.

**Estimated coverage**: 20-30% of opaque atoms.

#### 10. `ingester_fix_message_cycle` — Deterministic Cycle Breaker

**Current LLM use**: Fixes message-passing cycles in ingested dataflow graphs.

**Why feasible**: Cycle-breaking is a graph algorithm problem:
- Find strongly connected components (Tarjan's)
- Break at the weakest edge (lowest data-flow weight)
- Insert explicit state-carry variable at the break point

**Proposed tool**: `DeterministicCycleBreaker` — detect SCCs, rank edges by
data-flow weight, break at minimum-weight edge, insert state variable. Fall
back to LLM when the cycle involves complex control flow or the break point
is ambiguous.

**Estimated coverage**: 40-50% of cycle errors (simple feedback loops).

---

## Implementation Priority

Based on the positioning as **verified retrieval-augmented composition** and the
deterministic-first principle, the recommended implementation order is:

1. **`synthesizer_tactic`** — directly reduces LLM calls in the repair loop,
   which is the most latency-sensitive phase.
2. **`ingester_hoist_state`** — pure AST analysis, high coverage, removes a
   reliable LLM call.
3. **`ingester_fix_type`** — reuses existing `classifier.py` infrastructure,
   minimal new code.
4. **`ingester_fix_ghost`** — structured error patterns, bounded fix space.
5. **`architect_decompose`** (full replacement) — high leverage but harder;
   depends on skeleton template coverage.
6. **`architect_critique`** (deeper heuristics) — incremental improvement over
   existing structural critic.
7. **`ingester_abstract`** — template-based, straightforward.
8. **`ingester_chunk`** — incremental extension of existing opaque fallback.
9. **`ingester_opaque_witness`** — narrow scope, low call volume.
10. **`ingester_fix_message_cycle`** — rare error case, graph algorithm.

---

## Design Pattern

All deterministic tools follow the same pattern:

```python
class DeterministicTool:
    """Implements LLMClient protocol for prompt key X."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "tool_name_v1"

    def __init__(self, fallback: LLMClient) -> None:
        self._fallback = fallback

    async def complete(self, system: str, user: str) -> str:
        result = self._try_deterministic(user)
        if result is not None:
            self._last_completion_metadata = {"source": "deterministic"}
            return result
        return await self._fallback.complete(system, user)
```

This ensures:
- Transparent fallback — the LLM is always available as a safety net.
- Telemetry — deterministic vs LLM resolution is tracked per call.
- Composability — tools can be stacked (e.g., EmbeddingReranker wraps
  HeuristicCandidateRanker wraps LLM).
- Zero-risk deployment — if the deterministic tool returns `None`, behavior is
  identical to the LLM-only path.
