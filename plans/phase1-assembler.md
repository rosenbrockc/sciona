# Phase 1: The Assembler

## Context

Rounds 1 and 2 produce a CDG (graph of algorithmic nodes) and a set of `MatchResult`s (each atomic leaf mapped to a verified library declaration). Phase 1 takes these two outputs and **stitches them into a single compilable Lean 4 / Coq source file** — the "skeleton file." The skeleton uses `sorry` for complex proofs so the compiler can validate structural correctness (types fit together) before Phase 2 tackles logical correctness.

**Existing infrastructure used**: `CDGExport` + `DependencyEdge` from `ageom/architect/handoff.py`, `MatchResult` from `ageom/types.py`, `ProofEnvironment.check_term`/`check_proof` from `ageom/judge/`, `LLMClient` from `ageom/hunter/llm.py`.

---

## New Files (6 source + 1 test)

### 1. `ageom/synthesizer/__init__.py`

Empty package init.

### 2. `ageom/synthesizer/models.py`

Pydantic + dataclass models for the synthesizer pipeline.

**`AssemblyUnit`** (Pydantic `BaseModel`):
- `node_id: str` — back-reference to CDG node
- `name: str` — human-readable (from `AlgorithmicNode.name`)
- `declaration_name: str` — fully-qualified library name (from `MatchResult.verified_match.candidate.declaration.name`)
- `type_signature: str` — formal type (from declaration)
- `raw_code: str` — declaration source if available
- `inputs: list[IOSpec]` — from CDG node
- `outputs: list[IOSpec]` — from CDG node
- `requires_glue: bool` — True if any inbound `DependencyEdge` has `requires_glue=True`

**`GlueEdge`** (Pydantic `BaseModel`):
- `source_id: str`
- `target_id: str`
- `output_name: str`
- `input_name: str`
- `source_type: str`
- `target_type: str`
- `cast_expr: str = ""` — filled in during assembly or repair (Phase 2)

**`SkeletonFile`** (Pydantic `BaseModel`):
- `prover: str` — `"lean4"` or `"coq"`
- `source_code: str` — the generated source
- `units: list[AssemblyUnit]` — ordered (topological)
- `glue_edges: list[GlueEdge]`
- `sorry_count: int` — number of `sorry` placeholders
- `metadata: dict` — goal, paradigm, timestamp

**`AssemblyResult`** (Pydantic `BaseModel`):
- `skeleton: SkeletonFile`
- `feedback: CompilerFeedback` — from initial compilation check
- `compiled_ok: bool` — True if skeleton compiles (possibly with sorry)

### 3. `ageom/synthesizer/toposort.py`

Utility: topological sort of CDG nodes by data-flow edges.

**`toposort_nodes(nodes, edges) -> list[str]`**:
- Input: `list[AlgorithmicNode]`, `list[DependencyEdge]`
- Output: node IDs in dependency order (leaves first, root last)
- Uses Kahn's algorithm
- Raises `ValueError` on cycles

### 4. `ageom/synthesizer/assembler.py`

The core assembly logic. No Jinja2 — uses f-string builders to keep dependencies at zero.

**`class Assembler`**:

Constructor:
- `__init__(self, prover: Prover)` — selects Lean 4 or Coq code generation

**`assemble(cdg, match_results) -> SkeletonFile`**:
1. Validate: every atomic leaf in `cdg` must have a matching `MatchResult` with `success=True`. Raise `AssemblyError` if not.
2. Build `AssemblyUnit` for each atomic leaf by joining CDG node data with its `MatchResult`.
3. Build `GlueEdge` for each `DependencyEdge`. If `requires_glue`, leave `cast_expr` empty (Phase 2 fills it). If types match, `cast_expr = ""` (no cast needed).
4. Topologically sort nodes via `toposort_nodes()`.
5. Generate source code by calling `_emit_lean4()` or `_emit_coq()`.

**`_emit_lean4(units, glue_edges, metadata) -> str`**:
- Emit `import Mathlib` header
- For each unit (topological order):
  - Emit `-- Node: {name} ({node_id})`
  - Emit `#check @{declaration_name}` (import validation)
  - If the unit has inbound glue edges with non-empty `cast_expr`, emit the cast as a `def` wrapper
  - Emit a `noncomputable def {sanitized_name} : {type_signature} := @{declaration_name}` (or `sorry` if type needs adaptation)
- For the root composition:
  - Emit a `theorem {goal_name} : {root_type} := by sorry` stub that wires children together
- Return the full source string

**`_emit_coq(units, glue_edges, metadata) -> str`**:
- Same structure with Coq syntax:
  - `Require Import` header
  - `Definition` / `Lemma` stubs
  - `Admitted.` instead of `sorry`

**`sanitize_name(name) -> str`**: convert node names to valid Lean/Coq identifiers (lowercase, underscores, dedup).

### 5. `ageom/synthesizer/compiler.py`

Wraps `ProofEnvironment` for whole-file compilation.

**`class SkeletonCompiler`**:

Constructor:
- `__init__(self, env: ProofEnvironment)`

**`async compile(skeleton: SkeletonFile) -> AssemblyResult`**:
1. Send `skeleton.source_code` to `env.check_proof()` (or a new `_run()` call for whole-file checking)
2. Parse `CompilerFeedback`
3. Return `AssemblyResult(skeleton=skeleton, feedback=feedback, compiled_ok=feedback.success)`

**`async check_unit(unit: AssemblyUnit) -> CompilerFeedback`**:
- Compile just the single unit's definition in isolation (faster feedback loop for debugging)

### 6. `ageom/synthesizer/pipeline.py`

End-to-end orchestration for Phase 1.

**`async assemble_and_check(cdg, match_results, env) -> AssemblyResult`**:
1. Create `Assembler(prover=env.prover_name)`
2. Call `assembler.assemble(cdg, match_results)` → `SkeletonFile`
3. Create `SkeletonCompiler(env)`
4. Call `compiler.compile(skeleton)` → `AssemblyResult`
5. Log sorry count and compilation status
6. Return result

### 7. `tests/test_assembler.py`

**Fixtures**:
- `sample_cdg()` — 3-node CDG (root + sort_step + search_step) with one data-flow edge, reuse pattern from `test_handoff.py`
- `sample_match_results()` — two `MatchResult`s with `success=True`, declarations pointing to `List.mergeSort` and `List.binSearch`
- `mock_env()` — mock `ProofEnvironment` that returns `(True, "ok")` for any `check_proof`

**Test classes**:

`TestToposort`:
- `test_linear_chain` — A→B→C produces [A, B, C]
- `test_diamond` — A→B, A→C, B→D, C→D produces valid order
- `test_single_node` — one node, no edges
- `test_cycle_raises` — cyclic edges raise `ValueError`

`TestAssembler`:
- `test_assemble_lean4_skeleton` — produces valid `SkeletonFile` with Lean 4 source containing `import Mathlib`, `#check`, node comments
- `test_assemble_coq_skeleton` — produces valid Coq source with `Require Import`
- `test_missing_match_raises` — atomic leaf without match raises `AssemblyError`
- `test_sorry_count` — sorry count matches number of composition stubs
- `test_glue_edges_flagged` — edges with `requires_glue=True` produce `GlueEdge` entries
- `test_sanitize_name` — spaces, hyphens, unicode → valid identifiers

`TestSkeletonCompiler`:
- `test_compile_success` — mock env returns success → `compiled_ok=True`
- `test_compile_failure` — mock env returns errors → `compiled_ok=False`, errors in feedback
- `test_check_unit_isolation` — single unit compiles independently

`TestPipeline`:
- `test_assemble_and_check_happy_path` — full pipeline returns `AssemblyResult` with `compiled_ok=True`
- `test_assemble_and_check_compile_failure` — pipeline returns result with errors, `compiled_ok=False`

---

## Modified Files (3)

### 8. `ageom/cli.py` — Add `assemble` subcommand

**New subparser** `assemble` (after `visualize`):
- `cdg_file` positional arg — path to CDG JSON
- `matches_file` positional arg — path to JSON with match results (list of serialized `MatchResult`s)
- `--prover` (choices: `lean4`, `coq`, default: `lean4`)
- `--output` (str) — output path for generated source file
- `--check` flag — also compile and report errors (requires proof environment)
- `--llm-provider`, `--llm-model`, `--llm-max-tokens` — for future glue generation

**New handler** `async _cmd_assemble(args)`:
1. Load CDG via `load_json(args.cdg_file)`
2. Load match results from `args.matches_file` (JSON array of dicts, deserialized into `MatchResult`s)
3. Call `Assembler(prover).assemble(cdg, match_results)` → `SkeletonFile`
4. Write `skeleton.source_code` to `args.output` (default: `{cdg_stem}_skeleton.lean` or `.v`)
5. If `--check`: set up `ProofEnvironment`, compile, print feedback
6. Print summary: unit count, sorry count, compile status

### 9. `ageom/types.py` — Add `MatchResult` serialization helpers

**New methods on `MatchResult`**:
- `to_dict() -> dict` — serialize to JSON-friendly dict (Declaration, CandidateMatch, VerificationResult are frozen dataclasses, so convert recursively)
- `@staticmethod from_dict(data: dict) -> MatchResult` — deserialize

These are needed so CLI can save/load Round 2 results as JSON for the `assemble` command.

### 10. `pyproject.toml` — Add `synthesizer` optional dependency group

```toml
synthesizer = [
    "ageo-matcher[hunter]",  # needs LLMClient, pydantic-graph
]
```

Update `all` to include `synthesizer`.

---

## Key Design Decisions

1. **No Jinja2** — source generation uses f-string builders. Keeps dependencies at zero for this phase, and the templates are straightforward enough that a template engine adds complexity without benefit.

2. **`sorry` / `Admitted` strategy** — the skeleton validates structural correctness (do the library functions' types compose correctly through the graph?) without requiring logical proofs. Phase 2's repair agent fills in the `sorry` gaps.

3. **Topological sort determines emission order** — leaves (atomic operations) are emitted first so they're in scope when parent composition stubs reference them. This mirrors how a human would write a Lean/Coq file.

4. **`AssemblyUnit` bridges CDG + MatchResult** — instead of passing both data structures through the pipeline, the assembler fuses them into a single representation. This simplifies downstream code (compiler, repair agent).

5. **`GlueEdge.cast_expr`** — left empty in Phase 1. Phase 2's repair agent fills these in when the compiler reports type mismatches at edge boundaries. This is the main "surgical fix" point.

6. **Isolated unit checking** — `SkeletonCompiler.check_unit()` enables faster iteration during repair (Phase 2) by compiling one definition at a time instead of the whole file.

---

## Verification

1. `pytest tests/test_assembler.py -v` — all unit tests pass
2. Manual: generate CDG + match results → `ageom assemble cdg.json matches.json --output skeleton.lean` → inspect generated Lean file
3. Manual: `ageom assemble cdg.json matches.json --check --prover lean4` → compiler feedback printed
4. `pytest tests/ -v` — full suite still green
