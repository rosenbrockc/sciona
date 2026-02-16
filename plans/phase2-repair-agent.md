# Phase 2: The Repair Agent

## Context

Phase 1 produces a "skeleton file" — a Lean 4 or Coq source that wires Round 2's matched library declarations together following the CDG's data-flow graph, with `sorry`/`Admitted` placeholders for composition proofs. Phase 2 sits in a **compile → analyze → patch → recompile** loop to eliminate those placeholders and fix type mismatches. The LLM generates glue code (casts, wrappers, tactic invocations); the compiler is the ground truth.

**Existing infrastructure used**: `ProofEnvironment.check_proof` / `_run` from `ageom/judge/`, `CompilerFeedback` from `ageom/judge/models.py`, `LLMClient` from `ageom/hunter/llm.py`, pydantic-graph from `ageom/hunter/nodes.py` pattern.

---

## New Files (5 source + 1 test)

### 1. `ageom/synthesizer/classifier.py`

Regex-based pre-filter that categorizes compiler errors before prompting the LLM. This avoids wasting LLM tokens on errors that have deterministic fixes.

**`class ErrorCategory(str, Enum)`**:
- `MISSING_IMPORT` — "unknown identifier", "unknown namespace"
- `TYPE_MISMATCH` — "type mismatch", "expected ... got ..."
- `UNSOLVED_GOAL` — "unsolved goals", goal markers (`⊢`)
- `UNIVERSE_MISMATCH` — "universe level" errors
- `SYNTAX` — parse errors
- `UNKNOWN` — anything else

**`classify_error(error_line: str) -> ErrorCategory`**:
- Sequence of regex checks against the error line
- Returns the first matching category, or `UNKNOWN`

**`classify_feedback(feedback: CompilerFeedback) -> list[tuple[ErrorCategory, str]]`**:
- Classifies each error in `feedback.errors` and each goal in `feedback.goals_remaining`
- Returns list of `(category, original_error_text)` pairs

**`suggest_deterministic_fix(category: ErrorCategory, error_text: str) -> str | None`**:
- `MISSING_IMPORT` → returns `"import Mathlib.{inferred_module}"` or `"open {namespace}"` extracted from the error text
- `SYNTAX` → returns `None` (syntax errors need LLM)
- All others → returns `None` (needs LLM analysis)

### 2. `ageom/synthesizer/prompts.py`

Prompt templates for the repair agent's LLM calls.

**`ANALYZE_ERROR_SYSTEM`**: System prompt instructing the LLM to act as a Lean 4 / Coq repair specialist. Explains: you receive a source file, a compiler error, and the error category. Generate a minimal patch — do NOT rewrite the core logic, only generate glue code (type casts, coercions, `simp`/`omega`/`ring` tactics, import additions).

**`ANALYZE_ERROR_USER`**: Template with `{source_code}`, `{error_text}`, `{error_category}`, `{error_context}` (surrounding lines).

**`GENERATE_TACTIC_SYSTEM`**: System prompt for generating tactic-mode proof bodies to replace `sorry`. Explains: you receive a goal type and available hypotheses. Return a `by` proof body using standard Mathlib tactics.

**`GENERATE_TACTIC_USER`**: Template with `{goal_type}`, `{hypotheses}`, `{available_lemmas}` (the matched library declarations in scope).

**`GENERATE_GLUE_SYSTEM`**: System prompt for generating type coercion expressions. Explains: source has type X, target expects type Y. Generate the minimal cast expression.

**`GENERATE_GLUE_USER`**: Template with `{source_type}`, `{target_type}`, `{source_expr}`, `{edge_context}`.

### 3. `ageom/synthesizer/patcher.py`

Applies patches to skeleton source code.

**`@dataclass class Patch`**:
- `line_start: int` — 1-indexed line to start replacing
- `line_end: int` — 1-indexed line to stop replacing (inclusive)
- `replacement: str` — new text for those lines
- `description: str` — human-readable note

**`apply_patches(source: str, patches: list[Patch]) -> str`**:
- Sorts patches by `line_start` descending (apply bottom-up to preserve line numbers)
- Validates no overlapping patches
- Returns the modified source string

**`find_sorry_locations(source: str, prover: str) -> list[tuple[int, str]]`**:
- Scans for `sorry` (Lean) or `Admitted.` (Coq)
- Returns list of `(line_number, surrounding_context)` for each placeholder

**`extract_error_context(source: str, error_line: int, radius: int = 3) -> str`**:
- Returns lines `[error_line - radius, error_line + radius]` from source
- Used to give the LLM focused context around an error

### 4. `ageom/synthesizer/repair.py`

The repair agent: a pydantic-graph state machine mirroring the Hunter's pattern.

**`@dataclass class RepairState`**:
- `skeleton: SkeletonFile` — current source (mutated across iterations)
- `max_iterations: int = 10`
- `iteration: int = 0`
- `patches_applied: list[Patch] = []`
- `error_history: list[tuple[int, ErrorCategory, str]] = []` — (iteration, category, error_text)
- `sorry_remaining: int` — count of sorry/Admitted left
- `compiled_ok: bool = False`

**`@dataclass class RepairDeps`**:
- `env: ProofEnvironment`
- `llm: LLMClient`

**Graph nodes** (pydantic-graph `BaseNode[RepairState, RepairDeps, SkeletonFile]`):

**`CompileCheck`** (entry point):
- Sends `state.skeleton.source_code` to `env._run()` (whole-file compile)
- If `feedback.success` → `End[SkeletonFile]` (done!)
- If errors → classify via `classify_feedback()` → `DeterministicFix` if any have deterministic fixes, else `LLMRepair`
- If only `sorry`/`Admitted` goals remain (no hard errors) → `SorryElimination`
- If `iteration >= max_iterations` → `End[SkeletonFile]` (budget exhausted, return best effort)

**`DeterministicFix`**:
- Applies deterministic fixes from `suggest_deterministic_fix()` (e.g., adding imports)
- Applies patches via `apply_patches()`
- Updates `state.skeleton.source_code`
- → `CompileCheck`

**`LLMRepair`**:
- Takes the first unresolved error (by category priority: `TYPE_MISMATCH` > `UNIVERSE_MISMATCH` > `UNKNOWN`)
- Extracts error context via `extract_error_context()`
- Calls `llm.complete(ANALYZE_ERROR_SYSTEM, ANALYZE_ERROR_USER.format(...))`
- Parses LLM response as a `Patch` (expects JSON with `line_start`, `line_end`, `replacement`)
- Applies patch, updates source
- Records in `error_history`
- Increments `iteration`
- → `CompileCheck`

**`SorryElimination`**:
- Finds next `sorry` location via `find_sorry_locations()`
- Extracts the goal type from the surrounding definition/theorem
- Calls `llm.complete(GENERATE_TACTIC_SYSTEM, GENERATE_TACTIC_USER.format(...))`
- Replaces the `sorry` with the generated tactic body
- Applies patch, updates source
- Decrements `sorry_remaining`
- → `CompileCheck`

**`repair_graph: Graph[RepairState, RepairDeps, SkeletonFile]`**:
```
CompileCheck → DeterministicFix → CompileCheck
            → LLMRepair → CompileCheck
            → SorryElimination → CompileCheck
            → End[SkeletonFile]
```

### 5. `ageom/synthesizer/agent.py`

High-level wrapper (mirrors `HunterAgent` / `DecompositionAgent` pattern).

**`class SynthesizerAgent`**:

Constructor:
- `__init__(self, env: ProofEnvironment, llm: LLMClient, max_iterations: int = 10)`
- Stores `RepairDeps(env=env, llm=llm)`

**`async synthesize(skeleton: SkeletonFile) -> SynthesisResult`**:
1. Create `RepairState` from skeleton
2. Run `repair_graph` starting at `CompileCheck`
3. Return `SynthesisResult` wrapping the final `SkeletonFile` + stats

**`@dataclass class SynthesisResult`** (in `models.py`):
- `skeleton: SkeletonFile` — final source (may still have sorrys if budget exhausted)
- `compiled_ok: bool`
- `sorry_remaining: int`
- `patches_applied: list[Patch]`
- `iterations_used: int`
- `error_history: list[tuple[int, str, str]]`

### 6. `tests/test_repair.py`

**Fixtures**:
- `lean_skeleton()` — a `SkeletonFile` with 2 sorry placeholders and one type mismatch
- `coq_skeleton()` — same for Coq
- `mock_env_with_errors(errors_sequence)` — mock that returns errors on first N calls, then success. Configurable error sequence.
- `mock_llm_repair()` — mock LLM that returns valid patches for known error patterns

**Test classes**:

`TestErrorClassifier`:
- `test_classify_type_mismatch` — "type mismatch, expected List Nat, got List Int" → `TYPE_MISMATCH`
- `test_classify_missing_import` — "unknown identifier 'Nat.add_comm'" → `MISSING_IMPORT`
- `test_classify_unsolved_goal` — "unsolved goals" → `UNSOLVED_GOAL`
- `test_classify_unknown` — garbage text → `UNKNOWN`
- `test_deterministic_fix_import` — returns import suggestion
- `test_deterministic_fix_returns_none` — type mismatch has no deterministic fix

`TestPatcher`:
- `test_apply_single_patch` — replaces lines correctly
- `test_apply_multiple_patches` — bottom-up application preserves line numbers
- `test_overlapping_patches_raises` — overlapping ranges raise ValueError
- `test_find_sorry_locations_lean` — finds `sorry` in Lean source
- `test_find_sorry_locations_coq` — finds `Admitted.` in Coq source

`TestRepairGraph`:
- `test_happy_path_no_errors` — skeleton compiles first try → 0 iterations, `compiled_ok=True`
- `test_deterministic_fix_resolves` — missing import → auto-fix → compiles
- `test_llm_repair_type_mismatch` — type mismatch → LLM patch → compiles
- `test_sorry_elimination` — skeleton with sorry → LLM generates tactic → compiles
- `test_budget_exhaustion` — errors persist → agent stops at `max_iterations`
- `test_mixed_errors` — deterministic + LLM fixes in one session

`TestSynthesizerAgent`:
- `test_synthesize_end_to_end` — full pipeline: skeleton → repair → `SynthesisResult`
- `test_synthesize_preserves_correct_code` — patches don't corrupt working definitions

---

## Modified Files (3)

### 7. `ageom/synthesizer/models.py` — Add `SynthesisResult` and `Patch`

Add the `SynthesisResult` and `Patch` dataclasses described above to the models file created in Phase 1.

### 8. `ageom/cli.py` — Add `synthesize` subcommand

**New subparser** `synthesize` (after `assemble`):
- `cdg_file` positional arg — path to CDG JSON
- `matches_file` positional arg — path to match results JSON
- `--prover` (choices: `lean4`, `coq`, default: `lean4`)
- `--output` (str) — output path for final verified source
- `--max-iterations` (int, default: 10)
- `--llm-provider`, `--llm-model`, `--llm-max-tokens`

**New handler** `async _cmd_synthesize(args)`:
1. Load CDG + match results
2. Run Phase 1 assembler → `SkeletonFile`
3. Set up `ProofEnvironment` + `LLMClient`
4. Create `SynthesizerAgent`, call `synthesize(skeleton)` → `SynthesisResult`
5. Write final source to `--output`
6. Print summary: iterations, patches, sorry count, compile status

### 9. `ageom/config.py` — Add synthesizer config fields

```python
# Synthesizer (Round 3)
synthesizer_max_iterations: int = 10
synthesizer_llm_provider: str = ""  # falls back to llm_provider
synthesizer_llm_model: str = "claude-sonnet-4-5-20250929"
```

---

## Key Design Decisions

1. **Error classifier before LLM** — deterministic fixes (missing imports, known coercions) are applied without an LLM call. This saves tokens and latency for the ~30% of errors that have mechanical solutions.

2. **pydantic-graph, not LangGraph** — the repair loop is a simple state machine (compile → fix → recompile) without the checkpoint/time-travel needs of Round 1. pydantic-graph matches the Hunter pattern and keeps the dependency light.

3. **Patch-based editing** — the LLM returns line-range patches, not full file rewrites. This prevents the LLM from accidentally rewriting correct code. Patches are applied bottom-up to preserve line numbers.

4. **Sorry elimination is a separate node** — once hard errors are resolved, the remaining work is filling in `sorry` placeholders with tactic proofs. This is a different LLM task (goal-directed tactic generation vs. error analysis) and gets its own prompt template.

5. **Error priority ordering** — `TYPE_MISMATCH` is fixed before `UNSOLVED_GOAL` because type errors cascade: fixing a type mismatch often resolves downstream goal errors. This reduces total iterations.

6. **Budget-bounded** — the agent stops after `max_iterations` even if sorrys remain. The output is still useful (partially verified) and prevents infinite loops on genuinely hard proofs.

7. **No Pantograph in Phase 2** — the plan described Pantograph integration for inspecting tactic state. This is deferred: `ProofEnvironment.check_proof()` + raw error parsing is sufficient for the repair loop. Pantograph can be added later as a `ProofEnvironment` enhancement if raw error messages prove insufficient for tactic suggestion.

---

## Verification

1. `pytest tests/test_repair.py -v` — all unit tests pass
2. Manual: `ageom synthesize cdg.json matches.json --output verified.lean` → watch repair iterations → inspect final file
3. Manual: deliberately introduce a type mismatch in a skeleton → verify the agent fixes it
4. `pytest tests/ -v` — full suite still green
