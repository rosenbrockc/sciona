# Plan: Three High-Impact Fixes

Based on the e2e benchmark results (structured mode finds all 3 ground truth
atoms but verifies 0/11), these three changes address the biggest bottlenecks
in order of impact.

---

## 1. Fix Python Verification Oracle (HIGH — unblocks all matches)

### Problem

The verification oracle rejects every Python candidate with:
```
error: Invalid syntax. Perhaps you forgot a comma?
```

**Root cause:** `PythonEnvironment.check_term()` builds:
```python
_result: (signal: np.ndarray) -> np.ndarray = @ageoa.pronto.bandpass_filter
```
This is invalid Python because:
- Function signatures like `(x: int) -> int` are not valid type annotations
- The `@` prefix is Lean syntax, not Python
- Even valid Callable types would fail because the RHS is a function object,
  not a value of that type

**Evidence:** All 617 verification attempts in the structured benchmark failed
with identical syntax errors.

### Fix

**File:** `ageom/judge/python_env.py`

Replace the current mypy-based type check with a structural compatibility
check that actually works for Python function matching:

```python
async def check_term(self, term: str, expected_type: str) -> CompilerFeedback:
    """Verify a Python function matches the expected signature."""
    # Strip @ prefix (Lean artifact)
    func_name = term.lstrip("@")

    code = textwrap.dedent(f"""\
        import typing, numpy as np
        from typing import Callable
        from {_module(func_name)} import {_leaf(func_name)}

        # Assert the function is callable
        assert callable({_leaf(func_name)})

        # If we have a type signature, check parameter count
        import inspect
        _sig = inspect.signature({_leaf(func_name)})
    """)

    result = await self._run_check(code)
    return result
```

The key insight: for Python atoms, "verification" should mean:
1. The function exists and is importable
2. It is callable
3. Its parameter count is compatible with the expected signature

Full mypy type-checking requires proper `Callable[[arg_types], return_type]`
annotations, which the architect's type signatures don't produce. A
signature-arity check is the right level of verification for the current
system.

### Steps

1. **Add `_build_python_check_code()`** in `python_env.py` that generates
   importable, runnable Python code instead of mypy annotations.

2. **Add arity extraction** — parse the expected type signature's parameter
   count and compare against `inspect.signature()`.

3. **Keep mypy path as opt-in** — add `AGEOM_PYTHON_VERIFY_MODE=mypy|import`
   config flag. Default to `import` (the new path).

4. **Update `checker.py`** — the `@` prefix stripping should happen in
   `verify_candidate()` before calling the environment, not in the env.

### Test

Re-run `ageom run` on the ECG goal in structured mode. Expect >0 verified
matches for `bandpass_filter`, `r_peak_detection`, `heart_rate_computation`.

### Risk

Low. The current oracle verifies nothing (100% rejection). Any improvement
is strictly better. The import-based check is weaker than mypy but actually
works.

---

## 2. Enable Hunter Concurrency (MEDIUM — reduces latency 3-5x)

### Problem

The orchestrator matches leaves sequentially (`orchestrator_hunter_concurrency=1`).
With 11 leaves and ~8-10s per LLM call, structured mode takes 18 minutes.
Verified mode (143 nodes, 3 rounds) took 2.7 hours.

### Current State

The concurrency infrastructure already exists:
- `ageom/orchestrator.py:422-446` — semaphore-bounded `asyncio.gather()` for
  concurrent `hunter.find_match()` calls
- `ageom/config.py:197` — `orchestrator_hunter_concurrency: int = 1`
- `HunterAgent.find_match()` creates per-call `HunterState` (no shared mutable state)
- FAISS index is read-only, LLM clients are async-safe, oracle is stateless

### Fix

**File:** `ageom/config.py`

```python
orchestrator_hunter_concurrency: int = 4  # was 1
```

### Safety Analysis

| Shared resource | Concurrent access | Safe? |
|---|---|---|
| FAISSStore (index) | Read-only search | Yes |
| LLMClient (shim) | Async API calls | Yes (2 worker sockets) |
| VerificationOracle | Async, stateless | Yes |
| SharedContextStore | Async put/search | Yes |
| HunterState | Per-call instance | Yes (no sharing) |

**Caveat:** The codex shim spawns 2 worker sockets. With concurrency=4, two
hunter calls will queue behind the workers. This still provides ~2x throughput.
For full 4x, increase shim workers to 4 (`--workers 4` in cli_daemon.py).

### Steps

1. **Change default to 4** in `config.py`.
2. **Add `AGEOM_ORCHESTRATOR_HUNTER_CONCURRENCY` env var** (already supported
   by pydantic-settings — just change the default).
3. **Verify shim worker count** — ensure codex_shim spawns ≥4 workers when
   concurrency is 4. Check `cli_socket_shim.py` for `--workers` default.
4. **Run e2e benchmark** — expect structured mode to drop from 18min to ~5-7min.

### Risk

Low. The code already exists and is tested. The default of 1 was conservative.
If any concurrency issues surface, users can set the env var back to 1.

---

## 3. Implement Lightweight Prompt Replacements (MEDIUM — eliminates 3 LLM calls)

### Current State

Key finding from exploration: **Tasks B and C are already implemented.**

- `StrategyClassifier` exists at `ageom/architect/strategy_classifier.py`
  with 77 phrase rules and embedding fallback
- `structural_critic.py` has all 4 deterministic checks (basic structure,
  IO coverage, duplicate detection, output reachability)
- `architect_critique_llm_enabled` defaults to `False` — critique LLM is
  already disabled

Only **Task A (EmbeddingReranker for hunter_score)** needs new code.

### Task A: hunter_score — Embedding Reranker

**Current behavior:** LLM receives `[i] name : type_sig` list, returns JSON
int array ranking. Called via `select_llm(deps.llm, HUNTER_SCORE)` in
`ageom/hunter/nodes.py:255`.

**File:** `ageom/hunter/embedding_reranker.py` (new)

```python
class EmbeddingReranker:
    """Rank candidates by embedding similarity — no LLM call."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def complete(self, system: str, user: str) -> str:
        query, candidates = _parse_score_prompt(user)
        query_vec = self._embedder.embed(query)
        scored = []
        for idx, (name, type_sig) in enumerate(candidates):
            cand_vec = self._embedder.embed(f"{name} : {type_sig}")
            sim = float(np.dot(query_vec, cand_vec))
            sim += _type_bonus(query, type_sig)
            scored.append((sim, idx))
        scored.sort(reverse=True)
        return json.dumps([idx for _, idx in scored])

    async def complete_with_grammar(self, system, user, grammar):
        return await self.complete(system, user)
```

**`_parse_score_prompt(user)`** extracts:
- Query: text between "Statement:" and "Candidates:" (or similar markers
  from `SCORE_CANDIDATES_USER` template)
- Candidates: `[i] name : type_sig` lines via regex

**`_type_bonus(query, type_sig)`** adds 0.1 per overlapping token between
the query and the type signature tokens (capped at 0.3).

### Steps

1. **Create `embedding_reranker.py`** with `EmbeddingReranker` class.
2. **Add `_parse_score_prompt()`** — regex parser for the SCORE_CANDIDATES_USER
   template format.
3. **Wire into HunterAgent init** — when embedder is available (it always is
   when the FAISS index is loaded), create the reranker.
4. **Add routing in `select_llm`** or in `HunterDeps` — if
   `deps.embedding_reranker` is set, return it for `HUNTER_SCORE`.
5. **Add confidence gate** (v2) — if top-2 scores are within 0.05, fall back
   to LLM.
6. **Benchmark** — run prompt benchmark to verify all 4 `score_*` cases pass.

### Task B: architect_strategy — Already Done

`StrategyClassifier` at `ageom/architect/strategy_classifier.py:98` already
implements the two-tier classifier (keyword rules + embedding fallback).
It's wired into the architect graph and active in production.

**Verify:** Check that the strategy classifier is being used instead of the
LLM in structured/verified modes. The routing log from the benchmark shows
`architect_strategy` is suppressed in structured mode — meaning the
deterministic classifier handles it.

### Task C: architect_critique — Already Done

The structural critic has all 4 checks implemented, and
`architect_critique_llm_enabled` defaults to `False`. The LLM critique is
opt-in.

**Verify:** Confirm `config.architect_critique_llm_enabled` is `False` in
production and that the 4 structural checks provide adequate coverage.

### Estimated Impact

| Prompt | Before | After | Speedup |
|---|---|---|---|
| hunter_score | ~8s (LLM) | ~40ms (embed) | 200x |
| architect_strategy | ~8s (LLM) | <5ms (classifier) | Already done |
| architect_critique | ~8s (LLM) | <1ms (structural) | Already done |

Net: **1 fewer LLM call per hunter leaf** (hunter_score). For 11 leaves in
structured mode: saves ~88s. For 143 nodes in verified mode: saves ~19 min.

---

## Implementation Order

1. **Fix #1 (verification oracle)** — highest impact, unblocks the entire
   pipeline from producing verified matches. Without this, nothing else
   matters for e2e quality.

2. **Fix #2 (hunter concurrency)** — trivial change (one integer), immediate
   latency improvement. Do alongside Fix #1.

3. **Fix #3 (embedding reranker)** — only Task A needs code. Tasks B and C
   are already shipped. Implement after Fix #1 confirms verified matches work.

---

## Success Criteria

After all three fixes, re-run `e2e_benchmark.sh`:

| Metric | Before | Target |
|---|---|---|
| Verified matches (structured) | 0/11 | ≥3/11 (ground truth atoms) |
| Verified matches (verified) | 0/143 | ≥3 (ground truth atoms) |
| Structured latency | 18 min | <7 min |
| Verified latency | 2.7 hr | <45 min |
| LLM calls per leaf (hunter) | 3 (score+reformulate+analyze) | 2 (reformulate+analyze) |
