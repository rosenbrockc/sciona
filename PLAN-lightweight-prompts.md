# Plan: Replace 3 LLM Prompts with Lightweight Alternatives

Replace `hunter_score`, `architect_strategy`, and `architect_critique` with
deterministic or embedding-based alternatives. These three prompts have the
highest pattern-matching-to-reasoning ratio and the simplest output structures.

---

## Constraints

- No new core dependencies. `numpy` is already core; `fastembed` and `faiss-cpu`
  are in the `indexer` optional group and already loaded when the hunter runs.
- Each replacement must implement the same async interface (`complete(system, user) -> str`)
  so it can be wired in via `select_llm` without touching the node logic.
- Graceful degradation: if the lightweight path produces low-confidence results,
  fall back to the LLM. This keeps the LLM as a safety net, not a hard removal.
- Benchmark must stay green. All existing prompt benchmark cases must pass with
  the new implementations.

---

## Task A: `hunter_score` — Embedding Reranker

**Current behavior**: LLM receives `[i] name : type_signature` list plus a
statement and description, returns a JSON integer array ranking candidates best
to worst. Output is ~15 tokens.

**Replacement**: Score each candidate by cosine similarity between the query
embedding and the candidate's pre-computed embedding, then sort.

### A1. Add `EmbeddingReranker` class

**File**: `ageom/hunter/embedding_reranker.py` (new)

```python
class EmbeddingReranker:
    """Rank candidates by embedding similarity — no LLM call."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def complete(self, system: str, user: str) -> str:
        """Parse candidates from the user prompt, rank by similarity."""
        query, candidates = _parse_score_prompt(user)
        query_vec = self._embedder.embed(query)
        scored = []
        for idx, (name, type_sig) in enumerate(candidates):
            cand_vec = self._embedder.embed(f"{name} : {type_sig}")
            sim = float(np.dot(query_vec, cand_vec))
            scored.append((sim, idx))
        scored.sort(reverse=True)
        return json.dumps([idx for _, idx in scored])

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
```

`_parse_score_prompt(user)` extracts the statement+description as query text and
the `[i] name : type_signature` lines as candidates. This is straightforward
regex on the formatted `SCORE_CANDIDATES_USER` template.

### A2. Add type-signature boost

Pure embedding similarity misses type compatibility. Add a bonus when the
candidate's return type overlaps with the query's expected output:

```python
def _type_bonus(statement: str, type_sig: str) -> float:
    """Small boost when return type tokens overlap with the statement."""
    sig_tokens = set(type_sig.lower().replace("->", " ").split())
    stmt_tokens = set(statement.lower().split())
    overlap = len(sig_tokens & stmt_tokens)
    return 0.1 * min(overlap, 3)  # cap at 0.3
```

Add to the score: `sim + _type_bonus(query, type_sig)`.

### A3. Wire into `select_llm` with confidence gate

**File**: `ageom/llm_router.py`

Add a `lightweight_override` path in `select_llm` for `HUNTER_SCORE`:

```python
if prompt_key == HUNTER_SCORE and deps.embedding_reranker is not None:
    return deps.embedding_reranker
```

The `embedding_reranker` is constructed in the HunterAgent init when the embedder
is available (it always is when the index is loaded).

### A4. Confidence gate (optional, defer to v2)

If the top-2 similarity scores are within 0.05, the ranking is ambiguous — fall
back to the LLM. This keeps the LLM as a safety net for close calls.

### A5. Benchmark expectations

- The `bge-small-en-v1.5` embedder should rank `apply_iir_filter` above
  `compute_frequency_response` for "Apply a stable bandpass filter to ECG
  samples" — the name overlap is strong.
- Similarly, `dijkstra` > `topological_sort` for "shortest path distances".
- All 4 `score_*` benchmark cases should pass. If any fail, tune the type bonus
  weight.

### Estimated latency

~2ms per candidate (embed + dot product). For 20 candidates: **~40ms** vs
**~8000ms** with codex shim. **200x speedup**.

---

## Task B: `architect_strategy` — Keyword Classifier with Embedding Fallback

**Current behavior**: LLM receives a goal string and a paradigm list, returns
`{"paradigm": "...", "rationale": "..."}`. Output is ~20 tokens.

**Replacement**: Two-tier classifier:
1. Keyword scan against known paradigm signatures
2. Embedding similarity fallback against paradigm descriptions

### B1. Add `StrategyClassifier` class

**File**: `ageom/architect/strategy_classifier.py` (new)

```python
# Tier 1: keyword → paradigm mapping (high precision, incomplete recall)
_KEYWORD_RULES: list[tuple[set[str], ConceptType]] = [
    ({"sort", "sorting", "order", "rank"}, ConceptType.SORTING),
    ({"search", "binary search", "find", "lookup"}, ConceptType.SEARCHING),
    ({"divide", "conquer", "split", "merge", "recursive"}, ConceptType.DIVIDE_AND_CONQUER),
    ({"greedy", "locally optimal"}, ConceptType.GREEDY),
    ({"dynamic programming", "dp table", "recurrence", "memoize", "edit distance", "lcs", "subsequence"}, ConceptType.DYNAMIC_PROGRAMMING),
    ({"shortest path", "dijkstra", "bellman", "minimum spanning"}, ConceptType.GRAPH_OPTIMIZATION),
    ({"bfs", "dfs", "traverse", "reachability", "connected component"}, ConceptType.GRAPH_TRAVERSAL),
    ({"pattern match", "regex", "kmp", "aho-corasick", "suffix"}, ConceptType.STRING_MATCHING),
    ({"fft", "fourier", "spectrum", "frequency", "wavelet"}, ConceptType.SIGNAL_TRANSFORM),
    ({"filter", "bandpass", "lowpass", "highpass", "iir", "fir"}, ConceptType.SIGNAL_FILTER),
    ({"cholesky", "linear system", "matrix factor", "lu decomp", "eigenvalue"}, ConceptType.ALGEBRA),
    ({"convex hull", "voronoi", "delaunay"}, ConceptType.GEOMETRY),
    ({"prime", "gcd", "modular", "sieve"}, ConceptType.NUMBER_THEORY),
    ({"mcmc", "metropolis", "hamiltonian"}, ConceptType.MCMC_KERNEL),
    ({"variational", "elbo", "kl divergence"}, ConceptType.VI_ELBO),
    ({"kalman", "particle filter", "sequential"}, ConceptType.SEQUENTIAL_FILTER),
    ({"belief propagation", "message passing"}, ConceptType.MESSAGE_PASSING),
]
```

Keyword scan: tokenize the goal, check if any keyword set is a subset of the
goal tokens (case-insensitive, with bigram matching for multi-word keywords).

### B2. Tier 2: embedding fallback

For each `ConceptType` that has a `SKELETON_TEMPLATES` entry, pre-compute an
embedding of its description. At runtime, embed the goal and pick the highest
cosine-similarity paradigm.

```python
# Tier 2: computed once at init
_PARADIGM_EMBEDDINGS: dict[ConceptType, np.ndarray] = {}

def _init_paradigm_embeddings(embedder: Embedder) -> None:
    for ct, skeleton in SKELETON_TEMPLATES.items():
        _PARADIGM_EMBEDDINGS[ct] = embedder.embed(
            f"{ct.value}: {skeleton.description}"
        )
```

### B3. Confidence gate

If the top embedding score is < 0.4 or the gap to the second is < 0.05, fall
back to the LLM. Return `ConceptType.CUSTOM` only as last resort.

### B4. Response format

The classifier returns the same JSON string the LLM would:

```python
async def complete(self, system: str, user: str) -> str:
    goal = _extract_goal(user)  # parse from SELECT_STRATEGY_USER template
    paradigm, confidence = self._classify(goal)
    if confidence < self._fallback_threshold:
        return await self._llm_fallback.complete(system, user)
    return json.dumps({
        "paradigm": paradigm.value,
        "rationale": f"keyword/embedding match (confidence={confidence:.2f})",
    })
```

### B5. Wire into `select_llm`

Same pattern as Task A: if `deps.strategy_classifier is not None`, return it
for `ARCHITECT_STRATEGY`.

### Estimated latency

Tier 1 (keyword): **<1ms**. Tier 2 (embed + 17 dot products): **~5ms**.
vs **~8000ms** with shim. **1600x speedup**.

---

## Task C: `architect_critique` — Structural Validator

**Current behavior**: Phase A runs 6 deterministic checks. If they pass, Phase B
calls the LLM for semantic judgment. The LLM returns
`{"approved": bool, "reason": "..."}`. The system already fail-opens on LLM
parse errors (deterministic checks are the hard gate).

**Replacement**: Extend Phase A with 3 additional structural checks that cover
what the LLM typically catches. Make the LLM call opt-in rather than default.

### C1. Add reachability check

**File**: `ageom/architect/structural_critic.py` (new)

Check that all parent outputs are reachable from parent inputs via the sub-node
DAG. This is the main thing the LLM catches that Phase A misses.

```python
def _check_output_reachability(
    parent: AlgorithmicNode,
    children: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> list[str]:
    """Verify all parent outputs are producible from parent inputs via children."""
    # Build DAG: node_id → set of output names it provides
    # BFS from parent inputs through edges
    # Flag any parent output not reached
```

### C2. Add coverage check

Verify that every parent input is consumed by at least one child, and every
parent output is produced by at least one child. Currently only edges are checked
for name validity, not coverage.

```python
def _check_io_coverage(
    parent: AlgorithmicNode,
    children: list[AlgorithmicNode],
) -> list[str]:
    """Verify parent I/O is fully covered by children."""
    parent_input_names = {io.name for io in parent.inputs}
    parent_output_names = {io.name for io in parent.outputs}
    child_input_names = {io.name for child in children for io in child.inputs}
    child_output_names = {io.name for child in children for io in child.outputs}
    issues = []
    uncovered_inputs = parent_input_names - child_input_names
    if uncovered_inputs:
        issues.append(f"parent inputs not consumed: {uncovered_inputs}")
    uncovered_outputs = parent_output_names - child_output_names
    if uncovered_outputs:
        issues.append(f"parent outputs not produced: {uncovered_outputs}")
    return issues
```

### C3. Add name-similarity deduplication check

Detect near-duplicate child nodes (e.g., "Initialize Table" and "Init Table")
that suggest the LLM decomposer stuttered.

```python
def _check_duplicate_children(
    children: list[AlgorithmicNode],
    threshold: float = 0.85,
) -> list[str]:
    """Flag near-duplicate child nodes by name similarity."""
    # Jaccard similarity on name tokens
    # If any pair > threshold, flag
```

### C4. Compose into `StructuralCritic`

```python
class StructuralCritic:
    """Deterministic decomposition validator — replaces LLM critique."""

    async def complete(self, system: str, user: str) -> str:
        parent, children, edges = _parse_critique_prompt(user)
        issues = []
        issues.extend(_check_output_reachability(parent, children, edges))
        issues.extend(_check_io_coverage(parent, children))
        issues.extend(_check_duplicate_children(children))
        approved = len(issues) == 0
        return json.dumps({
            "approved": approved,
            "reason": "; ".join(issues) if issues else "structural checks passed",
            "io_issues": issues,
            "flagged_nodes": [],
        })
```

### C5. Integration strategy

The critique node already has Phase A (deterministic) and Phase B (LLM). The
cleanest integration is:

1. Add the 3 new checks to Phase A (lines 1051-1123 in `architect/nodes.py`)
2. Make Phase B (LLM call) conditional on a config flag:
   `config.architect_critique_llm_enabled` (default: `False`)
3. When disabled, Phase A alone decides. When enabled, LLM runs as before.

This avoids the `complete()` interface hack entirely — the checks live where
they belong, in the deterministic validation layer.

### C6. Benchmark adaptation

The prompt benchmark cases for `architect_critique` test the LLM response format.
With the LLM disabled by default, those cases need a `use_llm=True` override or
a separate "structural critic" benchmark that validates the deterministic path.

Add a config flag to the benchmark runner:
```python
# In benchmark_validation.py, FixturePromptBenchmarkLLM still returns
# correct JSON for critique cases. The benchmark tests the prompt/response
# contract, not the production path — so no change needed there.
```

### Estimated latency

3 structural checks: **<1ms** total (graph traversal on ~5 nodes).
vs **~8000ms** with shim. Eliminates the LLM call entirely for >95% of cases.

---

## Implementation Order

1. **Task B** (strategy classifier) — simplest, self-contained, no index dependency
2. **Task A** (embedding reranker) — depends on embedder being available
3. **Task C** (structural critic) — modifies existing Phase A, most invasive

Each task is a single commit. Run full regression + prompt benchmarks after each.

---

## Metrics to Track

After implementation, re-run `benchmark.sh` to compare:

| Metric | Before | Target |
|---|---|---|
| `hunter_score` avg latency | ~8000ms (codex) | <50ms |
| `architect_strategy` avg latency | ~8000ms (codex) | <10ms |
| `architect_critique` avg latency | ~8000ms (codex) | <1ms |
| Tuned pass rate | 66/66 (codex) | 66/66 (no regression) |
| LLM calls per pipeline run | ~16 | ~13 (3 fewer) |

---

## Risks

1. **Embedding quality**: `bge-small-en-v1.5` (384d) may not distinguish
   fine-grained type differences. Mitigation: type bonus in scorer, confidence
   gate with LLM fallback.

2. **Keyword brittleness**: Novel algorithm families won't match any keyword
   rule. Mitigation: Tier 2 embedding fallback, then LLM fallback.

3. **Structural critic false negatives**: The 3 new checks may miss subtle
   semantic issues (e.g., correct structure but wrong algorithm choice).
   Mitigation: LLM critique still available via config flag; the existing
   fail-open behavior means false approval is already accepted.

4. **Prompt parsing fragility**: The lightweight alternatives must parse the
   formatted prompt text to extract inputs. If prompt templates change, the
   parsers break. Mitigation: share constants/templates, add regression tests
   that break if the template format changes.
