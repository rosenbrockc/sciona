# Benchmark Hardening Plan

The current benchmarks are structurally sound as plumbing tests but insufficient
for proving the value proposition. This document is an actionable implementation
plan with exact file paths, data structures, and test case specifications.

---

## Task 1: Continuous Quality Metrics on `FlowBenchmarkResult` ✓ DONE

**File**: `ageom/flow_benchmark.py`

### 1a. Add fields to `FlowBenchmarkResult` (line ~47)

Add these fields after `node_count` (line 58):

```python
@dataclass
class FlowBenchmarkResult:
    # ... existing fields ...
    node_count: int
    leaf_coverage: float = 0.0          # matched_leaves / total_leaves
    best_similarity: float = 0.0        # max token-overlap score across all leaves
    decomposition_depth: int = 0        # max depth of CDG tree
    decomposition_leaf_count: int = 0   # len(cdg.leaf_nodes())
    decomposition_edge_count: int = 0   # len(cdg.edges)
    error: str = ""
```

### 1b. Compute `leaf_coverage` at each call site

In every `_run_*_case` function, set `leaf_coverage=matched / max(1, len(case.leaves))`.

### 1c. Compute `best_similarity` via `_LexicalSemanticIndex._score`

Add a helper after `_matched_leaf_count` (line ~375):

```python
def _best_similarity_score(
    case: FlowBenchmarkCase,
    index: _LexicalSemanticIndex,
) -> float:
    """Max token-overlap score between any leaf query_hint and any declaration."""
    best = 0.0
    for leaf in case.leaves:
        for decl, score in index.search_by_embedding(leaf.query_hint, k=1):
            best = max(best, score)
    return best
```

Call this from each `_run_*_case` that instantiates an index. For `direct_baseline`,
call it on the single raw prompt against the index.

### 1d. Compute decomposition metrics from `CDGExport`

Add a helper:

```python
def _cdg_metrics(cdg: CDGExport) -> tuple[int, int, int]:
    """Return (depth, leaf_count, edge_count) for a CDG."""
    leaf_count = len(cdg.leaf_nodes())
    edge_count = len(cdg.edges)
    # depth: BFS from roots
    children: dict[str, list[str]] = {}
    for edge in cdg.edges:
        children.setdefault(edge.source, []).append(edge.target)
    node_ids = {n.node_id for n in cdg.nodes}
    child_ids = {edge.target for edge in cdg.edges}
    roots = node_ids - child_ids or node_ids
    depth = 0
    frontier = list(roots)
    while frontier:
        depth += 1
        next_frontier = []
        for nid in frontier:
            next_frontier.extend(children.get(nid, []))
        frontier = next_frontier
    return depth, leaf_count, edge_count
```

Call from `_run_structured_case` and `_run_verified_case` after decomposition.
For `direct_baseline` and `rapid`, set `decomposition_depth=1`,
`decomposition_leaf_count=1`, `decomposition_edge_count=0`.

### 1e. Add aggregate fields to `FlowBenchmarkAggregate` (line ~65)

```python
@dataclass
class FlowBenchmarkAggregate:
    # ... existing fields ...
    avg_leaf_coverage: float = 0.0
    avg_best_similarity: float = 0.0
```

Update `record()` to compute running averages of `leaf_coverage` and
`best_similarity` (same pattern as `avg_latency_ms`). Update `to_dict()` to
include both.

### 1f. Update validation gate in `benchmark_validation.py` (line ~560)

After the existing `benchmark_passed` computation, add a monotonicity check:

```python
flow_agg_map = {agg.variant: agg for agg in flow_aggregates}
coverage_monotonic = (
    flow_agg_map.get("structured", _zero).avg_leaf_coverage
    >= flow_agg_map.get("rapid", _zero).avg_leaf_coverage
    and flow_agg_map.get("verified", _zero).avg_leaf_coverage
    >= flow_agg_map.get("structured", _zero).avg_leaf_coverage
)
```

Add `coverage_monotonic` to the `benchmark_passed` conjunction and include
`"coverage_monotonic": coverage_monotonic` in the summary dict.

### 1g. Update tests

**File**: `tests/test_flow_benchmark.py`

In `test_flow_benchmark_summary_orders_variants_by_success`, add:

```python
assert aggregate_map["structured"].avg_leaf_coverage == pytest.approx(1.0)
assert aggregate_map["verified"].avg_leaf_coverage == pytest.approx(1.0)
assert aggregate_map["rapid"].avg_leaf_coverage == pytest.approx(0.0)
```

**File**: `tests/test_benchmark_validation.py`

In `test_run_benchmark_validation_writes_bundle`, add:

```python
assert payload["coverage_monotonic"] is True
```

---

## Task 2: Expand Flow Benchmark Cases ✓ DONE

**File**: `ageom/flow_benchmark.py`, function `default_flow_benchmark_cases`

**Implementation notes**: Single-leaf case was replaced with a 2-leaf binary
search case because the `DecompositionAgent` graph state machine requires at
least 2 sub-nodes for clean decomposition. All leaf names and query hints were
tuned to avoid ambiguous token overlap in `_hint_matches` (e.g. "Compute FFT"
→ "Fourier Transform" since "fft" doesn't appear in descriptions; matrix
split/combine hints made fully distinct to avoid first-match collision in
`_LeafOracle`).

**Original plan below for reference**:

### 2a. Add a single-leaf case (rapid should pass)

```python
FlowBenchmarkCase(
    case_id="search_binary",
    domain="sorting",
    prompt="Find the position of a target value in a sorted array using binary search.",
    concept_type=ConceptType.SORTING,
    leaves=(
        FlowLeafSpec(
            name="Binary Search",
            description="Find the position of a target in a sorted array.",
            type_signature="list[int] -> int -> int",
            query_hint="Binary Search",
            declaration_name="algorithms.binary_search",
            inputs=(("sorted_array", "list[int]"), ("target", "int")),
            outputs=(("position", "int"),),
        ),
    ),
),
```

This case has exactly 1 leaf, so `direct_baseline` and `rapid` can match it.

### 2b. Add a 3-leaf case (deeper pipeline)

```python
FlowBenchmarkCase(
    case_id="fft_spectral_analysis",
    domain="dsp",
    prompt="Window a signal, compute its FFT, then extract the magnitude spectrum.",
    concept_type=ConceptType.SIGNAL_FILTER,
    leaves=(
        FlowLeafSpec(
            name="Apply Window",
            description="Apply a Hann window to a signal segment.",
            type_signature="signal -> signal",
            query_hint="Apply Window",
            declaration_name="algorithms.apply_hann_window",
            inputs=(("signal", "np.ndarray"),),
            outputs=(("windowed", "np.ndarray"),),
        ),
        FlowLeafSpec(
            name="Compute FFT",
            description="Compute the Fast Fourier Transform of a windowed signal.",
            type_signature="signal -> complex_spectrum",
            query_hint="Compute FFT",
            declaration_name="algorithms.compute_fft",
            inputs=(("windowed", "np.ndarray"),),
            outputs=(("spectrum", "np.ndarray"),),
        ),
        FlowLeafSpec(
            name="Extract Magnitude",
            description="Extract magnitude spectrum from complex FFT output.",
            type_signature="complex_spectrum -> magnitude",
            query_hint="Extract Magnitude",
            declaration_name="algorithms.extract_magnitude",
            inputs=(("spectrum", "np.ndarray"),),
            outputs=(("magnitude", "np.ndarray"),),
        ),
    ),
),
```

### 2c. Add a diamond-dependency case

```python
FlowBenchmarkCase(
    case_id="matrix_multiply_strassen",
    domain="linear_algebra",
    prompt="Multiply two matrices by splitting into quadrants, recursing, then combining.",
    concept_type=ConceptType.LINEAR_ALGEBRA,
    leaves=(
        FlowLeafSpec(
            name="Split Quadrants",
            description="Split a matrix into four quadrants.",
            type_signature="matrix -> tuple[matrix, matrix, matrix, matrix]",
            query_hint="Split Quadrants",
            declaration_name="algorithms.split_matrix_quadrants",
            inputs=(("matrix", "np.ndarray"),),
            outputs=(("q11", "np.ndarray"), ("q12", "np.ndarray"),
                     ("q21", "np.ndarray"), ("q22", "np.ndarray")),
        ),
        FlowLeafSpec(
            name="Combine Quadrants",
            description="Combine four quadrant results into a single matrix.",
            type_signature="tuple[matrix, matrix, matrix, matrix] -> matrix",
            query_hint="Combine Quadrants",
            declaration_name="algorithms.combine_matrix_quadrants",
            inputs=(("q11", "np.ndarray"), ("q12", "np.ndarray"),
                    ("q21", "np.ndarray"), ("q22", "np.ndarray")),
            outputs=(("result", "np.ndarray"),),
        ),
    ),
),
```

### 2d. Add a string/DP domain case

```python
FlowBenchmarkCase(
    case_id="string_edit_distance",
    domain="strings",
    prompt="Compute the minimum edit distance between two strings using dynamic programming.",
    concept_type=ConceptType.DYNAMIC_PROGRAMMING,
    leaves=(
        FlowLeafSpec(
            name="Initialize DP Table",
            description="Initialize the edit distance DP table with base cases.",
            type_signature="str -> str -> list[list[int]]",
            query_hint="Initialize DP Table",
            declaration_name="algorithms.init_edit_distance_table",
            inputs=(("source", "str"), ("target", "str")),
            outputs=(("table", "list[list[int]]"),),
        ),
        FlowLeafSpec(
            name="Fill DP Table",
            description="Fill the edit distance table using recurrence relation.",
            type_signature="list[list[int]] -> str -> str -> list[list[int]]",
            query_hint="Fill DP Table",
            declaration_name="algorithms.fill_edit_distance_table",
            inputs=(("table", "list[list[int]]"), ("source", "str"), ("target", "str")),
            outputs=(("filled", "list[list[int]]"),),
        ),
    ),
),
```

### 2e. Update test assertions

**File**: `tests/test_flow_benchmark.py`

Change `test_default_flow_benchmark_cases_cover_multiple_domains`:

```python
cases = default_flow_benchmark_cases()
assert len(cases) == 7  # was 3
assert {case.domain for case in cases} >= {"sorting", "graph", "dsp", "linear_algebra", "strings"}
```

Update `test_flow_benchmark_summary_orders_variants_by_success`:

The single-leaf `search_binary` case means `direct_baseline` and `rapid` will
now pass 1 case instead of 0. Adjust:

```python
assert aggregate_map["direct_baseline"].passed_cases >= 1
assert aggregate_map["rapid"].passed_cases >= 1
assert aggregate_map["structured"].passed_cases == len(cases)
assert aggregate_map["verified"].passed_cases == len(cases)
```

---

## Task 3: Noisy Mock LLM for Meaningful Stability

**File**: `ageom/flow_benchmark.py`

### 3a. Add `_NoisyFlowArchitectLLM` after `_FlowArchitectLLM` (line ~183)

```python
import random

class _NoisyFlowArchitectLLM(_FlowArchitectLLM):
    """Architect mock that introduces controlled perturbations."""

    def __init__(
        self,
        case: FlowBenchmarkCase,
        *,
        seed: int | None = None,
        shuffle_prob: float = 0.3,
        drop_field_prob: float = 0.1,
        alter_desc_prob: float = 0.2,
    ) -> None:
        super().__init__(case)
        self._rng = random.Random(seed)
        self._shuffle_prob = shuffle_prob
        self._drop_field_prob = drop_field_prob
        self._alter_desc_prob = alter_desc_prob

    async def complete(self, system: str, user: str) -> str:
        raw = await super().complete(system, user)
        system_lower = system.lower()
        if not ("sub-nodes" in system_lower or "sub_nodes" in system_lower):
            return raw
        parsed = json.loads(raw)
        nodes = parsed.get("sub_nodes", [])
        if self._rng.random() < self._shuffle_prob:
            self._rng.shuffle(nodes)
        for node in nodes:
            if self._rng.random() < self._alter_desc_prob:
                node["description"] = node["description"] + " (variant)"
            if self._rng.random() < self._drop_field_prob:
                node.pop("matched_primitive", None)
        parsed["sub_nodes"] = nodes
        return json.dumps(parsed)
```

### 3b. Add a `noisy` kwarg to `_decompose_case` (line ~391)

```python
async def _decompose_case(
    case: FlowBenchmarkCase,
    *,
    noisy: bool = False,
    noise_seed: int | None = None,
) -> tuple[Any, _FlowArchitectLLM]:
    catalog = _make_catalog(case)
    if noisy:
        architect_llm = _NoisyFlowArchitectLLM(case, seed=noise_seed)
    else:
        architect_llm = _FlowArchitectLLM(case)
    agent = DecompositionAgent(
        catalog=catalog,
        skill_index=_EmptySkillIndex(),
        llm=architect_llm,
        max_depth=6,
    )
    cdg = await agent.decompose(case.prompt)
    return cdg, architect_llm
```

### 3c. Wire noisy mode into `run_flow_benchmark` (line ~573)

Add a `noisy: bool = False` parameter. When `noisy=True` and `repeats > 1`,
pass `noisy=True, noise_seed=repeat_index` to `_decompose_case` in the
structured and verified variants. Keep `direct_baseline` and `rapid` unchanged
(they don't use the Architect).

```python
async def run_flow_benchmark(
    *,
    cases: Sequence[FlowBenchmarkCase],
    variants: Sequence[str] = ("direct_baseline", "rapid", "structured", "verified"),
    repeats: int = 1,
    noisy: bool = False,
) -> list[FlowBenchmarkResult]:
```

### 3d. Add test for noisy stability

**File**: `tests/test_flow_benchmark.py`

```python
@pytest.mark.asyncio
async def test_flow_benchmark_noisy_stability():
    cases = default_flow_benchmark_cases()[:1]  # single case
    results = await run_flow_benchmark(
        cases=cases,
        variants=("structured",),
        repeats=5,
        noisy=True,
    )
    aggregates = summarize_flow_benchmark(results)
    assert aggregates[0].repeat_groups == 1
    # With noisy mocks, stability may be < 1.0 — that's the point.
    # The test validates the code path runs without error.
    assert 0.0 <= aggregates[0].stability_rate <= 1.0
```

---

## Task 4: Refinement-Required Case for Verified Mode

**File**: `ageom/flow_benchmark.py`

### 4a. Add `_FailFirstHunterLLM` after `_BenchmarkHunterLLM` (line ~202)

This mock fails on the first call for a specific leaf, forcing the orchestrator's
refinement loop to trigger.

```python
class _FailFirstHunterLLM(_BenchmarkHunterLLM):
    """Hunter mock that reverses ranking on the first call, then succeeds."""

    def __init__(self) -> None:
        super().__init__()
        self._first_call_done = False

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        lower = system.lower()
        if ("rank" in lower or "score" in lower) and not self._first_call_done:
            self._first_call_done = True
            return "[3, 2, 1, 0]"  # reversed: distractor first
        return await super().complete(system, user)
```

### 4b. Add a `_make_fail_first_hunter` factory

```python
def _make_fail_first_hunter(
    case: FlowBenchmarkCase,
) -> tuple[HunterAgent, _FailFirstHunterLLM]:
    declarations = _make_declarations(case)
    hunter_llm = _FailFirstHunterLLM()
    hunter = HunterAgent(
        index=_LexicalSemanticIndex(declarations),
        oracle=_LeafOracle({
            leaf.query_hint: leaf.declaration_name for leaf in case.leaves
        }),
        llm=hunter_llm,
        max_iterations=3,  # allow retry after refinement
        top_k_verify=2,
        search_k=5,
    )
    return hunter, hunter_llm
```

### 4c. Add a `verified_with_refinement` variant

Add a new runner `_run_verified_refinement_case` that uses
`_make_fail_first_hunter` instead of `_make_hunter`. Register it as variant
`"verified_refinement"` in `run_flow_benchmark`'s dispatch.

This variant should pass for multi-leaf cases because the orchestrator calls
`refine_on_failure` after round 1 failures, then retries with the normal
ranking.

### 4d. Add to default variants

Do NOT add `verified_refinement` to the default variants tuple (it would break
existing tests). Instead, add a dedicated test:

**File**: `tests/test_flow_benchmark.py`

```python
@pytest.mark.asyncio
async def test_verified_refinement_recovers_from_initial_failure():
    cases = default_flow_benchmark_cases()[:1]
    results = await run_flow_benchmark(
        cases=cases,
        variants=("verified_refinement",),
    )
    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].prompt_calls > 0
```

---

## Task 5: New Flow Benchmark Variant — `llm_from_scratch`

**File**: `ageom/flow_benchmark.py`

### 5a. Add `_LLMFromScratchMock` class

This mock simulates what an LLM would do if asked to identify all needed library
functions in a single response, without ageo-matcher infrastructure:

```python
class _LLMFromScratchMock:
    """Simulates a raw LLM identifying library functions from a goal prompt."""

    def __init__(self, case: FlowBenchmarkCase) -> None:
        self._case = case
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        # Return a JSON list of {name, type_signature} for each leaf.
        # In deterministic mode, return the correct answers.
        # This establishes the baseline ceiling; the stochastic variant
        # (Task 6) will introduce noise to model real LLM behavior.
        return json.dumps([
            {
                "name": leaf.declaration_name,
                "type_signature": leaf.type_signature,
                "description": leaf.description,
            }
            for leaf in self._case.leaves
        ])
```

### 5b. Add `_run_llm_from_scratch_case`

```python
async def _run_llm_from_scratch_case(
    case: FlowBenchmarkCase,
) -> FlowBenchmarkResult:
    started = time.perf_counter()
    llm = _LLMFromScratchMock(case)
    raw = await llm.complete(
        "Identify all library functions needed to implement this algorithm. "
        "Return a JSON array of {name, type_signature, description}.",
        case.prompt,
    )
    parsed = json.loads(raw)
    # Fuzzy match: for each returned function, check if its name matches
    # any expected declaration via token overlap.
    expected = {leaf.declaration_name for leaf in case.leaves}
    matched: set[str] = set()
    for entry in parsed:
        name = str(entry.get("name", ""))
        if name in expected:
            matched.add(name)
    latency_ms = (time.perf_counter() - started) * 1000.0
    total = len(case.leaves)
    n_matched = len(matched)
    return FlowBenchmarkResult(
        case_id=case.case_id,
        domain=case.domain,
        variant="llm_from_scratch",
        execution_path="llm_from_scratch",
        ok=n_matched == total,
        latency_ms=latency_ms,
        prompt_calls=llm.calls,
        matched_leaves=n_matched,
        total_leaves=total,
        node_count=0,
        leaf_coverage=n_matched / max(1, total),
        best_similarity=0.0,
        decomposition_depth=0,
        decomposition_leaf_count=0,
        decomposition_edge_count=0,
    )
```

### 5c. Register in `run_flow_benchmark` dispatch

Add to the dispatch block:

```python
if variant == "llm_from_scratch":
    results.append(await _run_llm_from_scratch_case(case))
    continue
```

Do NOT add to default variants yet. The deterministic mock always succeeds, which
is the ceiling. Task 6 (below) adds the stochastic version for real measurement.

### 5d. Add `_EXPECTED_FLOW_EXECUTION_PATHS` entry

**File**: `ageom/benchmark_validation.py` (line ~59)

```python
"llm_from_scratch": "llm_from_scratch",
```

---

## Task 6: Stochastic `llm_from_scratch` for Real Measurement

**File**: `ageom/flow_benchmark.py`

### 6a. Add `_NoisyLLMFromScratchMock`

```python
class _NoisyLLMFromScratchMock(_LLMFromScratchMock):
    """LLM-from-scratch mock with realistic error modes."""

    def __init__(
        self,
        case: FlowBenchmarkCase,
        *,
        seed: int | None = None,
        miss_leaf_prob: float = 0.2,    # prob of omitting a leaf
        hallucinate_prob: float = 0.15, # prob of adding a wrong function
        rename_prob: float = 0.1,       # prob of using a close-but-wrong name
    ) -> None:
        super().__init__(case)
        self._rng = random.Random(seed)
        self._miss_leaf_prob = miss_leaf_prob
        self._hallucinate_prob = hallucinate_prob
        self._rename_prob = rename_prob

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        results = []
        for leaf in self._case.leaves:
            if self._rng.random() < self._miss_leaf_prob:
                continue  # LLM missed this leaf
            name = leaf.declaration_name
            if self._rng.random() < self._rename_prob:
                name = name.replace(".", ".wrong_")  # close but wrong
            results.append({
                "name": name,
                "type_signature": leaf.type_signature,
                "description": leaf.description,
            })
        if self._rng.random() < self._hallucinate_prob:
            results.append({
                "name": "algorithms.hallucinated_function",
                "type_signature": "any -> any",
                "description": "This function does not exist.",
            })
        return json.dumps(results)
```

This models three real LLM failure modes: missing a required function, naming
it slightly wrong, and hallucinating a function that doesn't exist.

### 6b. Wire into `run_flow_benchmark`

When `noisy=True` and variant is `llm_from_scratch`, use
`_NoisyLLMFromScratchMock(case, seed=repeat_index)`.

---

## Task 7: Architect Prompt Benchmarks

**File**: `ageom/prompt_benchmark.py`

### 7a. Import Architect prompt templates

Locate the Architect prompt strings. They follow the same pattern as Hunter
prompts in `ageom/hunter/prompts.py`. Find them at:

```
ageom/architect/prompts.py  (or wherever STRATEGY_SYSTEM, DECOMPOSE_SYSTEM,
                              CRITIQUE_SYSTEM are defined)
```

Import `ARCHITECT_STRATEGY`, `ARCHITECT_DECOMPOSE`, `ARCHITECT_CRITIQUE` from
`ageom/llm_router.py` (prompt key constants).

### 7b. Add `_strategy_case`, `_decompose_case`, `_critique_case` factories

Follow the same pattern as `_score_case` etc. Each factory returns a
`PromptBenchmarkCase` with:

- Appropriate system/user prompts from the Architect prompt templates
- `expected` dict with a `kind` key for validation
- `baseline_system`/`baseline_user` for the direct comparison

**Strategy case** expected output:
```python
{"kind": "strategy_json", "required_keys": ["paradigm", "rationale"]}
```

**Decompose case** expected output:
```python
{"kind": "decompose_json", "required_keys": ["sub_nodes"], "min_sub_nodes": 2}
```

**Critique case** expected output:
```python
{"kind": "critique_json", "required_keys": ["approved", "reason"]}
```

### 7c. Add validation branches in `_validate_case_output` (line ~396)

```python
if kind == "strategy_json":
    parsed = extract_json(output)
    if not isinstance(parsed, dict):
        return "expected JSON object"
    required_keys = case.expected.get("required_keys", [])
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        return f"missing keys: {missing}"
    return ""

if kind == "decompose_json":
    parsed = extract_json(output)
    if not isinstance(parsed, dict):
        return "expected JSON object"
    sub_nodes = parsed.get("sub_nodes", [])
    if not isinstance(sub_nodes, list):
        return "expected sub_nodes to be a list"
    min_count = case.expected.get("min_sub_nodes", 1)
    if len(sub_nodes) < min_count:
        return f"expected at least {min_count} sub_nodes, got {len(sub_nodes)}"
    return ""

if kind == "critique_json":
    parsed = extract_json(output)
    if not isinstance(parsed, dict):
        return "expected JSON object"
    required_keys = case.expected.get("required_keys", [])
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        return f"missing keys: {missing}"
    return ""
```

### 7d. Add 4 cases to `default_prompt_benchmark_cases`

One per domain (dsp, graph, linear_algebra, strings) for each of the 3 Architect
prompt keys = 12 new cases. Start with strategy only (4 cases) and expand.

### 7e. Update `FixturePromptBenchmarkLLM` in `benchmark_validation.py`

Add deterministic responses for Architect prompt patterns. Match on system prompt
keywords like "paradigm" (strategy), "sub-nodes"/"sub_nodes" (decompose),
"critic"/"evaluate" (critique) — the same patterns `_FlowArchitectLLM` uses.

---

## Task 8: Update Validation Gate for New Variants

**File**: `ageom/benchmark_validation.py`

### 8a. Update `_REQUIRED_FLOW_BENCHMARK_VARIANTS` (line 58)

No change — keep `{"structured", "verified"}` as required. The new variants
(`llm_from_scratch`, `verified_refinement`) are comparison-only.

### 8b. Update `_EXPECTED_FLOW_EXECUTION_PATHS` (line 59)

Add entries for new variants:

```python
"llm_from_scratch": "llm_from_scratch",
"verified_refinement": "verified_orchestration_with_refinement",
```

### 8c. Add coverage monotonicity to `run_benchmark_validation` (line ~560)

After the existing `benchmark_passed` computation:

```python
flow_agg_map = {agg.variant: agg for agg in flow_aggregates}
_rapid_cov = getattr(flow_agg_map.get("rapid"), "avg_leaf_coverage", 0.0)
_struct_cov = getattr(flow_agg_map.get("structured"), "avg_leaf_coverage", 0.0)
_verif_cov = getattr(flow_agg_map.get("verified"), "avg_leaf_coverage", 0.0)
coverage_monotonic = _struct_cov >= _rapid_cov and _verif_cov >= _struct_cov
```

Add to summary dict:

```python
"coverage_monotonic": coverage_monotonic,
"coverage_by_variant": {
    "rapid": _rapid_cov,
    "structured": _struct_cov,
    "verified": _verif_cov,
},
```

Add `coverage_monotonic` to `benchmark_passed` conjunction.

---

## Task 9: Test Updates Summary

All test changes are additive. No existing assertions should be removed — only
relaxed where the new single-leaf case changes expected counts.

**`tests/test_flow_benchmark.py`**:
- Update case count assertion: `3` -> `7`
- Update domain set assertion to include `linear_algebra`, `strings`
- Relax `direct_baseline.failed_cases` and `rapid.failed_cases` (now >= 1 pass)
- Add `test_flow_benchmark_noisy_stability`
- Add `test_verified_refinement_recovers_from_initial_failure`

**`tests/test_benchmark_validation.py`**:
- Add assertion for `coverage_monotonic` in summary payload
- Update `flow_cases` count assertion: `3` -> `7`

**`tests/test_prompt_benchmark.py`** (if it exists):
- Add assertions for new Architect prompt cases

---

## Implementation Order

Execute in this order to keep tests green at each step:

1. **Task 1** (continuous metrics) — additive fields, no behavior change
2. **Task 2** (new cases) — update test assertions in same commit
3. **Task 8** (validation gate) — wire coverage monotonicity
4. **Task 3** (noisy mocks) — new code path behind `noisy=True` flag
5. **Task 4** (refinement case) — new variant, new test
6. **Task 5** (llm_from_scratch deterministic) — new variant, new test
7. **Task 6** (llm_from_scratch stochastic) — extends Task 5
8. **Task 7** (Architect prompts) — independent of flow changes

Each task is a single commit. Run `pytest tests/test_flow_benchmark.py
tests/test_benchmark_validation.py tests/test_prompt_benchmark.py` after each.
