# Benchmark Infrastructure Analysis (2026-03-08)

## Files Analyzed
- `ageom/flow_benchmark.py` - flow benchmark harness (3 cases, 4 variants)
- `ageom/prompt_benchmark.py` - prompt-key benchmark (12 cases, 3 prompt keys, 4 domains)
- `ageom/benchmark_validation.py` - release validation bundle
- `ageom/orchestrator.py` - verified mode feedback loop
- `tests/test_flow_benchmark.py`, `tests/test_e2e_flow_benchmark.py`
- `tests/test_prompt_benchmark.py`, `tests/test_e2e_prompt_benchmark.py`
- `tests/test_benchmark_validation.py`
- `build/prompt_benchmark_live_small.json` - live benchmark data

## Structural Issues Found

### 1. Mock LLMs make flow benchmarks tautological
- `_FlowArchitectLLM` returns canned JSON keyed on system prompt substrings
- `_BenchmarkHunterLLM` returns fixed arrays regardless of input
- `_LeafOracle` verifies by exact name match against pre-seeded expected map
- Result: structured/verified always pass, rapid/direct_baseline always fail
- Stability is trivially 1.0 -- no variance source exists

### 2. "direct_baseline" is not an LLM-from-scratch baseline
- It runs a single HunterAgent.find_match() on the raw prompt
- Uses the same mock LLM, same lexical index, same oracle
- It measures "Hunter without decomposition" not "LLM without ageo-matcher"
- The ROADMAP explicitly asks for proof that the package outperforms direct LLM coding

### 3. Prompt benchmarks have a live artifact but no live flow benchmarks
- `build/prompt_benchmark_live_small.json` shows real codex_shim and gemini_shim results
- No equivalent live flow benchmark results exist
- Prompt benchmark covers only Hunter prompt keys, not Architect or Synthesizer

### 4. No quality gradient metrics
- FlowBenchmarkResult tracks binary ok + matched_leaves/total_leaves
- No partial credit, no semantic similarity, no type-compatibility scoring
- No decomposition quality metrics (depth, leaf count, edge density)

### 5. Mode monotonicity is structurally guaranteed, not empirically validated
- Mocks are designed so structured/verified pass and rapid/direct_baseline fail
- `_MODE_RUNTIME_BUDGETS` checks config complexity, not outcome quality
- No test validates rapid < structured < verified on quality metrics

## Recommendations Summary
1. True LLM-from-scratch baseline needed (no Hunter, no CDG, just raw LLM generation)
2. Live flow benchmarks with real LLMs needed alongside deterministic validation
3. Semantic quality metrics: partial leaf coverage, type compatibility, decomposition quality
4. Stability measurement requires stochastic LLM mock or real providers
5. Mode progression validation needs continuous quality metrics, not binary pass/fail
