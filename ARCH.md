# Architecture

AGEO-Matcher implements **verified retrieval-augmented composition**: decompose a goal into typed sub-problems, match those sub-problems against a catalog of real library functions, verify that the matches actually work, and assemble the result.

The system is organized as a four-round Agentic Development Cycle, wrapped by an optional **Principal** meta-optimizer. All rounds follow the **deterministic-first, LLM-fallback** principle: every operation that can be handled by a regex, AST walk, embedding lookup, or type check is handled deterministically. LLMs are reserved for conceptual decomposition and ambiguous cases.

- **Round 0 (Ingester)**: Parses existing source code (Python, C++, Julia, Rust) into data-flow graphs, chunks them into macro-atoms (deterministic AST-based splitting when confident, LLM chunking otherwise), and generates `@register_atom` wrappers, state models, ghost witnesses, and FFI bindings. Produces `IngestionBundle`s that feed directly into the Synthesizer.
- **Round 1 (Architect)**: Decomposes a high-level goal into an atomic Conceptual Dependency Graph (CDG) via LangGraph, with PostgreSQL-backed persistence for checkpoint time-travel and forking.
- **Round 2 (Hunter)**: Grounds each atomic CDG leaf into a verified library function in Lean 4/Mathlib, Coq/Rocq, or Python. Candidate ranking uses embedding reranking (cosine similarity + type-token bonus) before falling back to LLM. Failure analysis uses regex-based pattern matching for known error classes before falling back to LLM.
- **Round 3 (Synthesizer)**: Assembles matched atoms into compilable skeleton files. Runs an optional ghost witness simulation pass (via `ageoa`) to catch structural mismatches before expensive compilation. Repair uses an expanded deterministic fix database (type coercions, common imports, syntax corrections) before LLM patching.
- **Principal (Meta-Optimizer)**: Wraps all four rounds in a NAS-style optimisation loop. Uses Optuna HPO, ghost-simulation early pruning, interval-arithmetic precision gradients, per-node credit assignment, and the Architect's checkpoint time-travel for coordinate descent over decomposition structure.

## Execution Modes

The pipeline supports graduated execution tiers so the overhead scales with the task's correctness requirements:

| Mode | What runs | Best for |
|------|-----------|----------|
| `rapid` | Direct Hunter match, no decomposition | Simple single-predicate lookups |
| `structured` | Architect decomposition + single-pass Hunter batch match | Standard tasks with clear decomposition |
| `single_agent` | Deterministic planner: direct match -> decompose -> batch match -> partial acceptance (>=70%) -> selective re-decomposition -> escalation | Most production tasks; avoids full orchestration overhead |
| `verified` | Full multi-round orchestration with Architect refinement loops | High-correctness tasks where every leaf must be verified |

Set via `--mode <mode>` or `SCIONA_EXECUTION_MODE` in `.env`.

## Design principle

**Deterministic -> Agentic -> Deterministic** sandwich (applies to all rounds).

Each prompt key in the system has a deterministic tool that tries to handle the request first. The LLM is only invoked when the deterministic tool's confidence is too low or the pattern is unrecognized. Current deterministic coverage:

| Phase | Prompt Key | Deterministic Tool | Fallback |
|-------|-----------|-------------------|----------|
| Ingester | `INGESTER_DECOMPOSE` | Control-flow decomposer (AST-based) | LLM chunking |
| Hunter | `HUNTER_SCORE` | Embedding reranker (cosine + type bonus) | LLM ranking |
| Hunter | `HUNTER_ANALYZE_FAILURE` | Regex failure analyzer | LLM analysis |
| Synthesizer | `SYNTHESIZER_REPAIR` | Classifier fix database (imports, coercions, syntax) | LLM patching |
| Orchestrator | refinement splits | Domain-specific split patterns (~20 domains) | LLM re-decomposition |

```
Existing Source Code (.py, .cpp, .jl, .rs)
  |
  v
[Smart Ingester]        deterministic + agentic -- tree-sitter / AST extraction,
  |                       LLM chunking, code generation, FFI binding emission
  v
 IngestionBundle (CDG + atoms + witnesses + state models)
  |
  v

High-Level Goal
  |
  v
[Architect Agent]       agentic -- LLM decomposes, critic approves/rejects (LangGraph)
  |
  v
 CDG (atomic leaves)
  |  validated handoff (type_signature + description required)
  v
[Semantic Indexer]      deterministic -- vector search, no LLM
  |
  v
[Retrieval Agent]       deterministic-first -- embedding reranker + regex failure analyzer,
  |                       LLM fallback for low-confidence ranking and unknown errors
  |
  v
[Verification Oracle]   deterministic -- compiler says yes or no
  |
  v
MatchResult
  |
  v
[Ghost Simulation]      deterministic -- abstract interpretation via ghost witnesses (optional)
  |
  v
[Assembler]             deterministic -- CDG + matches -> skeleton source code
  |
  v
[Compiler]              deterministic -- proof assistant says yes or no
  |
  v
[Repair Agent]          deterministic-first -- classifier fix DB, then LLM patching
  |
  v
SkeletonFile + VerificationCertificate
  |
  v
[Principal Evaluator]   deterministic -- subprocess benchmark + trace parsing
  |
  v
[Credit Assigner]       deterministic -- per-node gradient computation
  |
  v
[Optuna HPO]            deterministic -- early pruning, parameter importance
  |
  v
[Time-Travel Update]    deterministic -- fork Architect checkpoint, inject constraint
  |
  v
 (loop back to Architect decomposition)
```

LLMs are confined to agentic layers. The indexer, oracle, ghost simulation, assembler, handoff validation, evaluator, credit assigner, and HPO are pure functions of their inputs -- no hallucination surface.

## Components

### Shared types (`sciona/types.py`)

Five frozen dataclasses form the vocabulary shared across all components:

| Type | Role |
|------|------|
| `Declaration` | A theorem/definition extracted from a proof library (name, type signature, docstring, prover) |
| `PDGNode` | A predicate to be grounded (statement, informal description, target prover) |
| `CandidateMatch` | A Declaration paired with a retrieval score and method |
| `VerificationResult` | Compiler output for a single candidate (verified/not, error message) |
| `MatchResult` | Final result for a PDG node (best verified match + all attempts) |

### Protocols (`sciona/protocols.py`)

Four `typing.Protocol` interfaces define the contracts:

- **`SemanticIndex`** -- `search_by_embedding()`, `search_by_type()`, `get_declaration()`
- **`ProofEnvironment`** -- `check_term()`, `check_proof()`, `get_type()`, `close()`
- **`VerificationOracle`** -- `verify_candidate()`, `verify_candidates()`
- **`RetrievalAgent`** -- `find_match()`

All component implementations are structural subtypes of these protocols. No inheritance required.

### Smart Ingester (`sciona/ingester/`) -- Round 0

Converts existing source code into stateless atom graphs compatible with the AGEO framework. Supports Python, C++, Julia, and Rust.

#### Multi-language extraction

The ingester dispatches by file extension via `_get_extractor()`:

| Extension | Extractor | Parser |
|-----------|-----------|--------|
| `.py` | `PythonASTExtractor` | Python `ast` module |
| `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp` | `TreeSitterExtractor(CPP)` | tree-sitter |
| `.jl` | `TreeSitterExtractor(JULIA)` | tree-sitter |
| `.rs` | `TreeSitterExtractor(RUST)` | tree-sitter |

All extractors implement the `BaseExtractor` protocol (`extract_class()`, `extract_procedural()`) and produce the same `RawDataFlowGraph` schema, so downstream phases are language-agnostic.

#### Pipeline

```
Source file (.py / .cpp / .jl / .rs)
  |
  v
Phase 1: Extraction          deterministic -- AST or tree-sitter parsing
  |  RawDataFlowGraph (methods, fields, reads/writes, init chain, call graph)
  v
Phase 2: Semantic Chunking   deterministic-first -- AST control-flow decomposer,
  |                             then LLM chunking if confidence is low
  |  ValidatedMacroPlan (macro_atoms, state_models, edges)
  v
Phase 3: Code Generation     deterministic -- @register_atom wrappers, state models,
  |                             ghost witnesses, CDG, FFI bindings
  v
Verification Loops            mypy type checking + ghost simulation repair
  |
  v
IngestionBundle (CDG + generated_atoms + generated_witnesses + match_results)
```

#### FFI binding generation

For non-Python sources, the emitter appends FFI stubs so generated atoms can call into the original implementation:

- **C++ / Rust**: `ctypes.CDLL` + function prototype stubs
- **Julia**: `juliacall.Main` + `jl.eval()` call stubs

#### Key modules

| Module | Role |
|--------|------|
| `models.py` | `RawDataFlowGraph`, `MethodFact` (with `decorators`, `is_external`), `MacroAtomSpec`, `IngestionBundle` |
| `base_extractor.py` | `BaseExtractor` protocol, `SourceLanguage` enum, `EXTENSION_MAP` |
| `python_extractor.py` | `PythonASTExtractor` -- adapter wrapping existing `ast`-based extraction |
| `treesitter_extractor.py` | `TreeSitterExtractor` -- C++/Julia/Rust class/struct extraction via tree-sitter |
| `extractor.py` | Phase 1 Python implementation: `_SelfAccessVisitor`, config branches, SSA edges |
| `control_flow_decomposer.py` | Deterministic AST-based function splitting at structural boundaries |
| `chunker.py` | Phase 2: semantic chunking (deterministic-first, then LLM) |
| `emitter.py` | Phase 3: `@register_atom` wrappers (with decorator passthrough), state models, ghost witnesses, CDG construction |
| `ffi_emitter.py` | FFI binding generation for C++, Julia, and Rust |
| `graph.py` | `IngesterAgent` state machine, `_get_extractor()` dispatch, verification/repair loops |

### Conceptual Dependency Agent (`sciona/architect/`) -- Round 1

Decomposes a high-level goal into an atomic Conceptual Dependency Graph (CDG) using an iterative LangGraph cycle.

#### Graph topology

```
select_strategy ──> decompose_node ──> critique ──> advance_node ──> END
                         ^                |               |
                         |          prepare_retry          |
                         |                |                |
                         +────────────────+        (more pending)
                                                          |
                                                   decompose_node (loop)
```

#### Key modules

| Module | Role |
|--------|------|
| `models.py` | `AlgorithmicNode`, `DependencyEdge`, `NodeStatus`, `ConceptType`, `AlgorithmicPrimitive` |
| `catalog.py` | `PrimitiveCatalog` -- in-memory store of known atomic operations (CLRS-30, coq-100-theorems) |
| `embedder.py` | `SkillIndex` -- FAISS index over primitives for atomic-match detection |
| `skeletons.py` | Pre-built graph templates per paradigm (D&C, DP, greedy, ...) |
| `nodes.py` | LangGraph node functions: `select_strategy`, `decompose_node`, `critique_decomposition`, `advance_node`, `prepare_retry` |
| `state.py` | `DecompositionState` (TypedDict with custom `_merge_nodes` reducer) + `DecompositionDeps` |
| `graph.py` | `build_graph()` assembles the `StateGraph`; `DecompositionAgent` wraps it with checkpointing, time-travel (`get_state`, `get_state_history`, `fork`), and thread management |
| `checkpointer.py` | `create_checkpointer()` async context manager -- tries `AsyncPostgresSaver`, falls back to `MemorySaver` |
| `handoff.py` | `CDGExport` model, `validate_handoff()`, `to_pdg_nodes()` (Round 1 -> Round 2 bridge), `HandoffValidationError` |

#### State persistence

`DecompositionAgent` accepts an optional `BaseCheckpointSaver`. When provided, every graph step is checkpointed:

- **`decompose(goal, thread_id=...)`** -- runs the full cycle under a thread ID (auto-generated 32-char hex if omitted). The thread ID is included in `CDGExport.metadata`.
- **`get_state(thread_id)`** -- retrieves the latest checkpoint.
- **`get_state_history(thread_id)`** -- returns all checkpoints (newest first).
- **`fork(source_thread_id, checkpoint_id)`** -- creates an independent thread from any historical checkpoint ("time travel").

`create_checkpointer(postgres_uri)` never raises: PostgreSQL unavailable silently falls back to in-memory with a log warning.

#### Handoff validation

Before Round 2 receives PDG nodes, `validate_handoff()` checks every atomic leaf for:
- Non-empty `description`
- Non-empty `type_signature`
- No non-atomic leaves remaining

`to_pdg_nodes(cdg, strict=True)` raises `HandoffValidationError` on any issue. Pass `strict=False` to skip validation.

### Semantic Indexer (`sciona/indexer/`)

Indexes formal declarations into a searchable vector store.

```
Library Source ──> Embedder ──> FAISS Store
(lean-explore       (UniXcoder)    (IndexFlatIP)
 or coqpyt)
```

- **`UniXcoderEmbedder`** -- Embeds `"{name} : {type_sig}\n{docstring}"` into L2-normalized 768-dim vectors using `microsoft/unixcoder-base`. Batch support for indexing.
- **`FAISSStore`** -- `IndexFlatIP` wrapped in `IndexIDMap`. Inner product on normalized vectors = cosine similarity. Persists to disk as `index.faiss` + `declarations.pkl` + `meta.json`.
- **`LeanDeclarationSource`** -- Enumerates Mathlib via `lean-explore`'s local Service. Also exposes hybrid type search.
- **`CoqDeclarationSource`** -- Parses `.v` files via `coqpyt`, extracting Theorem/Lemma/Definition declarations.
- **`SemanticIndexImpl`** -- Implements the `SemanticIndex` protocol. Embedding search goes through FAISS; type search delegates to lean-explore for Lean, falls back to embedding for Coq.

### Verification Oracle (`sciona/judge/`)

Compiler-based proof checking. No approximation -- the compiler is the ground truth.

- **`LeanEnvironment`** -- Wraps `lean-interact`'s `LeanServer` with a `TempRequireProject` configured for Mathlib. Checks terms via `example : {type} := {term}` and proofs via `theorem _check : {stmt} := by {body}`. All calls are `async` (sync REPL wrapped in `asyncio.to_thread`).
- **`CoqEnvironment`** -- Writes to temporary `.v` files and compiles via `coqpyt`. Same interface: `Definition _check` / `Lemma _check`.
- **`PythonEnvironment`** -- Import-based verification (default): checks that the function is importable, callable, and has compatible arity via `inspect.signature`. Optional mypy --strict mode via `verify_mode="mypy"`.
- **`VerificationOracleImpl`** -- Routes to the correct environment based on `pdg_node.prover`. `verify_candidates()` short-circuits on first verified match.

Verification strategy: for Lean/Coq, attempt direct type unification -- `@CandidateName` as a term for the PDG node's statement. For Python, verify the function exists and is callable with the expected signature. If the check passes, the match is accepted.

### Retrieval Agent (`sciona/hunter/`)

An LLM-driven search loop implemented as a typed state machine with `pydantic-graph`.

#### Graph topology

```
InitialSearch ──> RankCandidates ──> VerifyTopK ──> End[MatchResult]
                                         |
                                    ReformulateQuery
                                         |
                                    InitialSearch  (loop)
```

#### Nodes

| Node | What it does | Transitions |
|------|-------------|-------------|
| `InitialSearch` | Vector search + type search, merge/dedup candidates | `RankCandidates` or `End` (no candidates) |
| `RankCandidates` | Embedding reranker (cosine similarity + type-token bonus); falls back to LLM when top-2 margin < 0.05 | `VerifyTopK` |
| `VerifyTopK` | Sends top-K to Verification Oracle | `End` (verified) or `ReformulateQuery` or `End` (budget exhausted) |
| `ReformulateQuery` | Deterministic failure analyzer for known error patterns; falls back to LLM for unrecognized errors | `InitialSearch` |

#### State and dependencies

- **`HunterState`** -- Mutable state threaded through the graph: accumulated candidates, verification results, queries tried, compiler feedback, iteration counter.
- **`HunterDeps`** -- Injected dependencies: `SemanticIndex`, `VerificationOracle`, `LLMClient`.

#### Why pydantic-graph

- **Type-safe transitions** -- The return type of each node's `run()` method declares which nodes it can transition to. Mypy verifies the graph is well-formed at type-check time.
- **Built-in persistence** -- State can be serialized for pause/resume.
- **Observability** -- Tracing/spans and Mermaid diagram generation come for free.

### LLM integration (`sciona/hunter/llm.py`)

`LLMClient` is a Protocol with a single method: `async complete(system, user) -> str`.
Built-in implementations:
- `ClaudeLLMClient` via `anthropic.AsyncAnthropic`
- `CodexLLMClient` via `openai.AsyncOpenAI` (Codex-compatible models)
- `LlamaCppLLMClient` via OpenAI-compatible llama.cpp endpoint (`extra_body.grammar`)

`create_llm_client(...)` selects the provider (`anthropic`, `codex`, or `llama_cpp`) from config/CLI.

Three prompt templates drive the agent's reasoning, each with a deterministic tool that intercepts when possible:
- **`REFORMULATE_QUERY`** -- Given failed queries + compiler errors, generate new search terms
- **`SCORE_CANDIDATES`** -- Embedding reranker handles most cases; LLM called only on low-confidence rankings
- **`ANALYZE_FAILURE`** -- Deterministic regex analyzer handles known error classes (unknown identifier, arity mismatch, type incompatibility, syntax errors); LLM called only for unrecognized patterns

### Synthesizer (`sciona/synthesizer/`) -- Round 3

Assembles CDG + MatchResults into compilable skeleton files, optionally pre-validated by ghost witness simulation.

#### Pipeline (`pipeline.py`)

`assemble_and_check()` orchestrates the full flow:

1. **Ghost simulation** (optional) -- converts CDG atomic leaves to `SimNode`s and runs `simulate_graph()` from `ageoa.ghost`. Catches domain mismatches (e.g., feeding frequency-domain data into an FFT that expects time-domain), shape errors, and type violations *before* expensive compilation.
2. **Assembly** -- topologically sorts CDG nodes, fuses each atomic leaf with its `MatchResult`, and emits source code (Lean 4, Coq, or Python).
3. **Compilation** -- sends the skeleton through the `ProofEnvironment` compiler.

The ghost simulation is best-effort: if `ageoa` is not installed or no atoms have registered witnesses, it is silently skipped. Non-DSP atoms (those without witnesses) are also skipped. Results are attached to `skeleton.metadata["ghost_simulation"]`.

#### Ghost Witness integration (`ghost_sim.py`)

The ghost simulation pass bridges the two repositories:

```
ageoa (ageo-atoms)                    sciona (sciona)
  ghost/                                synthesizer/
    registry.py  REGISTRY  <-------->   ghost_sim.py  run_ghost_simulation()
    witnesses.py  witness_fft, ...        |
    simulator.py  simulate_graph()        +-- converts CDGExport to SimNodes
    abstract.py   AbstractSignal          +-- builds initial abstract state from IOSpec
                                          +-- calls simulate_graph()
  numpy/fft.py   @register_atom(witness_fft)
  scipy/signal.py @register_atom(witness_butter)
  ...
```

16 DSP atoms are auto-registered when their modules are imported:

| Module | Atoms |
|--------|-------|
| `ageoa.numpy.fft` | `fft`, `ifft`, `rfft`, `irfft` |
| `ageoa.scipy.fft` | `dct`, `idct` |
| `ageoa.scipy.signal` | `butter`, `cheby1`, `cheby2`, `firwin`, `sosfilt`, `lfilter`, `freqz` |
| `ageoa.scipy.sparse_graph` | `graph_laplacian`, `graph_fourier_transform`, `heat_kernel_diffusion` |

#### Key modules

| Module | Role |
|--------|------|
| `pipeline.py` | `assemble_and_check()` -- ghost sim + assembly + compilation |
| `ghost_sim.py` | `run_ghost_simulation()` -- CDG to SimNode conversion, abstract state construction, `GhostSimReport` |
| `assembler.py` | `Assembler` -- CDG + matches -> `SkeletonFile` with Lean 4 / Coq / Python source code |
| `compiler.py` | `SkeletonCompiler` -- wraps `ProofEnvironment` for whole-file compilation |
| `toposort.py` | `toposort_nodes()` -- Kahn's algorithm for dependency ordering |
| `contracts.py` | `ContractGenerator` -- generates icontract decorators, recognizes DSP constraint patterns |
| `models.py` | `AssemblyUnit`, `GlueEdge`, `SkeletonFile`, `AssemblyResult`, `SynthesisResult`, `VerificationCertificate`, `ExportBundle` |
| `classifier.py` | Regex error classifier + deterministic fix database (imports, type coercions, syntax) |
| `repair.py` | Compile-analyze-patch loop (deterministic fixes first, then LLM for remaining errors) |
| `extractor.py` | FFI export and verification certificate generation |

### Principal Meta-Optimizer (`sciona/principal/`)

Wraps the four-round pipeline in a NAS-style optimisation loop: Forward → Evaluate → Backward → Update.

#### Graph topology

```
seed ──> forward ──> evaluate ──> gradients ──> time_travel ──> forward (loop)
                        |              |              |
                   (pruned early)   (no grads)     (done/budget)
                        |              |              |
                        v              v              v
                    time_travel       END            END
```

#### Key modules

| Module | Role |
|--------|------|
| `models.py` | `OptimizationMetric` (LATENCY, MEMORY, PRECISION, FLOP_COUNT), `NodeTelemetry`, `BenchmarkResult`, `NodeGradient` |
| `evaluator.py` | `ExecutionSandbox` -- runs instrumented artifacts as subprocesses, parses `trace.jsonl` into `NodeTelemetry`, computes scalar loss |
| `backprop.py` | `CreditAssigner` -- per-node optimisation gradients (latency %, memory %, precision via ghost-sim intervals or telemetry error expansion) |
| `hpo.py` | `OptunaManager` -- wraps `optuna.Study`, early pruning via `GhostSimReport`, fANOVA parameter importance |
| `graph.py` | `PrincipalState`, `PrincipalDeps`, 5 async node functions, 3 routing functions, `build_principal_graph()` |

#### Optimisation loop

1. **seed_population** -- Optuna suggests paradigm parameters; Architect decomposes goal under a fresh thread.
2. **execute_forward** -- Ghost simulation for early pruning (raises `TrialPrunedEarly` on structural failure or inf/NaN error bounds). If not pruned, delegates to `synthesize_fn` for the full synthesis pipeline.
3. **evaluate_run** -- `ExecutionSandbox` runs the instrumented artifact, parses trace telemetry, computes global loss.
4. **compute_gradients** -- `CreditAssigner` identifies the top bottleneck node by metric-specific credit assignment.
5. **time_travel_update** -- Walks the Architect's checkpoint history, finds the checkpoint just before the bottleneck node was created, forks a new thread, injects a CONSTRAINT describing the bottleneck, and re-decomposes.

#### Early pruning

`OptunaManager.check_early_prune(ghost_report)` aborts trials when:
- Ghost simulation ran and failed (structural mismatch)
- Any node's precision gradient is infinite or NaN

This avoids expensive compilation and benchmarking for structurally broken trials.

#### Precision gradients

Interval arithmetic in `ghost_sim.py` computes per-node error expansion (output interval width − input interval width) using known error factors for 18 common atoms (FFT, SVD, Cholesky, etc.). These feed into `CreditAssigner._gradient_precision()` as the primary signal, with telemetry `error_expansion` as fallback.

#### Assembly instrumentation

The `Assembler` supports a `with_telemetry=True` flag that wraps each atomic call in a `_sciona_probe()` helper emitting `trace.jsonl` records with `node_id`, `execution_time_ms`, `peak_memory_bytes`, and `error_expansion`. The `ExecutionSandbox` parses these traces after subprocess execution.

## Telemetry and Dashboard

### Pipeline telemetry (`sciona/telemetry.py`)

All commands emit structured pipeline events (``PipelineEvent``) and maintain per-run snapshots (``RunSnapshot``) tracking stage progress, prompt dispatch counters, and metadata. The module keeps:

- **EventLog** -- append-only structured events with JSONL export and live SSE subscribers.
- **TelemetryRegistry** -- tracks active/past runs, prompt dispatch state, and stage heartbeats. Snapshots are persisted as `run_{id}.json` files for cross-process dashboard reads.

Key public APIs: `log_event()`, `start_run()`, `finish_run()`, `update_stage()`, `start_prompt_dispatch()`, `finish_prompt_dispatch()`, `telemetry_scope()`, `telemetry_stage()`.

### Postgres telemetry persistence (`sciona/telemetry_store.py`)

When `SCIONA_POSTGRES_URI` is set and `SCIONA_TELEMETRY_BACKEND` is not `file`, telemetry is durably persisted to PostgreSQL alongside the existing Architect checkpoints and shared context data. The in-memory hot path for live SSE streaming is preserved; Postgres is the durable store for cross-run queries.

| Component | Role |
|-----------|------|
| `PostgresTelemetryStore` | Async Postgres persistence with `psycopg_pool.AsyncConnectionPool` (min 1, max 4 connections). Creates `pipeline_runs` and `pipeline_events` tables with indexes. Provides `upsert_run()`, `insert_events()`, `list_runs()`, `list_events()` with server-side filtering and pagination |
| `TelemetryDrain` | Bridges sync producers to async Postgres writes. Buffers events in a bounded deque (max 10,000) and run snapshots with last-writer-wins semantics. A background asyncio task flushes every 500ms. Write failures are silently caught to never block the pipeline |

**Table schemas:**

- `pipeline_runs` -- one row per run, upserted on every snapshot update. Fields mirror `RunSnapshot`: status, stage progress, prompt counters, metadata (JSONB). Indexed on `(status, started_at DESC)`.
- `pipeline_events` -- append-only event log. Fields mirror `PipelineEvent`: timestamp, round, phase, event_type, payload (JSONB), duration_ms, prompt_key, provider, model. Indexed on `(run_id, event_type)` and `(event_type, timestamp)`.

**Wiring:** Each async command entry point (`run_cmds`, `decompose_cmds`, `match_cmds`, `benchmark_cmds`) and the FastAPI lifespan create the store, start the drain, and shut both down on exit. The drain hooks are injected into `EventLog.append()` and `TelemetryRegistry._persist_locked()` via module-level globals set by `configure_postgres_telemetry()`.

### Dashboard endpoints

The visualizer API (`sciona/visualizer_api.py`) exposes a telemetry dashboard at `/api/dashboard/`:

| Endpoint | Description |
|----------|-------------|
| `GET /api/dashboard/runs` | List runs with stale/hang annotations. Params: `limit`, `state` (all\|running\|completed\|failed) |
| `GET /api/dashboard/runs/{run_id}` | Single run snapshot with derived summaries |
| `GET /api/dashboard/latest` | Most recently updated run |
| `GET /api/dashboard/runs/{run_id}/events` | Paginated events with server-side filters: `phase`, `event_type`, `prompt_key`, `round`, `has_error` |
| `GET /api/dashboard/runs/{run_id}/stream` | SSE endpoint for live event streaming (in-memory only, not affected by Postgres) |
| `GET /api/dashboard/runs/{run_id}/coverage` | Deterministic vs LLM fallback coverage per prompt key |
| `GET /api/dashboard/runs/{run_id}/errors` | Structured error list with retry grouping |

All query endpoints try Postgres first and fall back to in-memory/file when unavailable. The SSE streaming endpoint always uses the in-memory event log for minimal latency.

### CLI telemetry commands

```
sciona telemetry list [--limit N] [--state STATE]   # List recent runs
sciona telemetry show <run_id>                       # Show run details as JSON
```

Both commands try Postgres first (via `asyncio.run()`) and fall back to file-based persistence.

## Configuration

`AgeomConfig` (pydantic-settings) reads from `.env` with prefix `SCIONA_`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SCIONA_INDEX_DIR` | `data/index` | FAISS index location |
| `SCIONA_LLM_PROVIDER` | `anthropic` | Global LLM provider (`anthropic`, `codex`, or `llama_cpp`) |
| `SCIONA_ANTHROPIC_API_KEY` | | Claude API key |
| `SCIONA_OPENAI_API_KEY` | | OpenAI API key for Codex |
| `SCIONA_OPENAI_BASE_URL` | *(empty)* | Optional OpenAI-compatible endpoint |
| `SCIONA_LLM_MODEL` | `claude-sonnet-4-5-20250929` | LLM model for Hunter |
| `SCIONA_LLAMA_CPP_BASE_URL` | `http://127.0.0.1:8080/v1` | Local llama.cpp endpoint |
| `SCIONA_HUNTER_LLM_PROVIDER` | `llama_cpp` | Hunter-specific provider (defaults local) |
| `SCIONA_HUNTER_LLM_MODEL` | `llama-3.1-8b-instruct` | Hunter local model |
| `SCIONA_HUNTER_MODE` | `speculative_local` | Hunter loop mode |
| `SCIONA_HUNTER_USE_GBNF` | `true` | Grammar-constrained decoding in Hunter |
| `SCIONA_EMBEDDING_MODEL` | `microsoft/unixcoder-base` | Embedding model |
| `SCIONA_LEAN_TOOLCHAIN` | `leanprover/lean4:v4.14.0` | Lean version |
| `SCIONA_HUNTER_MAX_ITERATIONS` | `5` | Max search-verify-refine loops |
| `SCIONA_HUNTER_TOP_K_VERIFY` | `3` | Candidates sent to Oracle per iteration |
| `SCIONA_HUNTER_SEARCH_K` | `20` | Candidates retrieved per search |
| `SCIONA_POSTGRES_URI` | *(empty)* | PostgreSQL URI for checkpoint, shared context, and telemetry persistence |
| `SCIONA_TELEMETRY_BACKEND` | `auto` | Telemetry storage backend: `auto` (use Postgres when URI is set), `postgres`, or `file` |
| `SCIONA_TELEMETRY_RUNS_DIR` | `output/telemetry_runs` | Directory for file-based run snapshot persistence |
| `SCIONA_TELEMETRY_STALE_SECONDS` | `120` | Seconds before a running stage is flagged as stale in the dashboard |
| `SCIONA_ARCHITECT_MAX_DEPTH` | `8` | Max CDG decomposition depth |
| `SCIONA_ARCHITECT_LLM_PROVIDER` | *(empty)* | Optional Round 1 provider override (falls back to `SCIONA_LLM_PROVIDER`) |
| `SCIONA_ARCHITECT_LLM_MODEL` | `claude-sonnet-4-5-20250929` | LLM model for Architect |
| `SCIONA_SKILL_INDEX_DIR` | `data/skill_index` | Skill catalog and index location |

## Data flow

### Round 1: Decomposition

```
1. User provides: goal = "Implement merge sort"

2. select_strategy:
   - LLM picks paradigm (divide_and_conquer)
   - Instantiates skeleton template (split / recurse / merge)

3. decompose_node (iterative, per pending node):
   - LLM decomposes into sub-nodes
   - Catalog match -> mark ATOMIC, otherwise PENDING

4. critique:
   - Depth check, IO consistency, LLM review
   - Reject -> prepare_retry -> decompose_node (up to 3x)
   - Approve -> advance_node

5. advance_node:
   - Pop next pending node, or END if none remain

6. Result: CDGExport { nodes, edges, metadata: { thread_id, ... } }
```

Each step is checkpointed. Use `sciona history <thread-id>` to inspect, or `agent.fork()` to branch from any checkpoint.

### Round 2: Matching

```
1. Handoff: to_pdg_nodes(cdg) validates and converts atomic leaves to PDGNodes

2. InitialSearch:
   - Embeds statement with UniXcoder
   - Searches FAISS index (top 20)
   - Type-searches via lean-explore (top 20)
   - Merges and deduplicates

3. RankCandidates:
   - LLM receives candidate list
   - Returns ranked indices [3, 0, 7, ...]
   - Candidates reordered

4. VerifyTopK:
   - Top 3 candidates sent to LeanEnvironment
   - Each: `example : forall n m, n + m = m + n := @Nat.add_comm`
   - Lean compiler: OK -> verified match

5. Result: MatchResult { verified_match: Nat.add_comm, verified: true }
```

If step 4 fails for all candidates, `ReformulateQuery` asks the LLM to analyze the compiler errors and generate new search queries, then loops back to step 2.

### Round 3: Synthesis

```
1. Ghost simulation (optional):
   - Convert atomic leaves to SimNodes using matched atom names
   - Build initial abstract state from IOSpec metadata
   - Run simulate_graph() -- propagates shape/dtype/domain
   - If PlanError: log warning, attach to metadata, continue

2. Assembly:
   - Topological sort (Kahn's algorithm)
   - For each atomic leaf: fuse with MatchResult -> AssemblyUnit
   - Emit source code: Lean 4 (sorry stubs), Coq (Admitted), or Python (NotImplementedError)

3. Compilation:
   - Send skeleton source to ProofEnvironment
   - AssemblyResult { compiled_ok, feedback }

4. Repair (if compilation fails):
   - LLM analyzes compiler errors
   - Generates patches for sorry/Admitted stubs
   - Iterate up to max_iterations

5. Result: SkeletonFile + VerificationCertificate
```

### Principal: Optimisation

```
1. seed_population:
   - Optuna suggests trial parameters
   - Architect decomposes goal under new thread_id

2. execute_forward:
   - Ghost simulation -> GhostSimReport
   - OptunaManager.check_early_prune() -- abort on structural failure / inf bounds
   - If not pruned: synthesize_fn(cdg, matches) -> ExportBundle

3. evaluate_run:
   - ExecutionSandbox runs instrumented artifact (subprocess)
   - Parses trace.jsonl -> NodeTelemetry per node
   - _compute_loss(telemetry, metric) -> global_loss
   - Tracks best_loss across trials

4. compute_gradients:
   - CreditAssigner.compute_gradients(cdg, benchmark, ghost_report, metric)
   - LATENCY/FLOP_COUNT: % of total execution time per node
   - MEMORY: % of total peak memory per node
   - PRECISION: ghost-sim interval gradients (primary) + error_expansion (fallback)
   - Returns sorted NodeGradient[] -- top = bottleneck

5. time_travel_update:
   - Walk Architect checkpoint history
   - Find checkpoint before bottleneck node was created
   - architect.fork(thread_id, checkpoint_id) -> new thread
   - Inject CONSTRAINT into forked goal
   - Re-decompose -> new CDG
   - Loop back to step 2
```

## Dependency graph

```
sciona/types.py, protocols.py, config.py    (no external deps beyond pydantic)
         |
    +----+----+----------+-----------+------------------+
    |         |          |           |                  |
 indexer/   judge/    architect/  ingester/    telemetry.py + telemetry_store.py
    |         |          |           |          (psycopg_pool, asyncio drain)
    +----+----+          |           |
         |               |           |
      hunter/            |           |         (pydantic-graph, anthropic, openai)
         |               |           |
         +-------+-------+-----+----+
                 |              |
           synthesizer/        |               (ageoa [optional])
                 |             |
           principal/          |               (optuna)
                 |             |
               cli.py          |
                               |
                          IngestionBundle
                     (CDG + atoms + witnesses)
```

```
                 ageoa (ageo-atoms)
                   |
        ghost/     |     numpy/, scipy/
     abstract.py   |     fft.py, signal.py, ...
     registry.py   |     @register_atom(witness_xxx)
     simulator.py  |
     witnesses.py  |
                   |
            [optional import]
                   |
           sciona/synthesizer/ghost_sim.py
           sciona/ingester/emitter.py
```

The indexer and judge are fully independent of each other. The hunter depends on both (through their protocol interfaces, not concrete classes). The architect is independent of the indexer/judge/hunter -- its output (CDGExport) is the input to the hunter via the handoff bridge. The ingester is independent of the hunter/indexer/judge -- it produces `IngestionBundle`s (CDG + generated source + match results) that the Synthesizer can consume directly. For non-Python sources, the ingester uses tree-sitter for extraction and generates FFI bindings (ctypes for C++/Rust, juliacall for Julia). The synthesizer depends on the hunter (MatchResult) and architect (CDGExport), and optionally on `ageoa` for ghost witness simulation. If `ageoa` is not installed, the synthesizer works normally without the simulation pass. The principal depends on the architect (for time-travel forking), synthesizer (ghost simulation + assembly), and optuna for HPO. It is the outermost layer: all other components are unaware of the principal's existence.
