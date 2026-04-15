# Architecture

AGEO-Matcher implements **verified retrieval-augmented composition**: decompose a goal into typed sub-problems, match those sub-problems against a catalog of real library functions, verify that the matches actually work, and assemble the result.

The system is organized as a four-round Agentic Development Cycle, wrapped by an optional **Principal** meta-optimizer. All rounds follow the **deterministic-first, LLM-fallback** principle: every operation that can be handled by a regex, AST walk, embedding lookup, or type check is handled deterministically. LLMs are reserved for conceptual decomposition and ambiguous cases.

- **Round 0 (Ingester)**: Parses existing source code (Python, C++, Julia, Rust) into data-flow graphs, chunks them into macro-atoms (deterministic AST-based splitting when confident, LLM chunking otherwise), and generates `@register_atom` wrappers, state models, ghost witnesses, and FFI bindings. Produces `IngestionBundle`s that feed directly into the Synthesizer.
- **Round 1 (Architect)**: Decomposes a high-level goal into an atomic Conceptual Dependency Graph (CDG) via LangGraph, with PostgreSQL-backed persistence for checkpoint time-travel and forking.
- **Round 2 (Hunter)**: Grounds each atomic CDG leaf into a verified library function in Lean 4/Mathlib, Coq/Rocq, or Python. Candidate ranking uses embedding reranking (cosine similarity + type-token bonus) before falling back to LLM. Failure analysis uses regex-based pattern matching for known error classes before falling back to LLM.
- **Round 3 (Synthesizer)**: Assembles matched atoms into compilable skeleton files. Runs an optional ghost witness simulation pass via `sciona.atoms` to catch structural mismatches before expensive compilation. Repair uses an expanded deterministic fix database (type coercions, common imports, syntax corrections) before LLM patching.
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

Decomposes a high-level goal into an atomic Conceptual Dependency Graph (CDG)
using an iterative LangGraph cycle, but the live enrichment path is no longer
just "LLM decompose, then bind leaves". The current strategy mixes:

- **Skeleton bootstrap** for the root goal (`select_strategy`)
- **Deterministic decomposition** for recognized local structures
- **Primitive / template / skeleton proposal surfacing** for node enrichment
- **Cross-family refinement retrieval** when verified exemplars match by IO and topology
- **Guarded live skeleton insertion** only when a higher-complexity skeleton clears a
  conservative margin over simpler alternatives

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

#### Node enrichment strategy

`decompose_node()` now collects and ranks three candidate classes before it
falls back to the LLM decomposition prompt:

1. **Primitive proposals** from the catalog / lexical / skill retrieval
2. **Template proposals** from verified exemplar retrieval
3. **Skeleton proposals** from a bounded allowlist of named skeletons

The current live acceptance policy is intentionally conservative:

- templates can still short-circuit on high confidence
- skeletons are only live candidates if they are the top-ranked proposal and
  clear a fixed positive margin over the best lower-complexity alternative
- otherwise the system keeps the existing template or LLM path

This gives the Architect an explicit structural enrichment surface without
letting larger subgraphs win cheaply.

#### Cross-family behavior

Cross-family reuse is now first-class in the Architect:

- **Catalog binding** keeps same-family as a prior, not a gate
- **Template retrieval** first searches same-family verified exemplars, then
  falls back to cross-family exemplar retrieval keyed by IO shape and topology
- **Deterministic decomposition** can emit skeleton-backed subgraphs for a node
  whose local semantics differ from the family that seeded the parent graph
- **Skeleton proposals** can be cross-family, but only within a tight
  allowlist and only under the complexity-margin guard

#### Key modules

| Module | Role |
|--------|------|
| `models.py` | `AlgorithmicNode`, `DependencyEdge`, `NodeStatus`, `ConceptType`, `AlgorithmicPrimitive` |
| `catalog.py` | `PrimitiveCatalog`; cross-family primitive ranking stays eligible with same-family priors |
| `embedder.py` | `SkillIndex` -- FAISS index over primitives for atomic-match detection |
| `skeletons.py` | Pre-built graph templates per paradigm and named variants |
| `proposal_models.py` | Passive `primitive`, `template`, and `skeleton` enrichment proposal models |
| `proposal_ranking.py` | Conservative unified ranking with explicit complexity penalties |
| `skeleton_proposals.py` | Bounded skeleton proposal generation with size and boundary checks |
| `template_retriever.py` | Same-family retrieval first, then cross-family verified exemplar fallback |
| `nodes.py` | LangGraph node functions plus live proposal handling in `decompose_node()` |
| `deterministic_decompose.py` | Deterministic local decomposition, including skeleton-backed emission for recognized strategies |
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

1. **Ghost simulation** (optional) -- converts CDG atomic leaves to `SimNode`s and runs `simulate_graph()` from `sciona.atoms.ghost`. Catches domain mismatches (e.g., feeding frequency-domain data into an FFT that expects time-domain), shape errors, and type violations *before* expensive compilation.
2. **Assembly** -- topologically sorts CDG nodes, fuses each atomic leaf with its `MatchResult`, and emits source code (Lean 4, Coq, or Python).
3. **Compilation** -- sends the skeleton through the `ProofEnvironment` compiler.

The ghost simulation is best-effort: if `sciona.atoms` is not installed or no atoms have registered witnesses, it is silently skipped. Non-DSP atoms (those without witnesses) are also skipped. Results are attached to `skeleton.metadata["ghost_simulation"]`.

#### Ghost Witness integration (`ghost_sim.py`)

The ghost simulation pass bridges the two repositories:

```
sciona.atoms (sciona-atoms)           sciona (sciona)
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
| `sciona.atoms.numpy.fft` | `fft`, `ifft`, `rfft`, `irfft` |
| `sciona.atoms.scipy.fft` | `dct`, `idct` |
| `sciona.atoms.scipy.signal` | `butter`, `cheby1`, `cheby2`, `firwin`, `sosfilt`, `lfilter`, `freqz` |
| `sciona.atoms.scipy.sparse_graph` | `graph_laplacian`, `graph_fourier_transform`, `heat_kernel_diffusion` |

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

Wraps the four-round pipeline in a proposal-driven optimisation loop. The
current Principal no longer assumes a simple "evaluate gradients, then
time-travel" update. It now combines:

- configurable objectives (`latency`, `memory`, `flop_count`, `precision`,
  `uncertainty`, `rmse`, `mse`, `mae`, `structure`)
- per-structure hyperparameter search via Optuna
- live cross-family expansion through shipped expansion rule sets
- sibling proposal comparison between expansion, local mutation, and
  redecomposition from the same evaluated baseline
- rollback and cached-evaluation reuse to avoid paying duplicate execution cost

#### Graph topology

```
seed ──> suggest_params ──> forward ──> evaluate ──> gradients ──> select_proposal
                              |             |             |               |
                         (pruned early)     |        (param loop)         |
                              |             |             |               |
                              v             |             +------> suggest_params
                         time_travel <------+                             |
                              |                                            |
                              +--------------------> suggest_params <------+
```

`select_proposal` compares sibling candidates from the same baseline rather
than committing to stage-order bias. If no proposal beats the baseline, the
loop falls back to `time_travel_update()`.

#### Key modules

| Module | Role |
|--------|------|
| `models.py` | `OptimizationMetric`, `NodeTelemetry`, `BenchmarkResult`, `NodeGradient` |
| `metric_selection.py` | Objective resolution for `rmse`, `mae`, `mse`, `structure`, `uncertainty`, etc. |
| `eval_spec.py` | Reference-loss configuration for benchmark-aware evaluation |
| `evaluator.py` | `ExecutionSandbox` -- runs instrumented artifacts as subprocesses, parses `trace.jsonl`, computes scalar loss |
| `profiler.py` | Metric-aware profiling and reporting |
| `backprop.py` | Per-node optimisation gradients for latency / memory / uncertainty / structure |
| `reference_attribution.py` | Loss-aware node attribution for reference metrics such as RMSE |
| `hpo.py` | `OptunaManager` plus per-structure hyperparameter suggestion plumbing |
| `variant_mutation.py` | Family-based variant mutation and cross-family ledger mutation |
| `expansion.py` | `ExpansionEngine` -- runs all registered rule sets over the current CDG |
| `expansion_rules/` | Family-specific rewrite rule sets used by the live expansion stage |
| `structure_summary.py` | Per-trial diversity and cross-family structure summaries |
| `graph.py` | `PrincipalState`, `PrincipalDeps`, proposal selection, rollback, reuse, and routing |

#### Optimisation loop

1. **seed_population** -- build the initial CDG and seed optimisation state.
2. **suggest_params** -- sample node-level hyperparameters for the current
   primitive signature when tunables are available.
3. **execute_forward** -- synthesize/export/evaluate the current CDG; early-prune
   on structural failure when possible.
4. **evaluate_run** -- compute scalar loss, persist trial history, structure
   summary, and parameter assignments.
5. **compute_gradients** -- compute node-level blame under the chosen objective.
   For reference-loss objectives (`rmse`, `mse`, `mae`), this uses measured
   loss-aware attribution instead of the older uncertainty-only ranking.
6. **select_proposal** -- evaluate sibling candidates from the same baseline:
   - **expansion** via `ExpansionEngine(default_rule_sets())`
   - **local mutation** via family plugins and the cross-family ledger path
   - **redecompose** via Architect checkpoint time-travel
7. **time_travel_update** -- only used when no sibling proposal beats the
   baseline or when the trial budget is exhausted for the current branch.

#### Shipped expansion rule families

The live expansion engine loads all built-in rule sets from
`sciona/principal/expansion_rules/__init__.py`. Current shipped families:

- `signal_event_rate`
- `sequential_filter`
- `mcmc`
- `graph_traversal`
- `dynamic_programming`
- `greedy`
- `divide_and_conquer`
- `graph_optimization`
- `sorting`
- `string_matching`
- `searching`
- `geometry`
- `number_theory`
- `signal_transform`
- `signal_filter`
- `signal_detect_measure`
- `graph_signal_processing`
- `vi_advi`
- `particle_filter`
- `kalman_filter`
- `belief_propagation`
- `linear_algebra`
- `optimization`
- `combinatorics`
- `neural_network`
- `clustering`
- `dimensionality_reduction`

These rule sets are family-owned, but the expansion engine is family-agnostic:
all diagnostics are collected together and the resulting proposal is compared
against other sibling candidates by measured loss.

#### Cross-family behaviour

The Principal now treats cross-family structure as normal, not exceptional:

- expansion rules can insert structure from any shipped family
- local mutation can choose structurally compatible cross-family primitives
  with a penalty, rather than banning them
- refinement retrieval can contribute cross-family verified exemplars
- structure summaries record diversity metrics such as family entropy,
  distinct primitive families, and cross-family edge counts

#### Safety and reuse

- harmful proposal branches can be rolled back to the baseline
- selected sibling proposals can reuse cached evaluations instead of
  re-running immediately as the next official trial
- optimize telemetry records proposal selection, rollback, cached reuse, and
  skeleton-proposal statistics when present in trial history

#### Assembly instrumentation

The `Assembler` supports a `with_telemetry=True` flag that wraps each atomic
call in a `_sciona_probe()` helper emitting `trace.jsonl` records with
`node_id`, `execution_time_ms`, `peak_memory_bytes`, and `error_expansion`.
The `ExecutionSandbox` parses these traces after subprocess execution.

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
   - Surface primitive, template, and bounded skeleton proposals
   - High-confidence template can short-circuit
   - Skeleton can short-circuit only if it clears the complexity margin
   - Otherwise deterministic / LLM decomposition emits sub-nodes
   - Catalog + retrieval match -> mark ATOMIC, otherwise PENDING

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
   - Architect decomposes goal under new thread_id

2. suggest_params:
   - Optuna samples node-level hyperparameters for the current primitive signature

3. execute_forward:
   - Ghost simulation -> GhostSimReport
   - OptunaManager.check_early_prune() -- abort on structural failure / inf bounds
   - If not pruned: synthesize_fn(cdg, matches) -> ExportBundle

4. evaluate_run:
   - ExecutionSandbox runs instrumented artifact (subprocess)
   - Parses trace.jsonl -> NodeTelemetry per node
   - _compute_loss(telemetry, metric) -> global_loss
   - Tracks best_loss across trials

5. compute_gradients:
   - CreditAssigner.compute_gradients(cdg, benchmark, ghost_report, metric)
   - LATENCY/FLOP_COUNT: % of total execution time per node
   - MEMORY: % of total peak memory per node
   - PRECISION/UNCERTAINTY: ghost-sim interval gradients or telemetry error expansion
   - RMSE/MSE/MAE: measured reference-loss attribution
   - Returns sorted NodeGradient[] -- top = bottleneck

6. select_proposal:
   - Evaluate sibling candidates from the same baseline:
     expansion, local_mutation, redecompose
   - Choose only measured-loss improvements
   - Reuse cached evaluation when the chosen proposal does not require fresh param search

7. time_travel_update:
   - Used when no sibling proposal beats the baseline
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
           synthesizer/        |               (sciona.atoms [optional])
                 |             |
           principal/          |               (optuna)
                 |             |
               cli.py          |
                               |
                          IngestionBundle
                     (CDG + atoms + witnesses)
```

```
                 sciona.atoms (sciona-atoms)
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

The indexer and judge are fully independent of each other. The hunter depends on both (through their protocol interfaces, not concrete classes). The architect is independent of the indexer/judge/hunter -- its output (CDGExport) is the input to the hunter via the handoff bridge. The ingester is independent of the hunter/indexer/judge -- it produces `IngestionBundle`s (CDG + generated source + match results) that the Synthesizer can consume directly. For non-Python sources, the ingester uses tree-sitter for extraction and generates FFI bindings (ctypes for C++/Rust, juliacall for Julia). The synthesizer depends on the hunter (MatchResult) and architect (CDGExport), and optionally on `sciona.atoms` for ghost witness simulation. If `sciona.atoms` is not installed, the synthesizer works normally without the simulation pass. The principal depends on the architect (for time-travel forking), synthesizer (ghost simulation + assembly), and optuna for HPO. It is the outermost layer: all other components are unaware of the principal's existence.
