# Architecture

AGEO-Matcher implements the full Two-Round Agentic Development Cycle:

- **Round 1 (Architect)**: Decomposes a high-level goal into an atomic Conceptual Dependency Graph (CDG) via LangGraph, with PostgreSQL-backed persistence for checkpoint time-travel and forking.
- **Round 2 (Hunter)**: Grounds each atomic CDG leaf into a verified library function in Lean 4/Mathlib or Coq/Rocq.

## Design principle

**Deterministic -> Agentic -> Deterministic** sandwich (applies to both rounds):

```
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
[Retrieval Agent]       agentic -- LLM ranks candidates, reformulates queries
  |
  v
[Verification Oracle]   deterministic -- compiler says yes or no
  |
  v
MatchResult
```

LLMs are confined to agentic layers. The indexer, oracle, and handoff validation are pure functions of their inputs -- no hallucination surface.

## Components

### Shared types (`ageom/types.py`)

Five frozen dataclasses form the vocabulary shared across all components:

| Type | Role |
|------|------|
| `Declaration` | A theorem/definition extracted from a proof library (name, type signature, docstring, prover) |
| `PDGNode` | A predicate to be grounded (statement, informal description, target prover) |
| `CandidateMatch` | A Declaration paired with a retrieval score and method |
| `VerificationResult` | Compiler output for a single candidate (verified/not, error message) |
| `MatchResult` | Final result for a PDG node (best verified match + all attempts) |

### Protocols (`ageom/protocols.py`)

Four `typing.Protocol` interfaces define the contracts:

- **`SemanticIndex`** -- `search_by_embedding()`, `search_by_type()`, `get_declaration()`
- **`ProofEnvironment`** -- `check_term()`, `check_proof()`, `get_type()`, `close()`
- **`VerificationOracle`** -- `verify_candidate()`, `verify_candidates()`
- **`RetrievalAgent`** -- `find_match()`

All component implementations are structural subtypes of these protocols. No inheritance required.

### Conceptual Dependency Agent (`ageom/architect/`) -- Round 1

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

### Semantic Indexer (`ageom/indexer/`)

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

### Verification Oracle (`ageom/judge/`)

Compiler-based proof checking. No approximation -- the compiler is the ground truth.

- **`LeanEnvironment`** -- Wraps `lean-interact`'s `LeanServer` with a `TempRequireProject` configured for Mathlib. Checks terms via `example : {type} := {term}` and proofs via `theorem _check : {stmt} := by {body}`. All calls are `async` (sync REPL wrapped in `asyncio.to_thread`).
- **`CoqEnvironment`** -- Writes to temporary `.v` files and compiles via `coqpyt`. Same interface: `Definition _check` / `Lemma _check`.
- **`VerificationOracleImpl`** -- Routes to the correct environment based on `pdg_node.prover`. `verify_candidates()` short-circuits on first verified match.

Verification strategy: attempt direct type unification -- `@CandidateName` as a term for the PDG node's statement. If the compiler accepts it, the match is proven correct.

### Retrieval Agent (`ageom/hunter/`)

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
| `RankCandidates` | LLM scores and reorders candidates by match likelihood | `VerifyTopK` |
| `VerifyTopK` | Sends top-K to Verification Oracle | `End` (verified) or `ReformulateQuery` or `End` (budget exhausted) |
| `ReformulateQuery` | LLM analyzes compiler errors, generates new search queries | `InitialSearch` |

#### State and dependencies

- **`HunterState`** -- Mutable state threaded through the graph: accumulated candidates, verification results, queries tried, compiler feedback, iteration counter.
- **`HunterDeps`** -- Injected dependencies: `SemanticIndex`, `VerificationOracle`, `LLMClient`.

#### Why pydantic-graph

- **Type-safe transitions** -- The return type of each node's `run()` method declares which nodes it can transition to. Mypy verifies the graph is well-formed at type-check time.
- **Built-in persistence** -- State can be serialized for pause/resume.
- **Observability** -- Tracing/spans and Mermaid diagram generation come for free.

### LLM integration (`ageom/hunter/llm.py`)

`LLMClient` is a Protocol with a single method: `async complete(system, user) -> str`.
Built-in implementations:
- `ClaudeLLMClient` via `anthropic.AsyncAnthropic`
- `CodexLLMClient` via `openai.AsyncOpenAI` (Codex-compatible models)
- `LlamaCppLLMClient` via OpenAI-compatible llama.cpp endpoint (`extra_body.grammar`)

`create_llm_client(...)` selects the provider (`anthropic`, `codex`, or `llama_cpp`) from config/CLI.

Three prompt templates drive the agent's reasoning:
- **`REFORMULATE_QUERY`** -- Given failed queries + compiler errors, generate new search terms
- **`SCORE_CANDIDATES`** -- Given predicate + candidate list, return ranked indices
- **`ANALYZE_FAILURE`** -- Given compiler error, explain why and suggest direction

## Configuration

`AgeomConfig` (pydantic-settings) reads from `.env` with prefix `AGEOM_`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGEOM_INDEX_DIR` | `data/index` | FAISS index location |
| `AGEOM_LLM_PROVIDER` | `anthropic` | Global LLM provider (`anthropic`, `codex`, or `llama_cpp`) |
| `AGEOM_ANTHROPIC_API_KEY` | | Claude API key |
| `AGEOM_OPENAI_API_KEY` | | OpenAI API key for Codex |
| `AGEOM_OPENAI_BASE_URL` | *(empty)* | Optional OpenAI-compatible endpoint |
| `AGEOM_LLM_MODEL` | `claude-sonnet-4-5-20250929` | LLM model for Hunter |
| `AGEOM_LLAMA_CPP_BASE_URL` | `http://127.0.0.1:8080/v1` | Local llama.cpp endpoint |
| `AGEOM_HUNTER_LLM_PROVIDER` | `llama_cpp` | Hunter-specific provider (defaults local) |
| `AGEOM_HUNTER_LLM_MODEL` | `llama-3.1-8b-instruct` | Hunter local model |
| `AGEOM_HUNTER_MODE` | `speculative_local` | Hunter loop mode |
| `AGEOM_HUNTER_USE_GBNF` | `true` | Grammar-constrained decoding in Hunter |
| `AGEOM_EMBEDDING_MODEL` | `microsoft/unixcoder-base` | Embedding model |
| `AGEOM_LEAN_TOOLCHAIN` | `leanprover/lean4:v4.14.0` | Lean version |
| `AGEOM_HUNTER_MAX_ITERATIONS` | `5` | Max search-verify-refine loops |
| `AGEOM_HUNTER_TOP_K_VERIFY` | `3` | Candidates sent to Oracle per iteration |
| `AGEOM_HUNTER_SEARCH_K` | `20` | Candidates retrieved per search |
| `AGEOM_POSTGRES_URI` | *(empty)* | PostgreSQL URI for checkpoint persistence (omit for in-memory) |
| `AGEOM_ARCHITECT_MAX_DEPTH` | `8` | Max CDG decomposition depth |
| `AGEOM_ARCHITECT_LLM_PROVIDER` | *(empty)* | Optional Round 1 provider override (falls back to `AGEOM_LLM_PROVIDER`) |
| `AGEOM_ARCHITECT_LLM_MODEL` | `claude-sonnet-4-5-20250929` | LLM model for Architect |
| `AGEOM_SKILL_INDEX_DIR` | `data/skill_index` | Skill catalog and index location |

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

Each step is checkpointed. Use `ageom history <thread-id>` to inspect, or `agent.fork()` to branch from any checkpoint.

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

## Dependency graph

```
ageom/types.py, protocols.py, config.py    (no external deps beyond pydantic)
         |
    +----+----+----------+
    |         |          |
 indexer/   judge/    architect/    (faiss, transformers, lean-interact,
    |         |          |           coqpyt, langgraph, psycopg)
    +----+----+          |
         |               |
      hunter/            |          (pydantic-graph, anthropic, openai)
         |               |
         +-------+-------+
                 |
               cli.py
```

The indexer and judge are fully independent of each other. The hunter depends on both (through their protocol interfaces, not concrete classes). The architect is independent of the indexer/judge/hunter -- its output (CDGExport) is the input to the hunter via the handoff bridge.
