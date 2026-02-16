# Architecture

AGEO-Matcher implements the "Second Round" Functional Matching Agent from the Two-Round Agentic Development Cycle. It takes high-level predicates from a Predicate Dependency Graph (PDG) and grounds each one into a verified library function in Lean 4/Mathlib or Coq/Rocq.

## Design principle

**Deterministic -> Agentic -> Deterministic** sandwich:

```
PDG Node
  |
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

The LLM is confined to the middle layer. The indexer and oracle are pure functions of their inputs -- no hallucination surface.

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

`LLMClient` is a Protocol with a single method: `async complete(system, user) -> str`. The `ClaudeLLMClient` implementation wraps `anthropic.AsyncAnthropic`. Swappable to any provider.

Three prompt templates drive the agent's reasoning:
- **`REFORMULATE_QUERY`** -- Given failed queries + compiler errors, generate new search terms
- **`SCORE_CANDIDATES`** -- Given predicate + candidate list, return ranked indices
- **`ANALYZE_FAILURE`** -- Given compiler error, explain why and suggest direction

## Configuration

`AgeomConfig` (pydantic-settings) reads from `.env` with prefix `AGEOM_`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGEOM_INDEX_DIR` | `data/index` | FAISS index location |
| `AGEOM_ANTHROPIC_API_KEY` | | Claude API key |
| `AGEOM_LLM_MODEL` | `claude-sonnet-4-5-20250929` | LLM model for Hunter |
| `AGEOM_EMBEDDING_MODEL` | `microsoft/unixcoder-base` | Embedding model |
| `AGEOM_LEAN_TOOLCHAIN` | `leanprover/lean4:v4.14.0` | Lean version |
| `AGEOM_HUNTER_MAX_ITERATIONS` | `5` | Max search-verify-refine loops |
| `AGEOM_HUNTER_TOP_K_VERIFY` | `3` | Candidates sent to Oracle per iteration |
| `AGEOM_HUNTER_SEARCH_K` | `20` | Candidates retrieved per search |

## Data flow

```
1. User provides: PDGNode { statement: "forall n m, n + m = m + n", prover: lean4 }

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
    +----+----+
    |         |
 indexer/   judge/          (faiss, transformers, lean-interact, coqpyt)
    |         |
    +----+----+
         |
      hunter/               (pydantic-graph, anthropic)
         |
       cli.py
```

The indexer and judge are fully independent of each other. The hunter depends on both (through their protocol interfaces, not concrete classes).
