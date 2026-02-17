# AGEO-Matcher

Functional Matching Agent that grounds high-level predicates into verified library functions in **Lean 4/Mathlib** and **Coq/Rocq**.

Given a predicate like `forall n m : Nat, n + m = m + n`, AGEO-Matcher searches a proof library, ranks candidates, and verifies matches through the compiler -- returning `Nat.add_comm` with a machine-checked proof that the types unify.

## How it works

The system implements a **Two-Round Agentic Development Cycle**:

**Round 1 -- Architect** (Conceptual Dependency Agent): decomposes a high-level goal into an atomic Conceptual Dependency Graph (CDG) using LangGraph, with PostgreSQL-backed checkpointing for time-travel and fork/resume.

**Round 2 -- Hunter** (Functional Matching Agent): grounds each atomic CDG leaf into a verified library function through a **Deterministic -> Agentic -> Deterministic** sandwich:

1. **Semantic Indexer** -- embeds library declarations with UniXcoder into a FAISS vector store
2. **Retrieval Agent** -- LLM-driven search loop with query reformulation (pydantic-graph state machine)
3. **Verification Oracle** -- compiler-based type checking via Lean 4 / Coq REPLs

The handoff from Round 1 to Round 2 is validated: every atomic leaf must have a `type_signature` and `description` before conversion to PDG nodes.

See [ARCH.md](ARCH.md) for the full architecture.

## Installation

```bash
# Core only
pip install -e .

# With Lean 4 support
pip install -e ".[indexer,lean,hunter]"

# With Coq support
pip install -e ".[indexer,coq,hunter]"

# Everything
pip install -e ".[all]"
```

### External requirements

- **Lean 4**: Install [elan](https://github.com/leanprover/elan) and run `lean-explore data fetch` for Mathlib data
- **Coq**: Install via opam with your project's dependencies
- **LLM**: Configure one provider in `.env`
  - Anthropic: `AGEOM_LLM_PROVIDER=anthropic` + `AGEOM_ANTHROPIC_API_KEY`
  - Codex: `AGEOM_LLM_PROVIDER=codex` + `AGEOM_OPENAI_API_KEY`
  - Local llama.cpp (Hunter default): `AGEOM_HUNTER_LLM_PROVIDER=llama_cpp`

## Configuration

All settings are read from `.env` (prefixed with `AGEOM_`) via pydantic-settings:

```bash
# .env
AGEOM_INDEX_DIR=data/index
AGEOM_LLM_PROVIDER=anthropic
AGEOM_ANTHROPIC_API_KEY=sk-ant-...
AGEOM_LLM_MODEL=claude-sonnet-4-5-20250929
# For Codex/OpenAI provider:
# AGEOM_OPENAI_API_KEY=sk-...
# AGEOM_LLM_MODEL=codex-mini-latest
# Hunter local defaults (GBNF + speculative retrieval):
AGEOM_HUNTER_LLM_PROVIDER=llama_cpp
AGEOM_HUNTER_LLM_MODEL=llama-3.1-8b-instruct
AGEOM_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080/v1
AGEOM_HUNTER_MODE=speculative_local
AGEOM_HUNTER_USE_GBNF=true
AGEOM_HUNTER_MAX_ITERATIONS=5

# PostgreSQL persistence (optional -- omit for in-memory only)
AGEOM_POSTGRES_URI=postgresql://localhost:5432/ageom_architect
```

CLI flags override `.env` when provided. See `ageom/config.py` for all options.

## Usage

### Build an index

```bash
# Index Lean 4 / Mathlib declarations
ageom index build --prover lean4

# Index a Coq project
ageom index build --prover coq --path ./my-coq-project
```

### Decompose a goal (Round 1)

```bash
# Basic decomposition (in-memory checkpointing)
ageom decompose "Implement merge sort" --no-persist --output cdg.json

# Use Codex for Round 1 (override .env)
ageom decompose "Implement merge sort" --llm-provider codex --llm-model codex-mini-latest --no-persist

# With a specific thread ID
ageom decompose "Sort and search" --thread-id my-run-01

# View checkpoint history for a thread
ageom history my-run-01
```

### Match predicates (Round 2)

```bash
# Single statement
ageom match --statement "forall n m : Nat, n + m = m + n" --prover lean4

# Use Codex for Round 2 (override .env)
ageom match --statement "forall n m : Nat, n + m = m + n" --prover lean4 --llm-provider codex --llm-model codex-mini-latest

# Use local llama.cpp for Round 2 (default if configured in .env)
ageom match --statement "forall n m : Nat, n + m = m + n" --prover lean4 --llm-provider llama_cpp --llm-model llama-3.1-8b-instruct

# Batch from a PDG file
ageom match --pdg-file predicates.json --prover lean4
```

The PDG file is a JSON array:

```json
[
  {
    "predicate_id": "p1",
    "statement": "forall n m : Nat, n + m = m + n",
    "informal_desc": "commutativity of addition"
  }
]
```

## Development

```bash
pip install -e ".[dev]"
pytest                     # run all tests (skips slow/model-download tests)
pytest -m slow             # run slow tests (requires transformers + torch)
mypy ageom/
```

## Project structure

```
ageom/
  types.py          Shared domain types (Declaration, PDGNode, MatchResult, ...)
  protocols.py      Protocol interfaces (SemanticIndex, ProofEnvironment, ...)
  config.py         AgeomConfig (pydantic-settings, reads .env)
  cli.py            CLI entrypoint (decompose, history, match, index, skill)
  architect/        Round 1 -- Conceptual Dependency Agent
    models.py         CDG node/edge Pydantic models
    catalog.py        PrimitiveCatalog (CLRS-30, coq-100-theorems)
    embedder.py       FAISS-based SkillIndex for primitive matching
    skeletons.py      Pre-built graph templates per paradigm
    nodes.py          LangGraph node functions (select_strategy, decompose, critique, ...)
    state.py          DecompositionState TypedDict + DecompositionDeps
    graph.py          StateGraph assembly + DecompositionAgent (with time-travel)
    checkpointer.py   Checkpointer factory (AsyncPostgresSaver / MemorySaver)
    handoff.py        CDGExport, validate_handoff, to_pdg_nodes (Round 1 -> 2 bridge)
    prompts.py        Prompt templates for decomposition LLM calls
    ingest_clrs.py    CLRS-30 ingestion
    ingest_coq100.py  coq-100-theorems ingestion
  indexer/          Semantic Indexer
    embedder.py       UniXcoder embeddings (768-dim, L2-normalized)
    faiss_store.py    FAISS IndexFlatIP vector store
    lean_source.py    Lean 4/Mathlib declarations via lean-explore
    coq_source.py     Coq declarations via coqpyt
    builder.py        IndexBuilder + SemanticIndexImpl
  judge/            Verification Oracle
    lean_env.py       Lean 4 REPL via lean-interact
    coq_env.py        Coq compiler via coqpyt
    checker.py        VerificationOracleImpl (routes by prover)
  hunter/           Round 2 -- Retrieval Agent
    nodes.py          pydantic-graph nodes (search/rank/verify/reformulate cycle)
    graph.py          Graph assembly + HunterAgent
    llm.py            LLM clients (Anthropic Claude + Codex/OpenAI)
    prompts.py        Prompt templates
tests/
```

## License

MIT
