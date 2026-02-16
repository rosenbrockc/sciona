# AGEO-Matcher

Functional Matching Agent that grounds high-level predicates into verified library functions in **Lean 4/Mathlib** and **Coq/Rocq**.

Given a predicate like `forall n m : Nat, n + m = m + n`, AGEO-Matcher searches a proof library, ranks candidates, and verifies matches through the compiler -- returning `Nat.add_comm` with a machine-checked proof that the types unify.

## How it works

The system follows a **Deterministic -> Agentic -> Deterministic** sandwich:

1. **Semantic Indexer** -- embeds library declarations with UniXcoder into a FAISS vector store
2. **Retrieval Agent** -- LLM-driven search loop with query reformulation (pydantic-graph state machine)
3. **Verification Oracle** -- compiler-based type checking via Lean 4 / Coq REPLs

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
- **LLM**: Set `AGEOM_ANTHROPIC_API_KEY` in `.env`

## Configuration

All settings are read from `.env` (prefixed with `AGEOM_`) via pydantic-settings:

```bash
# .env
AGEOM_INDEX_DIR=data/index
AGEOM_ANTHROPIC_API_KEY=sk-ant-...
AGEOM_LLM_MODEL=claude-sonnet-4-5-20250929
AGEOM_HUNTER_MAX_ITERATIONS=5
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

### Match predicates

```bash
# Single statement
ageom match --statement "forall n m : Nat, n + m = m + n" --prover lean4

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
  cli.py            CLI entrypoint
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
  hunter/           Retrieval Agent
    nodes.py          pydantic-graph nodes (search/rank/verify/reformulate cycle)
    graph.py          Graph assembly + HunterAgent
    llm.py            LLM client (Claude API)
    prompts.py        Prompt templates
tests/
```

## License

MIT
