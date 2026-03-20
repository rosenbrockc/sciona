# AGEO-Matcher Roles

AGEO-Matcher implements **verified retrieval-augmented composition**: decompose a
goal into typed sub-problems, match those sub-problems against a catalog of real
library functions, verify that the matches actually work, and assemble the result.

The pipeline is organized into specialized roles, each solving a distinct
subproblem. No single role has full knowledge of the pipeline -- they communicate
through typed data contracts (CDG, PDGNode, MatchResult, SkeletonFile).

Roles follow the **deterministic-first, LLM-fallback** principle: every operation
that can be handled by a regex, AST walk, embedding lookup, or type check is
handled deterministically. LLMs are reserved for conceptual decomposition and
ambiguous cases. This is the core architectural bet -- each deterministic tool is
a permanent cost, latency, and reliability win.

The pipeline supports **graduated execution tiers** (`rapid`, `structured`,
`single_agent`, `verified`) so the overhead scales with the task's correctness
requirements. Not every role is active in every mode.

---

## Smart Ingester

**Problem solved:** Extracting algorithmic structure from existing source code in
multiple languages and converting it into the same atom graph format used by the
rest of the pipeline.

**What it does:** A three-phase pipeline (Extract -> Chunk -> Emit) that ingests
existing classes/structs/modules and produces `IngestionBundle`s:

1. **Phase 1 -- Extraction**: Language-specific parsing via `BaseExtractor`
   protocol. Dispatches by file extension: Python uses `ast`, C++/Julia/Rust use
   tree-sitter. Produces a `RawDataFlowGraph` with method facts, field
   reads/writes, init chains, call graphs, and SSA edges. All extractors produce
   the same schema -- downstream phases are language-agnostic.

2. **Phase 2 -- Semantic Chunking**: A control-flow decomposer first attempts
   deterministic AST-based splitting at structural boundaries (loops,
   conditionals, significant function calls). If the confidence is high enough,
   this replaces the LLM call entirely. Otherwise, the LLM groups methods into
   `MacroAtomSpec`s, hoists cross-window state into `StateModelSpec`s, and
   defines dependency edges. A critic validates attribute coverage.

3. **Phase 3 -- Code Generation**: Deterministic emission of `@register_atom`
   wrappers (with decorator passthrough and external atom support), Pydantic state
   models, ghost witness functions, CDG nodes/edges, and match results. For
   non-Python sources, appends FFI bindings (ctypes for C++/Rust, juliacall for
   Julia).

Verification loops run mypy type checking and ghost simulation, with LLM-assisted
repair on failure.

**Interacts with:**
- **Synthesizer** -- `IngestionBundle` output is format-identical to what the
  Synthesizer expects (CDG + atoms + witnesses + match results)
- **Ghost Registry** (ageoa) -- generated witnesses follow the `@register_atom`
  pattern

**Key files:** `sciona/ingester/graph.py`, `sciona/ingester/base_extractor.py`,
`sciona/ingester/treesitter_extractor.py`, `sciona/ingester/emitter.py`,
`sciona/ingester/ffi_emitter.py`, `sciona/ingester/control_flow_decomposer.py`

---

## Orchestrator

**Problem solved:** Coordinating multi-round feedback between decomposition and
grounding when the first attempt fails.

**What it does:** Runs the top-level loop: Architect decomposes a goal, Hunter
tries to ground each atomic node, and failures feed back to the Architect for
re-decomposition. Manages a configurable round budget (default: 3) and decides
when to mark nodes as ungroundable. Includes deterministic split patterns for
~20 domain-specific failure cases (matrix factorization, optimization, FFT,
clustering, sorting, etc.) that re-decompose failed leaves without an LLM call.

In `single_agent` mode, the **SingleAgentPlanner** wraps the orchestrator with
a deterministic decision tree: direct match -> decompose -> batch match ->
partial acceptance (>=70% leaves) -> selective re-decomposition -> escalation.
Most goals resolve before reaching full orchestration.

**Interacts with:**
- **Architect** -- invokes decomposition and refinement
- **Hunter** -- invokes matching for each atomic node
- **User** -- receives the initial goal, returns final artifacts

**Key files:** `sciona/orchestrator.py`, `sciona/services/planner_service.py`,
`sciona/cli.py`

---

## Architect

**Problem solved:** Turning a vague, high-level goal into a precise graph of
atomic algorithmic operations that can be individually grounded to library
functions.

**What it does:** A LangGraph state machine that recursively decomposes a goal
into a Conceptual Dependency Graph (CDG). At each node, it:
1. Selects a paradigm strategy (divide-and-conquer, greedy, etc.)
2. Instantiates a skeleton template for that paradigm
3. Decomposes non-atomic nodes into sub-nodes via LLM
4. Validates each decomposition through a Critic pass
5. Repeats until all leaf nodes are marked ATOMIC

**Interacts with:**
- **Catalog** -- checks whether a node matches a known algorithmic primitive
  (the stop condition for decomposition)
- **Skill Index** -- semantic search over primitives to suggest candidates
  during decomposition
- **Critic** -- validates each decomposition before accepting it
- **Orchestrator** -- receives the goal, returns CDG; accepts refinement
  requests when Hunter fails

**Key files:** `sciona/architect/graph.py`, `sciona/architect/nodes.py`,
`sciona/architect/handoff.py`

---

## Critic

**Problem solved:** Preventing the Architect from producing structurally invalid
or semantically nonsensical decompositions.

**What it does:** A two-phase validation gate within the Architect round:
1. **Deterministic checks** -- arity (>=2 children), depth bounds, no
   self-loops, I/O name matching, atomicity claims verified against catalog
2. **LLM critique** -- semantic evaluation of completeness, type compatibility,
   and whether the sub-nodes actually solve the parent's problem

If the Critic rejects a decomposition, the Architect retries (up to 3 times)
with the rejection reason injected into the prompt.

**Interacts with:**
- **Architect** -- receives decomposition proposals, returns approve/reject
  with reasons and flagged nodes

**Key files:** `sciona/architect/nodes.py` (critique_decomposition, prepare_retry)

---

## Catalog

**Problem solved:** Providing the universe of known algorithmic primitives that
serve as the "ground truth" for what the Architect can decompose into.

**What it does:** Maintains a searchable collection of ~200-500 algorithmic
primitives (from CLRS-30, Coq 100 Theorems, Mathlib, etc.). Provides keyword
overlap scoring and exact-match lookup. Acts as the stop condition oracle: a
node is ATOMIC if and only if it matches a catalog entry.

**Interacts with:**
- **Architect** -- consulted during decomposition to suggest primitives and
  validate atomicity claims
- **Critic** -- used to verify that nodes claimed as ATOMIC actually exist

**Key files:** `sciona/architect/catalog.py`

---

## Skill Index

**Problem solved:** Finding semantically similar primitives even when the
Architect's natural language description doesn't exactly match catalog keywords.

**What it does:** Wraps a FAISS vector store over primitive descriptions,
embedded with UniXcoder. Returns the top-K most similar primitives for a given
query. Complements the Catalog's keyword-based search with embedding-based
semantic search.

**Interacts with:**
- **Architect** -- queried during decomposition to find relevant primitives

**Key files:** `sciona/architect/embedder.py`

---

## Hunter

**Problem solved:** Grounding each atomic predicate from the CDG to a specific,
verified library function.

**What it does:** A pydantic-graph state machine with 4 nodes that, for each
PDGNode:
1. **InitialSearch** -- queries the FAISS index by embedding and type signature
2. **RankCandidates** -- an embedding reranker scores candidates by cosine
   similarity with a type-token bonus; falls back to LLM only when the top-2
   margin is below threshold
3. **VerifyTopK** -- sends top candidates to the Judge for type-checking
4. **ReformulateQuery** -- a deterministic failure analyzer handles known error
   patterns (unknown identifier, arity mismatch, type incompatibility) via
   regex; falls back to LLM for unrecognized errors

Loops until a verified match is found or the iteration budget is exhausted.

**Interacts with:**
- **Index** -- semantic search for candidate declarations
- **Judge** -- type-checks candidates against formal specifications
- **Orchestrator** -- receives PDGNodes, returns MatchResults; failures trigger
  Architect refinement

**Key files:** `sciona/hunter/nodes.py`, `sciona/hunter/graph.py`,
`sciona/hunter/state.py`, `sciona/hunter/embedding_reranker.py`,
`sciona/hunter/failure_analyzer.py`

---

## Index

**Problem solved:** Fast retrieval of library declarations that are semantically
similar to a natural language or type-signature query.

**What it does:** A FAISS IndexFlatIP (cosine similarity) over declaration
embeddings produced by UniXcoder. Stores declaration metadata (name, type
signature, docstring, prover, source library) alongside the vectors. Supports
both embedding-based search and type-signature search.

Must be built offline as a prerequisite (`sciona index build`) from Mathlib,
Coq libraries, or Python packages.

**Interacts with:**
- **Hunter** -- queried during candidate retrieval
- **Skill Index** -- shares the same FAISSStore infrastructure

**Key files:** `sciona/indexer/builder.py`, `sciona/indexer/faiss_store.py`,
`sciona/indexer/unified.py`

---

## Judge

**Problem solved:** Determining whether a candidate library function actually
inhabits the type specified by a PDGNode -- using the target prover's own
compiler as the oracle.

**What it does:** The Verification Oracle routes each candidate to the
appropriate proof environment:
- **Lean 4:** compiles `example : {type} := @{candidate}` via lean-interact
- **Coq:** compiles `Definition _check : {type} := {candidate}.` via coqpyt
- **Python:** import-based verification (default) -- checks that the function
  is importable, callable, and has compatible arity via `inspect.signature`.
  Optional mypy --strict mode available via `verify_mode="mypy"`.

Returns a VerificationResult with success/failure, compiler output, and a
VerificationLevel (KERNEL_PROOF, TYPE_CHECKED, CONTRACT_CHECKED, UNVERIFIED).

Supports parallel verification with configurable concurrency.

**Interacts with:**
- **Hunter** -- called to verify candidate matches
- **Repair Agent** -- called to compile repaired skeletons

**Key files:** `sciona/judge/checker.py`, `sciona/judge/lean_env.py`,
`sciona/judge/coq_env.py`, `sciona/judge/python_env.py`

---

## Assembler

**Problem solved:** Composing individually matched atoms into a single
compilable source file with correct data flow between them.

**What it does:** Takes the CDG and MatchResults and produces a SkeletonFile:
1. Builds AssemblyUnits from each verified atomic match
2. Infers GlueEdges with cast expressions for type mismatches
3. Topologically sorts units by data dependencies
4. Emits language-specific source (Lean 4, Coq, or Python) with:
   - Atomic definitions referencing matched library functions
   - Composition functions connecting atoms via glue edges
   - Import/open statements for required modules

**Interacts with:**
- **Synthesizer pipeline** -- produces the initial SkeletonFile consumed by
  Ghost Simulator and Repair Agent

**Key files:** `sciona/synthesizer/assembler.py`, `sciona/synthesizer/models.py`

---

## Ghost Simulator

**Problem solved:** Catching structural mismatches (shape, dtype, domain) before
committing to expensive compilation.

**What it does:** Runs abstract witness functions on metadata instead of real
data. For each atomic node with a registered ghost witness, it:
1. Builds abstract input values (AbstractSignal, AbstractArray, etc.)
2. Executes witnesses in topological order
3. Propagates metadata through the graph
4. Catches mismatches (e.g., feeding frequency-domain data into a time-domain
   filter)

Reports coverage (fraction of nodes with witnesses) and warns when coverage is
low. Nodes without witnesses are silently skipped.

**Interacts with:**
- **Assembler** -- runs on the CDG before or alongside assembly
- **Ghost Registry** (ageoa) -- looks up witness functions for each atom

**Key files:** `sciona/synthesizer/ghost_sim.py`

---

## Repair Agent

**Problem solved:** Iteratively fixing compilation errors and filling in proof
obligations (sorry/Admitted/NotImplementedError) in the assembled skeleton.

**What it does:** A pydantic-graph state machine that loops:
1. **CompileCheck** -- compiles the skeleton, classifies errors, detects
   regressions (rolls back if error count increases)
2. **DeterministicFix** -- applies all mechanical fixes in one pass (missing
   imports/opens, type coercions for known type pairs, common syntax
   corrections, undefined name resolution via common import database)
3. **LLMRepair** -- sends the highest-priority error to the LLM for a targeted
   patch (JSON with line range + replacement)
4. **SorryElimination** -- generates proof tactics or implementations to replace
   sorry/Admitted/NotImplementedError stubs

Tracks the best source seen (lowest error count) and returns it if the iteration
budget is exhausted.

**Interacts with:**
- **Judge** -- compiles the skeleton at each iteration
- **LLM** -- generates patches and proof tactics
- **Assembler** -- consumes the SkeletonFile produced by assembly

**Key files:** `sciona/synthesizer/repair.py`

---

## Extractor

**Problem solved:** Packaging the verified source code into a distributable
artifact with cryptographic certificates.

**What it does:** Takes the final SynthesisResult and:
1. Writes verified source to disk
2. Builds the target artifact (lake build / coqc / mypy / cargo)
3. Generates SHA-256 verification certificates
4. Optionally generates FFI bindings (C headers, Rust crate)

Supports export targets: LEAN_LIB, COQ_LIB, PYTHON_PKG, C_HEADER, RUST_FFI.

**Interacts with:**
- **Repair Agent** -- consumes the final repaired SkeletonFile
- **Native toolchains** -- invokes lake, coqc, mypy, cargo via subprocess

**Key files:** `sciona/synthesizer/extractor.py`,
`sciona/synthesizer/certificate.py`

---

## Principal

**Problem solved:** Finding the best decomposition structure and synthesis
configuration for a given goal, without requiring the user to manually tune
hyperparameters or guess which paradigm works best.

**What it does:** A LangGraph state machine that wraps the entire four-round
pipeline in a NAS-style optimisation loop:

1. **Seed** -- Uses Optuna to suggest trial parameters. The Architect decomposes
   the goal under a fresh checkpoint thread.
2. **Forward** -- Ghost simulation for early pruning (aborts trials with
   structural failures or infinite error bounds). If not pruned, runs the full
   synthesis pipeline to produce an ExportBundle with telemetry instrumentation.
3. **Evaluate** -- `ExecutionSandbox` runs the instrumented artifact as a
   subprocess, parses `trace.jsonl` into per-node `NodeTelemetry`, and computes a
   scalar loss (latency, memory, precision, or FLOP count).
4. **Backward** -- `CreditAssigner` computes per-node optimisation gradients,
   identifying the top bottleneck (e.g., "Node 'sort_array' consumed 85% of total
   execution time").
5. **Update** -- Walks the Architect's checkpoint history, finds the checkpoint
   just before the bottleneck node was created, forks a new thread via
   `architect.fork()`, injects a CONSTRAINT describing the bottleneck into the
   forked goal, and re-decomposes. Loops back to step 2.

The loop terminates when the trial budget is exhausted, no gradients can be
computed, or a non-recoverable error occurs.

**Interacts with:**
- **Architect** -- invokes decomposition; uses `get_state_history()` and `fork()`
  for time-travel coordinate descent
- **Synthesizer** -- ghost simulation for early pruning; precision gradients for
  credit assignment; telemetry instrumentation via `with_telemetry=True`
- **Optuna** -- study management, early pruning, fANOVA parameter importance
- **User** -- receives the goal and benchmark dataset, returns the best trial's
  artifact

**Key files:** `sciona/principal/graph.py`, `sciona/principal/evaluator.py`,
`sciona/principal/backprop.py`, `sciona/principal/hpo.py`,
`sciona/principal/models.py`

---

## FFI Emitter (ingester)

**Problem solved:** Generating Python-callable wrappers for foreign-language
implementations so that CDG output remains format-identical to pure Python.

**What it does:** Given a list of `MacroAtomSpec`s and a source language, emits a
complete Python module with:
- **C++ / Rust**: `ctypes.CDLL` loading, `argtypes`/`restype` declarations, and
  call-through wrapper functions
- **Julia**: `juliacall.Main` imports and `jl.eval()` call stubs

These stubs are appended to the generated atoms module by the emitter when
`source_language != "python"`.

**Interacts with:**
- **Emitter** (Phase 3) -- called to append FFI stubs to generated atom wrappers
- **TreeSitterExtractor** -- the upstream extractor that determines which language
  the FFI stubs target

**Key files:** `sciona/ingester/ffi_emitter.py`
