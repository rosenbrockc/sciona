# Shared Context Plan

## Goal

Enable agents to share useful context across parallel prompt calls and across repeated runs without re-ingesting unchanged inputs.

## Where Shared Context Helps Most

1. Hunter per-node matching loop in orchestration (`ageom/orchestrator.py`):
   - Today pending PDG nodes are processed sequentially.
   - Parallel Hunter runs should share successful declaration patterns, failed candidate fingerprints, and namespace hints in the same orchestration round.

2. Hunter ranking/reformulation/analyze prompts (`ageom/hunter/nodes.py`):
   - Repeated failures and near-duplicate query expansions are common.
   - Prompt-time retrieval of shared context can reduce redundant bad searches.

3. Ingester per-atom loops (`ageom/ingester/chunker.py`, `ageom/ingester/emitter.py`):
   - Recursive decomposition and conceptual abstraction process atoms one by one.
   - Shared naming conventions, conceptual glossary entries, and witness-shape heuristics should be reused.

4. Architect + Principal forked runs (`ageom/architect/*`, `ageom/principal/graph.py`):
   - Forks can repeat rejected decomposition motifs.
   - Reusing prior critique and accepted subgraph patterns reduces repeated exploration.

5. Ingestion reuse (`ageom/ingester/graph.py`):
   - The same source/class often gets ingested again.
   - A content-addressed cache should return the existing `IngestionBundle` when source + settings are unchanged.

## Architecture

1. Introduce a shared context abstraction:
   - `SharedContextStore` protocol:
     - `put(namespace, text, metadata)`
     - `recent(namespace, limit)`
     - `search(namespace, query, limit)`
   - Start with in-memory implementation for local runs/tests.
   - Add persistent backends later (LangGraph store, Redis, Postgres, Memgraph bridge).

2. Namespacing strategy:
   - Run-scoped: `run/{run_id}/hunter/*`, `run/{run_id}/ingester/*`
   - Repo-scoped durable memory: `repo/{repo_hash}/...`
   - Keep run-scoped memory high-signal and short-lived; promote only high-confidence items to durable scope.

3. Prompt integration:
   - For relevant prompts, prepend a compact “Shared Context” block from store hits.
   - Enforce strict char/token budgets to avoid context bloat.

4. Parallelization:
   - Add bounded concurrency for Hunter node matching in orchestrator.
   - Reuse same shared context namespace across concurrently running nodes.

5. Re-ingestion avoidance:
   - Compute `content_key = hash(source_text + class_name + ingest_config_version)`.
   - If key exists in durable cache, return stored `IngestionBundle` and skip full ingest.

## Rollout Phases

### Phase 1 (now)

1. Add `SharedContextStore` + in-memory backend.
2. Wire Hunter to read/write shared context for ranking/reformulation/failure analysis.
3. Parallelize orchestration Hunter calls with configurable concurrency.

Implementation status:
- Completed in code (`ageom/shared_context.py`, Hunter nodes/graph/orchestrator wiring).

### Phase 2

1. Add persistent backend for shared context.
2. Add ingestion content-addressed cache.
3. Add metrics:
   - context hit rate
   - duplicate prompt suppression rate
   - token/latency deltas
   - match success deltas

Implementation status:
- In progress:
  - Ingestion content-addressed cache implemented (`ageom/ingester/cache.py`, `IngesterAgent` cache hit/save path).
  - Ingester prompt-loop shared context + bounded parallelism implemented (`chunker.py`, `emitter.py`).
- Completed:
  - Postgres-backed shared-context store added and wired via existing `AGEOM_POSTGRES_URI`.
  - Shared-context metrics added (hit/miss rate, latency, write counts, injected block/chars) with CLI summaries.

### Phase 3

1. Extend context sharing to Architect/Ingester/Synthesizer loops.
2. Add promotion and eviction policies (confidence/age/frequency based).
3. Add optional provenance labels for explainable context usage.

Implementation status:
- Completed:
  - Architect shared context wired into strategy/decompose/critique prompts.
  - Principal optimization reuses a single Architect shared-context namespace across trials/forks.
  - Synthesizer repair/tactic loops now read/write shared context.
  - Promotion policy: high-confidence records can be promoted into repo-scoped namespaces.
  - Eviction policy: Postgres backend prunes by TTL and keeps top records by confidence/frequency/recency.
  - Provenance labels are injected into shared-context blocks for explainability.

## Open Source Options

1. LangGraph memory/store:
   - Best fit for this codebase (already using LangGraph and checkpointers).
   - Use for run-thread state + long-term store integration.

2. LangMem:
   - Purpose-built memory layer in LangChain/LangGraph ecosystem.
   - Useful when you want richer memory workflows quickly.

3. Mem0:
   - Dedicated memory stack for agent systems with OSS support.
   - Useful for production-grade memory retrieval and management.

4. Letta / AG2 memory layers:
   - Good alternatives if architecture migrates toward agent-framework-native orchestration.

## Risks and Guardrails

1. Context poisoning:
   - Only persist verified/high-confidence facts.
   - Keep failure notes scoped and time-bounded.

2. Prompt bloat:
   - Hard size caps and ranking for injected context.

3. Cross-task leakage:
   - Strict run/repo namespacing and opt-in promotion.

4. Race conditions in parallel writes:
   - Use append-only writes with immutable entries and deterministic read ranking.
