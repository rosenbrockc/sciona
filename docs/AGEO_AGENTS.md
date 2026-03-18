# Agent Prompt Call Sites

In `ageom/`, these are the concrete places where agent logic calls the LLM with explicit prompts.

## Architect agent

- `ageom/architect/nodes.py:278` uses `SELECT_STRATEGY_SYSTEM` + `SELECT_STRATEGY_USER`
- `ageom/architect/nodes.py:432` uses `DECOMPOSE_NODE_SYSTEM` + `DECOMPOSE_NODE_USER`
- `ageom/architect/nodes.py:654` uses `CRITIQUE_SYSTEM` + `CRITIQUE_USER`

## Hunter agent

- `ageom/hunter/nodes.py:162` ranks candidates via `_complete_with_optional_grammar(...)` with `SCORE_CANDIDATES_SYSTEM` + `SCORE_CANDIDATES_USER`
- `ageom/hunter/nodes.py:284` analyzes failures with `ANALYZE_FAILURE_SYSTEM` + `ANALYZE_FAILURE_USER`
- `ageom/hunter/nodes.py:312` reformulates queries with `REFORMULATE_QUERY_SYSTEM` + `REFORMULATE_QUERY_USER`

## Ingester agent (chunking and abstraction)

- `ageom/ingester/chunker.py:308` uses `SEMANTIC_CHUNK_SYSTEM` + `SEMANTIC_CHUNK_USER`
- `ageom/ingester/chunker.py:391` uses `HOIST_STATE_SYSTEM` + `HOIST_STATE_USER`
- `ageom/ingester/chunker.py:681` uses `DECOMPOSE_ATOM_SYSTEM` + `DECOMPOSE_ATOM_USER`
- `ageom/ingester/chunker.py:821` uses `CONCEPTUAL_ABSTRACT_SYSTEM` + `CONCEPTUAL_ABSTRACT_USER`

## Ingester agent (repair loop)

- `ageom/ingester/graph.py:663` uses `FIX_TYPE_ERROR_SYSTEM` + `FIX_TYPE_ERROR_USER`
- `ageom/ingester/graph.py:721` uses `FIX_GHOST_ERROR_SYSTEM` + `FIX_GHOST_ERROR_USER`
- `ageom/ingester/graph.py:786` uses `FIX_MESSAGE_CYCLE_SYSTEM` + `FIX_MESSAGE_CYCLE_USER`

## Ingester emitter

- `ageom/ingester/emitter.py:154` uses `DRAFT_OPAQUE_WITNESS_SYSTEM` + `DRAFT_OPAQUE_WITNESS_USER`

## Synthesizer (repair agent)

- `ageom/synthesizer/repair.py:262` uses `ANALYZE_ERROR_USER` with system prompt chosen from `ANALYZE_ERROR_SYSTEM` or `ANALYZE_ERROR_SYSTEM_PYTHON`
- `ageom/synthesizer/repair.py:328` uses `GENERATE_TACTIC_USER` with system prompt chosen from `GENERATE_TACTIC_SYSTEM` or `GENERATE_IMPLEMENTATION_SYSTEM_PYTHON`

## Orchestrator refinement step

- `ageom/orchestrator.py:91` calls the LLM with inline `system_prompt` and `user_prompt` built just above at `ageom/orchestrator.py:76` and `ageom/orchestrator.py:83`
