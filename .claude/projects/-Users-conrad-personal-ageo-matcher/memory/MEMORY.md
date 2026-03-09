# AGEO-Matcher Memory

## Project Structure
- `ageom/` — main package (architect, hunter, synthesizer, ingester, graph_store, cli)
- `tests/` — pytest suite; mock LLMs route by system prompt keywords (`"paradigm"`, `"sub-nodes"`, `"critic"`)
- `docker-compose.yml` — memgraph-mage + postgres + visualizer
- Sibling repo: `../ageo-atoms/ageoa/biosppy/` — ingested CDG atoms

## Key Patterns
- LLM mocking: `async def complete(system, user)` that checks `system.lower()` for keywords
- DecompositionAgent flow: `select_strategy` → (optional skeleton) → `decompose_node` → `critique` → `advance_node`
- CUSTOM paradigm has NO skeleton → root is DECOMPOSED with no pending children → done immediately
- Paradigms WITH skeletons (SIGNAL_FILTER, DIVIDE_AND_CONQUER, etc.) create PENDING intermediate nodes
- `_AcceptAllCatalog` trick: override `is_atomic()` to always return True, stops decomposition recursion

## Biosppy CDGs in Memgraph
- Upserted per-subdirectory as `biosppy.<subdir>` repos (not flat `biosppy`)
- 12 repos, 105 atoms, 26 decomposed nodes with topo_hashes
- EMG detectors (Solnik/Abbink/Bonato) are `signal_filter` concept_type with 5-7 inputs
- `upsert_cdg.py` uses non-recursive glob; subdirs need individual upsert calls

## Completed Work
- E2E test `tests/test_retrieval_e2e_hodges.py` — all 11 passing
- Fix: bypassed `select_strategy`/skeleton by calling `decompose_node` directly with pre-built state (Option D)
- Root cause: `_AcceptAllCatalog` marks all skeleton nodes ATOMIC → empty pending queue → done immediately
