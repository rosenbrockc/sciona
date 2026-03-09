# Hodges E2E Test — Implementation State & Debug Notes

## What exists
- `tests/test_retrieval_e2e_hodges.py` — partially working, 7 pass / 4 fail
- All biosppy CDGs upserted into Memgraph as separate repos (`biosppy.ecg_christov`, `biosppy.emg_solnik`, etc.)

## Test file location
`/Users/conrad/personal/ageo-matcher/tests/test_retrieval_e2e_hodges.py`

## What's passing (7 tests)
- **Part 1 (all 5)**: Retrieval against live Memgraph works perfectly
  - `test_layer2_finds_emg_detectors` ✅
  - `test_retrieval_returns_children_and_edges` ✅
  - `test_format_prompt_produces_usable_text` ✅
  - `test_score_ordering_is_descending` ✅
  - `test_exclude_repo_is_respected` ✅
- **Part 3 (2 of 3)**:
  - `test_architect_topo_hash_matches_graph_store_function` ✅
  - `test_topo_hash_differs_from_stored_cdgs` ✅

## What's failing (4 tests) — all same root cause
- `test_decomposition_produces_valid_hodges_cdg`
- `test_retrieval_examples_injected_into_prompt`
- `test_decomposition_without_retrieval_still_works`
- `test_hodges_degree_sequence_is_correct`

## Root cause of failures

The mock LLM's strategy response returns `paradigm: "custom"`. CUSTOM has no skeleton in `SKELETON_TEMPLATES`. When `select_strategy` runs:
1. Root node is created with `status=NodeStatus.DECOMPOSED`
2. No skeleton → no children → `pending = []`
3. `done = len(pending) == 0` → True
4. But `route_after_strategy` returns `"decompose"` (not conjugate)
5. `decompose_node` is called with `current_node_id=""` → "Node not found"
6. Loops through critique/retry 3 times, then advance_node → done

Result: CDG has only 1 root node, 0 children, 0 edges.

## Fix needed

Change the mock LLM strategy response to use a paradigm that HAS a skeleton. Options:

### Option A: Use `SIGNAL_FILTER` paradigm (recommended)
- `ConceptType.SIGNAL_FILTER` has a skeleton with 4 nodes: Design Filter, Validate Stability, Apply Filter, Frequency Response
- These become PENDING and go through decompose_node
- Problem: mock LLM returns same Hodges 6-node decomposition for EVERY skeleton node (it keys on system prompt keywords, not user prompt content)
- Sub-problem: with `_AcceptAllCatalog`, all sub-nodes are marked ATOMIC immediately, so cycle terminates after one decompose per skeleton node
- Result: 4 skeleton nodes × 6 sub-nodes each = 24 sub-nodes. Structural invariants about Hodges won't hold since it's 4 independent decompositions.

### Option B: Make mock LLM return Hodges decomposition only for first call
- Track call count; first decompose call returns Hodges 6-node response, subsequent calls return a minimal 2-node atomic response
- Cleaner but the structural invariants still apply to one skeleton node's children, not the whole CDG

### Option C (simplest, recommended): Key mock LLM on user prompt content
- The user prompt contains `Node to decompose: <name>`
- When the node name matches the root or first skeleton node, return Hodges decomposition
- For all other nodes, return a minimal atomic response
- This way only one subtree has the Hodges structure

### Option D: Skip strategy entirely, test decompose_node directly
- Construct a state manually with the Hodges root as PENDING
- Call `decompose_node` as a standalone function (not through the graph)
- This tests retrieval injection without the strategy/skeleton complexity
- Probably the cleanest approach for testing the retrieval-to-prompt pipeline

## Key data points from Memgraph

### Decomposed intermediate nodes (the useful ones for retrieval):
```
biosppy.emg_solnik.threshold_based_onset_detection: concept=signal_filter, in=5, out=1, topo=9025c0265f9bf41e
biosppy.emg_abbink.detect_onsets_with_rest_aware_thresholds: concept=signal_filter, in=7, out=1, topo=2d17e5dfdc52c9f0
biosppy.emg_bonato.bonato_onset_detection: concept=signal_filter, in=7, out=1, topo=1790e01fabf31e69
biosppy.ecg_christov.christovqrsdetect: concept=signal_filter, in=2, out=1, topo=09c52d4d263ad1ad
biosppy.ecg_hamilton.hamilton_segmentation: concept=signal_transform, in=2, out=1, topo=1cd3f373431bae0f
```

### Query results confirmed working:
- `query_by_structure(concept_type='signal_filter', n_inputs=5, n_outputs=1, ...)` → finds Solnik (exact match)
- `query_by_structure(concept_type='signal_filter', n_inputs=6, n_outputs=1, ...)` → finds Abbink, Solnik, Bonato (±1 tolerance)
- `query_by_structure(concept_type='custom', n_inputs=0, n_outputs=0, min_children=2, ...)` → finds ZZ2018_root (4 children)

### Hodges query node (for Part 1, works correctly):
```python
AlgorithmicNode(
    node_id="hodges_onset_detection",
    parent_id="hodges_root",
    concept_type=ConceptType.SIGNAL_FILTER,
    inputs=[signal, rest_signal, sampling_rate, threshold, active_state_duration],  # 5 inputs
    outputs=[onsets],  # 1 output
    depth=1, status=PENDING,
)
```

### Hodges expected degree sequence:
```
[(0, 2), (1, 1), (1, 1), (1, 2), (2, 0), (2, 1)]
```
From: Baseline(0,2), Remove(1,1), TestStat(2,1), Smooth(1,1), Threshold(1,2), Merge(2,0)

## Current state of test file modifications
- `_hodges_node()` returns intermediate-level node (concept_type=SIGNAL_FILTER, 5 inputs, 1 output) ✅
- `_AcceptAllCatalog` confirms any node as atomic ✅
- `HODGES_DECOMPOSE_RESPONSE` has `is_atomic: True, matched_primitive: "hodges_stub"` ✅
- `HodgesArchitectLLM` strategy returns `paradigm: "custom"` ← NEEDS CHANGE

## Recommended next steps

1. **Adopt Option C or D** for fixing the 4 failing tests
2. If Option D: refactor Part 2 to call `decompose_node` directly with a pre-built state, or build a minimal agent wrapper that skips strategy
3. If Option C: change strategy to return `"signal_filter"` and have the mock LLM check user prompt for node name to decide which response to return
4. For `test_hodges_degree_sequence_is_correct`: the children need to be found via `parent_id` of the decomposed root — ensure the mock flow produces children with correct parent_id
5. Run `pytest tests/test_retrieval_e2e_hodges.py -v` to verify all 11 pass

## Related files modified in the retrieval feature
- `docker-compose.yml` — memgraph-mage image
- `ageom/config.py` — 4 graph_retrieval_* fields
- `ageom/graph_store.py` — topo_hash index + 3 query methods
- `ageom/architect/graph_retrieval.py` — NEW: retriever, data model, prompt formatter, factory
- `ageom/architect/state.py` — graph_retriever field on DecompositionDeps
- `ageom/architect/prompts.py` — {example_decompositions} placeholder
- `ageom/architect/nodes.py` — retriever call in decompose_node
- `ageom/architect/graph.py` — graph_retriever param on DecompositionAgent
- `ageom/cli.py` — wiring in _cmd_decompose and _cmd_run

## Upsert note
`upsert_cdg.py` uses `glob("*cdg*.json")` (non-recursive). Biosppy subdirectories must be upserted individually:
```bash
for dir in ../ageo-atoms/ageoa/biosppy/*/; do
  if [ -f "$dir/cdg.json" ]; then
    name=$(basename "$dir")
    ageom upsert-cdg "$dir" --repo-name "biosppy.$name"
  fi
done
```
Don't use rglob — it causes cascading stale-deletion within the same repo namespace.
