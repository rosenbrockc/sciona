# MAP Combinator — Phase 1: Model & Skeleton

## Goal

Establish the MAP_OVER data model and skeleton template. After this phase,
`ConceptType.MAP_OVER` exists, `AlgorithmicNode` has `map_window_size`/`map_hop_size`
fields, the skeleton template is registered, and all model/skeleton tests pass.

No runtime changes (assembler, toposort, prescreen, backprop) in this phase.

---

## Prerequisites

- None (this is the first MAP combinator phase)

## Depends on

- Nothing — this phase is self-contained

---

## Changes

### 1. `sciona/architect/models.py`

**Add enum member.** Insert after `FIXED_POINT = "fixed_point"` (line 58):

```python
MAP_OVER = "map_over"
```

**Add node fields.** Insert after `fixed_point_convergence_field` (line 117):

```python
map_window_size: int = 0    # 0 means not a MAP node; >0 = window length in samples
map_hop_size: int = 0       # 0 means not a MAP node; >0 = hop between windows
```

### 2. `sciona/architect/skeletons.py`

**Add builder function.** Place near `_build_fixed_point()` (around line 1877).
The MAP skeleton has 5 nodes and 3 edges:

```python
def _build_map_over() -> SkeletonGraph:
    """MAP combinator: apply a body subgraph to each window/slice."""
    root = _node(
        "MAP Root",
        "Top-level MAP combinator node",
        ConceptType.MAP_OVER,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="results", type_desc="list[any]")],
        depth=1,
    )
    root = root.model_copy(
        update={"map_window_size": 1024, "map_hop_size": 512}
    )

    window_slicer = _node(
        "Window Slicer",
        "Produce overlapping windows from the input signal",
        ConceptType.MAP_OVER,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="window", type_desc="np.ndarray")],
    )
    body_init = _node(
        "Body Init",
        "Initialize per-window processing state",
        ConceptType.STATE_INIT,
        inputs=[IOSpec(name="window", type_desc="np.ndarray")],
        outputs=[IOSpec(name="state", type_desc="any")],
    )
    body_process = _node(
        "Body Process",
        "Process a single window",
        ConceptType.CUSTOM,
        inputs=[IOSpec(name="state", type_desc="any")],
        outputs=[IOSpec(name="result", type_desc="any")],
    )
    collect_results = _node(
        "Collect Results",
        "Aggregate per-window results into final output",
        ConceptType.CUSTOM,
        inputs=[IOSpec(name="result", type_desc="any")],
        outputs=[IOSpec(name="results", type_desc="list[any]")],
    )

    edges = [
        _edge(window_slicer, body_init, "window", "window", "np.ndarray"),
        _edge(body_init, body_process, "state", "state", "any"),
        _edge(body_process, collect_results, "result", "result", "any"),
    ]

    return SkeletonGraph(
        paradigm=ConceptType.MAP_OVER,
        name="MAP Over",
        description=(
            "MAP combinator: slice input into windows, apply a body "
            "subgraph to each window, collect results."
        ),
        template_nodes=[root, window_slicer, body_init, body_process, collect_results],
        template_edges=edges,
        variants=["sliding_window", "chunked_map", "strided_apply"],
    )
```

**Register in `SKELETON_TEMPLATES` dict:**

```python
ConceptType.MAP_OVER: _build_map_over(),
```

**Add alias entries** in the `SKELETON_ALIASES` dict (or wherever aliases are defined):

```python
"sliding_window": SKELETON_TEMPLATES[ConceptType.MAP_OVER],
"map_over": SKELETON_TEMPLATES[ConceptType.MAP_OVER],
```

### 3. Test updates

**`tests/test_architect_models.py`** — `test_expected_members`:
Add `"map_over"` to the `expected` set.

**`tests/test_skeletons.py`** — `test_all_templates_present`:
Add `ConceptType.MAP_OVER` to the `expected` set.

**`tests/test_skeletons.py`** — `_ALLOWED_HETEROGENEOUS`:
Add entry:
```python
ConceptType.MAP_OVER: {
    ConceptType.MAP_OVER,
    ConceptType.STATE_INIT,
    ConceptType.CUSTOM,
},
```

**`tests/test_dsp_integration.py`** — `test_total_skeleton_count`:
Update count from 29 → 30.

---

## Verification

```bash
python -m pytest tests/test_architect_models.py tests/test_skeletons.py tests/test_dsp_integration.py -v
```

All existing tests must continue to pass:

```bash
python -m pytest tests/ -x --tb=short \
    --ignore=tests/test_profile_varset.py \
    --ignore=tests/test_rapid_mode.py \
    --ignore=tests/test_receipt.py \
    --ignore=tests/test_e2e_principal_hodges.py
```

---

## Files Summary

| File | Action |
|---|---|
| `sciona/architect/models.py` | **Modify** — add `MAP_OVER` enum, add 2 fields to `AlgorithmicNode` |
| `sciona/architect/skeletons.py` | **Modify** — add `_build_map_over()`, register + aliases |
| `tests/test_architect_models.py` | **Modify** — add `"map_over"` to expected members |
| `tests/test_skeletons.py` | **Modify** — add MAP_OVER to expected + allowed heterogeneous |
| `tests/test_dsp_integration.py` | **Modify** — update skeleton count 29 → 30 |

**Total: 0 new files, 5 modified files**
