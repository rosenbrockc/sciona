# MAP Combinator — Implementation Plan

## Overview

The MAP combinator is a new higher-order CDG pattern that expresses
"apply this subgraph to each slice of the input."  It is the fan-out
dual of FIXED_POINT (which expresses "apply this subgraph repeatedly
until convergence").

**Motivating use case:** Sliding-window signal processing — a window
atom produces overlapping slices, and the body subgraph runs once per
window.

**Design principle:** Mirror every FIXED_POINT integration point with a
MAP equivalent.  FIXED_POINT already paved the road in assembler,
toposort, prescreen, and backprop.

---

## Architecture

### Node structure

A MAP node is an `AlgorithmicNode` with:

```
concept_type = ConceptType.MAP_OVER
children = [body_node_1, body_node_2, ...]   # body subgraph IDs
map_window_size > 0                           # window length (samples)
map_hop_size > 0                              # hop between windows
```

The body subgraph is a DAG of child nodes (same pattern as FIXED_POINT
`children`).  Each body invocation receives one window slice as input
and produces one result.  The MAP node collects results into an ordered
sequence.

### Skeleton topology

```
MAP Root
  ├── Window Slicer        (ConceptType.MAP_OVER)
  ├── Body Init            (ConceptType.STATE_INIT)
  ├── Body Process         (ConceptType.CUSTOM)
  └── Collect Results      (ConceptType.CUSTOM)
```

Edges: `Window Slicer → Body Init → Body Process → Collect Results`

The root node's `children` field lists `[body_init, body_process, collect_results]`.
`Window Slicer` is NOT a child — it feeds the body from outside.

---

## Step 1: Add ConceptType.MAP_OVER to models.py

**File:** `sciona/architect/models.py`

Insert after `FIXED_POINT = "fixed_point"` (line 58):

```python
MAP_OVER = "map_over"
```

Add new fields to `AlgorithmicNode` after the `fixed_point_*` fields (lines 116–117):

```python
map_window_size: int = 0    # 0 means not a MAP node; >0 = window length in samples
map_hop_size: int = 0       # 0 means not a MAP node; >0 = hop between windows
```

**Tests to update:**
- `tests/test_architect_models.py` `test_expected_members` — add `"map_over"` to expected set

---

## Step 2: Skeleton template

**File:** `sciona/architect/skeletons.py`

Add `_build_map_over()` builder.  Pattern mirrors `_build_fixed_point()`.

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

Register in `SKELETON_TEMPLATES`:
```python
ConceptType.MAP_OVER: _build_map_over(),
```

Add alias entries:
```python
"sliding_window": SKELETON_TEMPLATES[ConceptType.MAP_OVER],
"map_over": SKELETON_TEMPLATES[ConceptType.MAP_OVER],
```

**Tests to update:**
- `tests/test_skeletons.py` `test_all_templates_present` — add `ConceptType.MAP_OVER`
- `tests/test_skeletons.py` `_ALLOWED_HETEROGENEOUS` — add:
  ```python
  ConceptType.MAP_OVER: {
      ConceptType.MAP_OVER,
      ConceptType.STATE_INIT,
      ConceptType.CUSTOM,
  },
  ```
- `tests/test_dsp_integration.py` `test_total_skeleton_count` — 29 → 30

---

## Step 3: Topological sort — `toposort.py`

**File:** `sciona/synthesizer/toposort.py`

### 3a. Update `_VALID_CYCLE_TYPES`

MAP bodies are DAGs (no cycles), so no change needed to cycle validation.
However, `toposort_with_fixed_points` must also handle MAP_OVER nodes.

### 3b. Generalize `toposort_with_fixed_points` → `toposort_with_combinators`

Rename the function (keep old name as alias for backwards compat).  Add
MAP_OVER detection alongside FIXED_POINT:

```python
_COMBINATOR_TYPES = {ConceptType.FIXED_POINT, ConceptType.MAP_OVER}

def toposort_with_combinators(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> tuple[list[str], dict[str, list[str]]]:
    """Topological sort treating FIXED_POINT and MAP_OVER subtrees as opaque."""
    node_map = {n.node_id: n for n in nodes}

    # Identify combinator nodes and their children
    combinator_nodes: dict[str, AlgorithmicNode] = {}
    combinator_children: dict[str, set[str]] = {}
    for n in nodes:
        if n.concept_type in _COMBINATOR_TYPES:
            combinator_nodes[n.node_id] = n
            combinator_children[n.node_id] = set(n.children) if n.children else set()

    # ... rest identical to existing logic with s/fp/combinator/ naming
```

Add backwards-compat alias:
```python
toposort_with_fixed_points = toposort_with_combinators
```

Return type is unchanged: `(top_level_order, combinator_bodies)`.

---

## Step 4: Assembler — `assembler.py`

**File:** `sciona/synthesizer/assembler.py`

### 4a. Detection (mirror lines 340–354)

After the existing `fp_node_ids` detection block, add MAP_OVER detection:

```python
map_node_ids: set[str] = set()
for n in cdg.nodes:
    if n.concept_type == ConceptType.MAP_OVER:
        map_node_ids.add(n.node_id)
```

### 4b. Python emission — `_emit_map_over_python()`

New method, mirrors `_emit_fixed_point_python()`:

```python
def _emit_map_over_python(
    self,
    map_node: AlgorithmicNode,
    body_units: list[AssemblyUnit],
    glue_edges: list[GlueEdge],
) -> list[str]:
    """Emit a Python for-loop over sliding windows for a MAP_OVER node."""
    window_size = getattr(map_node, "map_window_size", 0) or 1024
    hop_size = getattr(map_node, "map_hop_size", 0) or window_size

    lines: list[str] = []
    mname = sanitize_name(map_node.name)
    lines.append(f"    # --- MAP over windows: {map_node.name} ---")
    lines.append(f"    _map_window_{mname} = {window_size}")
    lines.append(f"    _map_hop_{mname} = {hop_size}")
    lines.append(f"    _map_results_{mname} = []")
    lines.append(f"    for _win_start_{mname} in range(0, len(signal) - _map_window_{mname} + 1, _map_hop_{mname}):")
    lines.append(f"        _window_{mname} = signal[_win_start_{mname}:_win_start_{mname} + _map_window_{mname}]")

    # Emit body units inside the for-loop
    for unit in body_units:
        sname = sanitize_name(unit.name)
        args = _resolve_body_args(unit, body_units, glue_edges)
        args_str = ", ".join(args)
        lines.append(f"        {sname}_result = {sname}({args_str})")

    if body_units:
        last = sanitize_name(body_units[-1].name)
        lines.append(f"        _map_results_{mname}.append({last}_result)")

    lines.append(f"    return _map_results_{mname}")
    return lines
```

### 4c. Lean4 emission — `_emit_map_over_lean4()`

```python
def _emit_map_over_lean4(self, map_node: AlgorithmicNode) -> list[str]:
    """Emit Lean 4 List.map-based stub for a MAP_OVER node."""
    lines: list[str] = []
    lines.append(f"  -- MAP over windows: {map_node.name}")
    lines.append("  -- TODO: List.map-based proof obligation")
    lines.append("  sorry")
    return lines
```

### 4d. Coq emission — `_emit_map_over_coq()`

```python
def _emit_map_over_coq(self, map_node: AlgorithmicNode) -> list[str]:
    """Emit Coq map_over stub."""
    lines: list[str] = []
    lines.append(f"  (* MAP over windows: {map_node.name} *)")
    lines.append("  Admitted.")
    return lines
```

### 4e. Integrate into emit blocks

Mirror the FIXED_POINT emit blocks (lines 775–786, 945–958, 1047–1058)
with MAP_OVER equivalents.  The pattern is identical: check
`concept_type == ConceptType.MAP_OVER`, look up body in
`combinator_bodies`, call the appropriate `_emit_map_over_*` method.

---

## Step 5: Prescreen — `prescreen.py`

**File:** `sciona/clearinghouse/prescreen.py`

### 5a. Rename parameter for generality

Rename `fixed_point_node_ids` → `combinator_node_ids` in both
`_check_structure()` and `prescreen()`.  Keep old kwarg as alias for
backwards compat.

### 5b. MAP_OVER bodies are acyclic

No cycle exemption needed for MAP_OVER.  The existing cycle check
already handles this correctly — MAP bodies don't form cycles.  Just
ensure `combinator_node_ids` includes MAP_OVER children so they aren't
flagged as disconnected.

---

## Step 6: Backprop — `backprop.py`

**File:** `sciona/principal/backprop.py`

### 6a. Window-count credit scaling

Mirror the FIXED_POINT convergence check (lines 256–270).  For MAP_OVER
nodes, scale credit by number of windows processed:

```python
for n in cdg.nodes:
    if (
        n.concept_type == ConceptType.MAP_OVER
        and n.map_window_size > 0
        and n.map_hop_size > 0
    ):
        # Estimate window count from signal length if available
        if sim_report.signal_length > 0:
            n_windows = (sim_report.signal_length - n.map_window_size) // n.map_hop_size + 1
            if n_windows > 100:
                for child_id in n.children or []:
                    child_name = node_names.get(child_id, child_id)
                    add(
                        child_name,
                        1.2,
                        f"is in a high-window-count MAP body ({n_windows} windows)",
                    )
```

### 6b. Add `signal_length` to SimReport if not present

Check whether `SimReport` already has a `signal_length` field.  If not,
add `signal_length: int = 0`.

---

## Step 7: Expansion rules (optional — no rules for MAP itself)

MAP_OVER is a structural combinator, not an algorithmic domain.  Like
FIXED_POINT, it has **no ExpansionRuleSet**.  Expansion rules apply to
the body subgraph nodes, not the MAP node itself.

---

## Step 8: Tests

**File:** `tests/test_map_combinator.py` — create

### Test categories:

1. **Skeleton tests** — MAP_OVER template exists, has correct node/edge count,
   instantiation produces fresh IDs
2. **Toposort tests** — MAP body sorted independently, top-level treats MAP
   as atomic
3. **Assembler tests** — Python emission produces for-loop with correct
   window/hop parameters
4. **Prescreen tests** — MAP bodies don't trigger cycle rejection
5. **Backprop tests** — High window count triggers credit scaling
6. **Model tests** — `map_window_size` and `map_hop_size` fields serialize
   correctly

---

## Files Modified/Created Summary

| File | Action |
|---|---|
| `sciona/architect/models.py` | **Modify** — add `MAP_OVER` to ConceptType, add `map_window_size`, `map_hop_size` fields |
| `sciona/architect/skeletons.py` | **Modify** — add `_build_map_over()`, register in SKELETON_TEMPLATES |
| `sciona/synthesizer/toposort.py` | **Modify** — generalize to handle MAP_OVER alongside FIXED_POINT |
| `sciona/synthesizer/assembler.py` | **Modify** — add `_emit_map_over_python/lean4/coq()`, integrate into emit blocks |
| `sciona/clearinghouse/prescreen.py` | **Modify** — include MAP_OVER children in combinator node set |
| `sciona/principal/backprop.py` | **Modify** — add window-count credit scaling for MAP_OVER |
| `tests/test_map_combinator.py` | **Create** — comprehensive MAP combinator tests |
| `tests/test_architect_models.py` | **Modify** — add `"map_over"` to expected members |
| `tests/test_skeletons.py` | **Modify** — add MAP_OVER to expected templates + allowed heterogeneous |
| `tests/test_dsp_integration.py` | **Modify** — update skeleton count 29 → 30 |

**Total: 1 new file, 9 modified files**

---

## Verification

```bash
python -m pytest tests/test_map_combinator.py -v
python -m pytest tests/test_skeletons.py -v
python -m pytest tests/test_architect_models.py -v
python -m pytest tests/ -x --tb=short \
    --ignore=tests/test_profile_varset.py \
    --ignore=tests/test_rapid_mode.py \
    --ignore=tests/test_receipt.py \
    --ignore=tests/test_e2e_principal_hodges.py
```
