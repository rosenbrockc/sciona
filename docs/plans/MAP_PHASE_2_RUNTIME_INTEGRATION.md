# MAP Combinator — Phase 2: Runtime Integration

## Goal

Wire MAP_OVER into the four runtime subsystems that FIXED_POINT already
touches: toposort, assembler, prescreen, and backprop. After this phase,
a CDG containing MAP_OVER nodes can be topologically sorted, emitted as
Python/Lean4/Coq code, pre-screened, and credit-scored.

Write a comprehensive `tests/test_map_combinator.py` covering all runtime
behaviors.

---

## Prerequisites

- MAP Phase 1 complete (ConceptType.MAP_OVER exists, skeleton registered,
  `map_window_size`/`map_hop_size` fields on AlgorithmicNode)

---

## Changes

### 1. `sciona/synthesizer/toposort.py` — Generalize combinator handling

**Goal:** `toposort_with_fixed_points` must also treat MAP_OVER subtrees as opaque.

**Approach:** Generalize the function to handle any combinator type. The existing
function identifies FIXED_POINT nodes by checking `n.concept_type == ConceptType.FIXED_POINT`.
Broaden this to check against a set:

```python
_COMBINATOR_TYPES = {ConceptType.FIXED_POINT, ConceptType.MAP_OVER}
```

Rename the internal logic to use "combinator" naming but keep the public function
name `toposort_with_fixed_points` unchanged for backwards compatibility (it now
handles both types). Alternatively, create `toposort_with_combinators` as the
implementation and alias:

```python
toposort_with_fixed_points = toposort_with_combinators
```

The body logic is identical — identify combinator nodes via `n.concept_type in _COMBINATOR_TYPES`,
collect their `children`, sort each body independently, then sort the top level
with combinator nodes treated as atomic.

**Also update `_VALID_CYCLE_TYPES`:** MAP_OVER bodies are acyclic, but the constant
is used for cycle _tolerance_. No need to add MAP_OVER there since MAP bodies
should never form cycles. Leave `_VALID_CYCLE_TYPES` unchanged.

### 2. `sciona/synthesizer/assembler.py` — MAP_OVER code emission

**2a. Detection.** After the existing FIXED_POINT detection block (around line 340–354),
add MAP_OVER detection:

```python
map_node_ids: set[str] = set()
for n in cdg.nodes:
    if n.concept_type == ConceptType.MAP_OVER:
        map_node_ids.add(n.node_id)
```

**2b. New method `_emit_map_over_python()`.** Mirror `_emit_fixed_point_python()` (lines 642–692).
Key differences:
- Emits a `for` loop instead of a while loop
- Uses `map_window_size` and `map_hop_size` instead of `fixed_point_max_iterations`
- Collects results into a list instead of checking convergence
- No convergence break — always iterates over all windows

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
    lines.append(
        f"    for _win_start_{mname} in range("
        f"0, len(signal) - _map_window_{mname} + 1, _map_hop_{mname}):"
    )
    lines.append(
        f"        _window_{mname} = signal["
        f"_win_start_{mname}:_win_start_{mname} + _map_window_{mname}]"
    )

    # Emit body units inside the for-loop (8-space indent)
    for unit in body_units:
        sname = sanitize_name(unit.name)
        args: list[str] = []
        for inp in unit.inputs:
            edge_for_inp = next(
                (
                    e
                    for e in glue_edges
                    if e.target_id == unit.node_id and e.input_name == inp.name
                ),
                None,
            )
            if edge_for_inp:
                src_unit = next(
                    (u for u in body_units if u.node_id == edge_for_inp.source_id),
                    None,
                )
                if src_unit:
                    args.append(sanitize_name(src_unit.name) + "_result")
                else:
                    args.append(inp.name)
            else:
                args.append(inp.name)
        args_str = ", ".join(args)
        lines.append(f"        {sname}_result = {sname}({args_str})")

    if body_units:
        last = sanitize_name(body_units[-1].name)
        lines.append(f"        _map_results_{mname}.append({last}_result)")

    lines.append(f"    return _map_results_{mname}")
    return lines
```

**2c. New method `_emit_map_over_lean4()`.** Mirror `_emit_fixed_point_lean4()`:

```python
def _emit_map_over_lean4(self, map_node: AlgorithmicNode) -> list[str]:
    lines: list[str] = []
    lines.append(f"  -- MAP over windows: {map_node.name}")
    lines.append("  -- TODO: List.map-based proof obligation")
    lines.append("  sorry")
    return lines
```

**2d. New method `_emit_map_over_coq()`.** Mirror `_emit_fixed_point_coq()`:

```python
def _emit_map_over_coq(self, map_node: AlgorithmicNode) -> list[str]:
    mname = sanitize_name(map_node.name)
    lines: list[str] = []
    lines.append(f"  (* MAP over windows: {map_node.name} *)")
    lines.append(f"  (* map {mname}_body (windows signal) *)")
    lines.append("  Admitted.")
    return lines
```

**2e. Integrate into emission blocks.** In each of the three emit sections
(Python ~line 945, Lean4 ~line 775, Coq ~line 1047), add a block that mirrors the
FIXED_POINT block but checks `concept_type == ConceptType.MAP_OVER` and calls the
corresponding `_emit_map_over_*` method. Use the same `combinator_bodies` dict
returned by the generalized toposort.

### 3. `sciona/clearinghouse/prescreen.py` — Combinator node handling

**Goal:** MAP_OVER children must not be flagged as disconnected or structurally
invalid. MAP bodies are acyclic — no cycle exemption needed (unlike FIXED_POINT).

**Approach:** The `fixed_point_node_ids` parameter already accepts a `frozenset[str]`
of node IDs whose cycles are tolerated. For MAP_OVER, we need the same set to
include MAP body children so they aren't treated as orphaned top-level nodes.

In the caller that invokes `prescreen()`, collect both FIXED_POINT and MAP_OVER
children into the `fixed_point_node_ids` parameter. No changes needed inside
`prescreen.py` itself if the caller provides the right set. Check who calls
`prescreen()` and update those call sites.

If `prescreen.py` does its own structural validation beyond cycles, ensure MAP
body nodes (which are children of a MAP_OVER node) aren't flagged. Read the
`_check_structure` function carefully to determine if any changes are needed.

### 4. `sciona/principal/backprop.py` — Window-count credit scaling

**Goal:** When a MAP_OVER body processes many windows, scale credit for body nodes.

**4a. Check SimReport for `signal_length` field.** Read the SimReport model
(likely in `sciona/principal/models.py` or similar). If it doesn't have
`signal_length: int = 0`, add it.

**4b. Add MAP_OVER credit scaling.** Mirror the FIXED_POINT non-convergence
detection block (lines 256–270). Place after that block:

```python
# High-window-count MAP body signal: if the MAP body processes
# >100 windows, flag body nodes for cost awareness.
for n in cdg.nodes:
    if (
        n.concept_type == ConceptType.MAP_OVER
        and n.map_window_size > 0
        and n.map_hop_size > 0
    ):
        if sim_report.signal_length > 0:
            n_windows = (
                (sim_report.signal_length - n.map_window_size) // n.map_hop_size + 1
            )
            if n_windows > 100:
                for child_id in n.children or []:
                    child_name = node_names.get(child_id, child_id)
                    add(
                        child_name,
                        1.2,
                        f"is in a high-window-count MAP body ({n_windows} windows)",
                    )
```

### 5. `tests/test_map_combinator.py` — Create comprehensive test file

Test categories:

**Model tests:**
- `map_window_size` and `map_hop_size` default to 0
- Fields serialize/deserialize correctly via `model_dump()`/`model_validate()`

**Skeleton tests:**
- MAP_OVER skeleton exists in `SKELETON_TEMPLATES`
- Has 5 template nodes, 3 template edges
- `instantiate_skeleton` produces fresh IDs across calls
- Root node has `map_window_size=1024`, `map_hop_size=512`

**Toposort tests:**
- Build a CDG with a MAP_OVER root + 3 body children
- `toposort_with_fixed_points` (or `toposort_with_combinators`) returns the
  MAP node in top-level order and body nodes in `combinator_bodies`
- Body nodes are sorted independently

**Assembler tests:**
- Build a minimal CDG with MAP_OVER node + body nodes
- Call the Python assembler and verify emitted code contains:
  - `for _win_start_` loop
  - `_map_window_` and `_map_hop_` variables
  - `_map_results_` list and `.append()`
- Verify Lean4 output contains `sorry` and MAP comment
- Verify Coq output contains `Admitted.` and MAP comment

**Prescreen tests:**
- A CDG with MAP_OVER children should not be rejected for cycles
- MAP body children should not be flagged as disconnected

**Backprop tests:**
- With `sim_report.signal_length` > 0 and many windows, verify credit
  scaling is applied to MAP body children
- With `signal_length` = 0, verify no scaling applied

---

## Verification

```bash
python -m pytest tests/test_map_combinator.py -v
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
| `sciona/synthesizer/toposort.py` | **Modify** — generalize to handle MAP_OVER alongside FIXED_POINT |
| `sciona/synthesizer/assembler.py` | **Modify** — add 3 `_emit_map_over_*` methods + integrate into emit blocks |
| `sciona/clearinghouse/prescreen.py` | **Modify** — ensure MAP_OVER children handled correctly (may be caller-side) |
| `sciona/principal/backprop.py` | **Modify** — add window-count credit scaling |
| `tests/test_map_combinator.py` | **Create** — comprehensive MAP combinator runtime tests |

**Total: 1 new file, 4 modified files**

---

## Reference

- FIXED_POINT toposort: `sciona/synthesizer/toposort.py` lines 102–170
- FIXED_POINT assembler: `sciona/synthesizer/assembler.py` lines 340–354, 642–711, 775–786, 945–958, 1047–1058
- FIXED_POINT prescreen: `sciona/clearinghouse/prescreen.py` lines 142–193, 259–305
- FIXED_POINT backprop: `sciona/principal/backprop.py` lines 256–270, 309–330
