# Baseline Analysis — Phase 4: Multi-Component Instantiation Helper

## Goal

Add a convenience function `instantiate_baseline_multi_component()` that
creates N-component baseline analysis CDGs from the single-component skeleton
template. This enables the dynamic fan-out pattern where multiple independent
signal processing chains feed into a shared Combine node.

---

## Prerequisites

- Baseline Phase 1–3 complete (skeleton, atoms, expansion rules all working)

---

## Design

The single-component skeleton is:

```
Acquire → Preprocess → Windowed Analysis → Fit →
Output Transform → Normalize → Combine → Regionize
```

For N components, the instantiated CDG should be:

```
                    ┌─ Preprocess_1 → WA_1 → Fit_1 → OT_1 → Norm_1 ─┐
Acquire ───────────┼─ Preprocess_2 → WA_2 → Fit_2 → OT_2 → Norm_2 ─┼─→ Combine → Regionize
                    └─ Preprocess_N → WA_N → Fit_N → OT_N → Norm_N ─┘
```

Acquire, Combine, and Regionize are shared. The per-component chain
(Preprocess through Normalize) is duplicated N times with unique node IDs.

---

## Changes

### 1. `sciona/architect/skeletons.py` — Add helper function

```python
def instantiate_baseline_multi_component(
    skeleton: SkeletonGraph,
    goal: str,
    n_components: int,
    *,
    parent_id: str | None = None,
    base_depth: int = 0,
) -> tuple[list[AlgorithmicNode], list[DependencyEdge]]:
    """Instantiate a multi-component baseline analysis CDG.

    Creates N copies of the per-component pipeline (Preprocess through
    Normalize) and wires all into a shared Acquire, Combine, and Regionize.

    Args:
        skeleton: The BASELINE_ANALYSIS skeleton template.
        goal: Goal description to embed in node descriptions.
        n_components: Number of parallel component pipelines.
        parent_id: Optional parent node ID for all created nodes.
        base_depth: Depth offset for all created nodes.

    Returns:
        (nodes, edges) — the instantiated CDG.
    """
```

**Implementation approach:**

1. Identify the shared nodes by name: "Acquire Data", "Combine", "Regionize"
2. Identify the per-component chain by name: "Preprocess", "Windowed Analysis",
   "Fit", "Output Transform", "Normalize"
3. Create one copy of each shared node (with fresh UUIDs)
4. For each component `i` in `range(n_components)`:
   - Create a copy of each chain node with fresh UUIDs and names suffixed
     with ` (Component {i+1})` — e.g., "Preprocess (Component 1)"
   - Create internal chain edges
   - Create edge from Acquire → Preprocess_i
   - Create edge from Normalize_i → Combine
5. Create edge from Combine → Regionize
6. Apply `parent_id` and `base_depth` offsets to all nodes

### 2. Tests

**File:** `tests/test_baseline_multi_component.py` — Create

Test cases:

1. **Single component** — `n_components=1` produces same structure as
   `instantiate_skeleton()` (same node count, same edge count)

2. **Two components** — `n_components=2`:
   - 3 shared nodes + 2×5 chain nodes = 13 total nodes
   - 2×4 internal chain edges + 2 Acquire→Preprocess + 2 Normalize→Combine
     + 1 Combine→Regionize = 13 total edges
   - All node IDs unique
   - All edges reference valid node IDs

3. **Three components** — `n_components=3`:
   - 3 + 3×5 = 18 nodes, verify count
   - All edge sources/targets valid

4. **Fresh IDs across calls** — Two calls with same `n_components` produce
   disjoint node ID sets

5. **Parent ID and depth** — All nodes get the specified `parent_id` and
   `depth >= base_depth + 1`

6. **Component naming** — Each component chain has nodes with "(Component N)"
   suffix

7. **Goal in descriptions** — Goal string appears in all node descriptions

---

## Verification

```bash
python -m pytest tests/test_baseline_multi_component.py -v

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
| `sciona/architect/skeletons.py` | **Modify** — add `instantiate_baseline_multi_component()` |
| `tests/test_baseline_multi_component.py` | **Create** — multi-component instantiation tests |

**Total: 1 new file, 1 modified file**

---

## Reference

- `instantiate_skeleton()` in `sciona/architect/skeletons.py` — the single-instance
  pattern this helper extends
- Baseline skeleton topology: see `_build_baseline_analysis()` in `skeletons.py`
