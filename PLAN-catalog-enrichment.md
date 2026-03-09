# Catalog Enrichment with De-duplication — Implementation Plan

## Goal

Enrich `PrimitiveCatalog` with primitives from external sources while preventing
semantic duplicates.  The existing catalog only deduplicates by exact name.  Two
primitives with different names but identical semantics (e.g. `butter` from
source A and `design_butterworth_filter` from source B) both enter today and
compete during decomposition, diluting search quality.

This plan adds embedding-based de-duplication, CDG structural de-duplication,
and a reporting mechanism to surface catalog gaps.

---

## Vocabulary

| Term | Definition |
|------|-----------|
| **primitive** | An `AlgorithmicPrimitive` in the catalog |
| **candidate** | A primitive about to be inserted |
| **incumbent** | The existing primitive a candidate is compared against |
| **merge** | Keep the richer primitive, register the other's name as an alias |
| **SkillIndex** | FAISS cosine-similarity index over primitive description embeddings (`ageom/architect/embedder.py`) |
| **topo_hash** | Structural fingerprint of a CDG decomposition subtree (`ageom/graph_store.py:_topo_hash`) |

---

## Task 1: Add `DedupResult` model and similarity helpers

### File: `ageom/architect/catalog.py`

Add a dataclass and a pure function at module level:

```python
from dataclasses import dataclass

@dataclass
class DedupResult:
    """Outcome of comparing a candidate against the catalog."""
    is_duplicate: bool
    incumbent_name: str | None = None  # name of the matched existing primitive
    similarity: float = 0.0           # cosine similarity score
    structural_match: bool = False     # True if category + arity also match
```

Add a static method to `PrimitiveCatalog`:

```python
@staticmethod
def _structural_match(a: AlgorithmicPrimitive, b: AlgorithmicPrimitive) -> bool:
    """Check category equality and IO arity compatibility."""
    if a.category != b.category:
        return False
    a_required_in = len([p for p in a.inputs if p.required])
    b_required_in = len([p for p in b.inputs if p.required])
    if abs(a_required_in - b_required_in) > 1:
        return False
    if abs(len(a.outputs) - len(b.outputs)) > 1:
        return False
    return True
```

### Tests: `tests/test_catalog.py`

Add a `TestDedupResult` class:
- `test_structural_match_same_category_same_arity` -> True
- `test_structural_match_different_category` -> False
- `test_structural_match_arity_off_by_one` -> True
- `test_structural_match_arity_off_by_two` -> False

---

## Task 2: Add `check_duplicate` method to `PrimitiveCatalog`

### File: `ageom/architect/catalog.py`

Add a new method that takes a candidate primitive and an optional SkillIndex:

```python
def check_duplicate(
    self,
    candidate: AlgorithmicPrimitive,
    skill_index: "SkillIndex | None" = None,
    threshold: float = 0.85,
) -> DedupResult:
    """Check if candidate is a semantic duplicate of an existing primitive.

    1. Exact name match -> always duplicate.
    2. If skill_index is provided, query top-1 by embedding similarity.
       If similarity >= threshold AND structural_match -> duplicate.
    3. Otherwise -> not duplicate.
    """
```

Implementation notes:
- Import `SkillIndex` with `TYPE_CHECKING` guard to avoid circular import
  (embedder.py already imports from catalog.py).
- The `SkillIndex.search_by_embedding` method returns
  `list[tuple[Declaration, float]]`.  The `Declaration.name` maps back to
  the primitive name.  Use `self.get(decl.name)` to retrieve the incumbent
  `AlgorithmicPrimitive` for `_structural_match`.
- If the skill_index is None or empty, only the exact-name check runs.

### Tests: `tests/test_catalog.py`

Add a `TestCheckDuplicate` class.  These tests should NOT require torch/FAISS
(mark any that do with `@pytest.mark.slow`):

- `test_exact_name_is_duplicate` — candidate with same name as existing
- `test_different_name_no_index_is_not_duplicate` — no SkillIndex provided
- `test_mock_skill_index_high_similarity_same_category` — mock SkillIndex
  returning similarity=0.9, same category -> duplicate
- `test_mock_skill_index_high_similarity_different_category` — similarity=0.9,
  different category -> NOT duplicate (structural_match fails)
- `test_mock_skill_index_below_threshold` — similarity=0.7 -> not duplicate

For mock: create a tiny helper class in the test that implements
`search_by_embedding(query, k=1)` returning a canned `(Declaration, score)`.

---

## Task 3: Add `add_with_dedup` method to `PrimitiveCatalog`

### File: `ageom/architect/catalog.py`

```python
def add_with_dedup(
    self,
    candidate: AlgorithmicPrimitive,
    skill_index: "SkillIndex | None" = None,
    threshold: float = 0.85,
) -> DedupResult:
    """Add a primitive, merging with existing if duplicate.

    If duplicate:
      - Keep the primitive with richer metadata (longer description,
        more IO specs, has type_signature).  Use _richer() to decide.
      - Register the other's name as an alias.
    If not duplicate:
      - Call self.add(candidate) normally.

    Returns the DedupResult.
    """
```

Add a private helper:

```python
@staticmethod
def _richer(a: AlgorithmicPrimitive, b: AlgorithmicPrimitive) -> AlgorithmicPrimitive:
    """Return the primitive with richer metadata."""
    def _score(p: AlgorithmicPrimitive) -> int:
        s = len(p.description)
        s += len(p.inputs) * 10
        s += len(p.outputs) * 10
        s += 50 if p.type_signature else 0
        return s
    return a if _score(a) >= _score(b) else b
```

When merging: call `self.add(winner)` (which handles category bucket
replacement), then `self.add_alias(loser.name, winner.name)`.

### Tests: `tests/test_catalog.py`

- `test_add_with_dedup_no_duplicate_adds_normally`
- `test_add_with_dedup_merges_and_keeps_richer` — insert A, then B (duplicate
  with richer description).  Assert B is in catalog, A.name is alias for B.
- `test_add_with_dedup_merges_and_keeps_incumbent` — insert A (richer), then
  B (duplicate with shorter description).  Assert A is in catalog, B.name
  is alias for A.
- `test_add_with_dedup_no_index_falls_through` — without SkillIndex, only
  exact-name dedup applies.

---

## Task 4: Wire de-duplication into `seed_catalog_from_sources`

### File: `ageom/architect/source_catalog.py`

Modify `seed_catalog_from_sources` signature to accept an optional SkillIndex
and dedup threshold:

```python
def seed_catalog_from_sources(
    catalog: PrimitiveCatalog,
    *,
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
    include_live_registries: bool = True,
    skill_index: "SkillIndex | None" = None,
    dedup_threshold: float = 0.85,
) -> int:
```

At the two insertion points (line 435 and line 449 in the current file), replace:
```python
catalog.add(primitive)
```
with:
```python
result = catalog.add_with_dedup(primitive, skill_index, dedup_threshold)
```

Adjust the `added` counter: only increment when `result.is_duplicate` is False.

Add alias registration: the existing `for alias in aliases:
catalog.add_alias(alias, primitive.name)` block needs to handle the case where
`result.is_duplicate` is True and the winner has a different name.  After
`add_with_dedup`, the primitive may have been merged under the incumbent's name.
Use the returned `result.incumbent_name` to determine the canonical name:

```python
canonical = result.incumbent_name if result.is_duplicate else primitive.name
for alias in aliases:
    try:
        catalog.add_alias(alias, canonical)
    except KeyError:
        pass  # alias target may have been merged
```

### File: `ageom/architect/embedder.py`

Add a method to `SkillIndex` for incrementally adding a single primitive
(needed so the index stays current as new primitives are added during
seeding):

```python
def add_primitive(self, primitive: AlgorithmicPrimitive) -> None:
    """Add a single primitive to the live index."""
    self._ensure_embedder()
    text = self._primitive_to_text(primitive)
    vec = self._embedder.embed(text)
    decl = Declaration(
        name=primitive.name,
        type_signature=primitive.type_signature,
        docstring=primitive.description,
        source_lib=primitive.source,
        prover=Prover.LEAN4,
    )
    entry = IndexEntry(declaration=decl, embedding=vec, source_text=text)
    if self._store is None:
        self._store = FAISSStore(dim=self._embedder.dim)
    self._store.add([entry])
    idx = len(self._primitives)
    self._primitives.append(primitive)
    self._id_to_primitive[idx] = primitive
```

Then in `seed_catalog_from_sources`, after a successful non-duplicate add:
```python
if skill_index is not None and not result.is_duplicate:
    skill_index.add_primitive(primitive)
```

### Tests: `tests/test_source_catalog.py`

Add a test `test_seed_catalog_dedup_merges_similar_primitives` that:
1. Sets up two sources with semantically identical atoms under different names
2. Provides a mock SkillIndex that returns high similarity
3. Asserts only one primitive in catalog, the other is an alias

---

## Task 5: CDG structural de-duplication in source catalog

### File: `ageom/architect/source_catalog.py`

Modify `_load_atomic_node_index` to also compute topo_hashes for decomposed
parent nodes:

```python
def _load_atomic_node_index(
    source: AtomSource, *, base_dir: Path | None = None
) -> tuple[dict[str, list[_AtomicNodeMeta]], dict[str, str]]:
    """Returns (atom_index, topo_hashes).

    topo_hashes: {topo_hash: source_name} for deduplication across sources.
    """
```

In the CDG JSON parsing loop, for each node with `status != "atomic"` that
has children, compute:
```python
from ageom.graph_store import _topo_hash
hash_val = _topo_hash(all_nodes_dicts, all_edges_dicts, node["node_id"])
```

Return the hash map alongside the atom index.

In `seed_catalog_from_sources`, accumulate `seen_topo_hashes` across sources.
When a CDG subtree from source B has the same topo_hash as source A, log a
debug message and skip its leaf primitives UNLESS they carry a `type_signature`
that no existing primitive in the catalog has (checked via
`catalog.get(name)` returning None or having empty `type_signature`).

### Tests: `tests/test_source_catalog.py`

Add `test_seed_catalog_skips_structurally_duplicate_cdg_subtrees`:
1. Two sources with identical CDG structure (same topo_hash)
2. Assert primitives from the second source are skipped
3. Assert primitives with novel type_signatures are still added

---

## Task 6: De-duplication report

### File: `ageom/architect/catalog.py`

Add a dataclass:

```python
@dataclass
class CatalogReport:
    """Summary of catalog population with de-duplication metrics."""
    total_candidates: int = 0
    added: int = 0
    merged: int = 0
    structural_skips: int = 0
    merge_details: list[tuple[str, str, float]] = field(default_factory=list)
    # (candidate_name, incumbent_name, similarity)
```

Modify `add_with_dedup` to accept an optional `CatalogReport` and populate it.

### File: `ageom/architect/source_catalog.py`

Modify `seed_catalog_from_sources` to return `tuple[int, CatalogReport]`
instead of `int`.  The `CatalogReport` accumulates all dedup decisions.

### File: `ageom/cli.py`

In the `decompose` command (and wherever `seed_catalog_from_sources` is
called), log the report summary:
```
Catalog: 247 primitives (12 merged, 3 structural skips from 262 candidates)
```

### Tests: `tests/test_catalog.py`

- `test_catalog_report_counts` — verify counts after a sequence of
  add_with_dedup calls

---

## Task 7: Gap detection command

### File: `ageom/architect/catalog.py`

Add a method:

```python
def find_gaps(
    self,
    fallback_nodes: list[AlgorithmicNode],
    skill_index: "SkillIndex | None" = None,
    similarity_ceiling: float = 0.6,
) -> list[list[AlgorithmicNode]]:
    """Cluster fallback nodes that don't match any primitive well.

    Returns groups of similar unmatched nodes (each group is a gap).
    Uses SkillIndex to check that no existing primitive is close
    (top-1 similarity < similarity_ceiling).
    Groups remaining nodes by pairwise embedding similarity > 0.75.
    """
```

This is a diagnostic method, not on the hot path.  Simple greedy clustering
(not worth pulling in sklearn).

### File: `ageom/cli.py`

Add a `catalog-gaps` subcommand:

```
ageom catalog-gaps --cdg cdg.json [--threshold 0.6]
```

Loads the CDG, identifies ATOMIC nodes that hit the generic fallback
(no `matched_primitive` and description matches one of the
`_GENERIC_FALLBACK` templates in `deterministic_decompose.py`), runs
`find_gaps`, and prints clusters.

### Tests: `tests/test_catalog.py`

- `test_find_gaps_returns_clusters` — 4 nodes, 2 are similar to each other
  but not to any primitive -> one cluster of 2

---

## Execution order and dependencies

```
Task 1  (DedupResult, _structural_match)
  |
  v
Task 2  (check_duplicate)
  |
  v
Task 3  (add_with_dedup, _richer)
  |
  +------+
  v      v
Task 4  Task 5   (can be done in parallel)
  |      |
  v      v
Task 6  (report — depends on 4)
  |
  v
Task 7  (gap detection — independent, can be done last)
```

Tasks 1-3 are pure additions to `catalog.py` with no external dependencies.
Task 4 modifies `source_catalog.py` and `embedder.py`.  Task 5 modifies
`source_catalog.py` and uses `graph_store._topo_hash`.  Task 6 threads a
report through the existing call chain.  Task 7 adds a CLI command.

## Testing strategy

- Tasks 1-3, 6: Unit tests in `tests/test_catalog.py`.  No torch/FAISS
  required — mock the SkillIndex.
- Task 4: Integration test in `tests/test_source_catalog.py` with mock
  SkillIndex.
- Task 5: Integration test in `tests/test_source_catalog.py` with real
  `_topo_hash` (pure Python, no external deps).
- Task 7: Unit test for `find_gaps` + a CLI smoke test (mock CDG file).
- Optional `@pytest.mark.slow` tests that exercise real UniXcoder embeddings
  to validate threshold tuning.

## Files modified

| File | Tasks | Nature of change |
|------|-------|-----------------|
| `ageom/architect/catalog.py` | 1, 2, 3, 6, 7 | New methods, new dataclasses |
| `ageom/architect/source_catalog.py` | 4, 5 | Wire dedup into seeding loop |
| `ageom/architect/embedder.py` | 4 | `add_primitive` method |
| `ageom/cli.py` | 6, 7 | Log report, new subcommand |
| `tests/test_catalog.py` | 1, 2, 3, 6, 7 | New test classes |
| `tests/test_source_catalog.py` | 4, 5 | New integration tests |

## Not in scope

- Adding new source ingesters (scipy, networkx, etc.) — that's a follow-up
  once the de-duplication gate is proven.
- LLM-assisted gap filling — Task 7 surfaces gaps; filling them is separate.
- Changing `PrimitiveCatalog.add()` behavior — it remains unchanged for
  built-in primitives.  `add_with_dedup` is the new entry point for
  external sources.
