# SQL Manifest Phase C: Manifest Semantic Index

## Status

Drafted on April 14, 2026 as Phase C of
[SQL Manifest Implementation Plan](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_IMPLEMENTATION_PLAN.md).

## Goal

Let the hunter retrieve manifest-backed declarations through the semantic index
path, not just through the primitive catalog.

## Purpose

Phase B broadens the architect catalog. Phase C broadens retrieval.

Without this phase, the hunter still searches only the persisted local index and
misses atoms that exist only in the downloaded manifest.

## Current Code Reality

Today:

- [sciona/indexer/builder.py](/Users/conrad/personal/sciona-matcher/sciona/indexer/builder.py)
  builds a `FAISSStore` from declarations.
- [sciona/indexer/unified.py](/Users/conrad/personal/sciona-matcher/sciona/indexer/unified.py)
  models one store and one embedder.
- [sciona/indexer/fallback_index.py](/Users/conrad/personal/sciona-matcher/sciona/indexer/fallback_index.py)
  provides lexical fallback when FAISS is unavailable.
- [sciona/commands/runtime_helpers.py](/Users/conrad/personal/sciona-matcher/sciona/commands/runtime_helpers.py)
  loads exactly one local index from disk.

Two practical constraints matter here:

1. A manifest-built FAISS store must use the same embedding backend and model as
   the local store if their scores are to be merged.
2. Lexical fallback is a different retrieval mode and should not pretend to
   share vector-space semantics with FAISS.

## Scope

Phase C should do all of the following:

1. Build declaration objects from `manifest.sqlite`.
2. Construct a manifest-backed semantic index.
3. Provide a runtime wrapper that can search local and manifest sources
   together.
4. Integrate the wrapper into `_load_semantic_index()`.
5. Preserve lexical fallback behavior.

## Non-Goals

Phase C should not:

- rebuild the persisted local index on disk
- change the manifest catalog seeding path
- introduce remote network retrieval at runtime
- normalize every possible scoring backend into one abstraction

## Files In Scope

Primary files:

- [sciona/indexer/builder.py](/Users/conrad/personal/sciona-matcher/sciona/indexer/builder.py)
- [sciona/indexer/unified.py](/Users/conrad/personal/sciona-matcher/sciona/indexer/unified.py)
- [sciona/commands/runtime_helpers.py](/Users/conrad/personal/sciona-matcher/sciona/commands/runtime_helpers.py)

Primary tests:

- [tests/test_indexer.py](/Users/conrad/personal/sciona-matcher/tests/test_indexer.py)
- [tests/test_fallback_index.py](/Users/conrad/personal/sciona-matcher/tests/test_fallback_index.py)

## Implementation Steps

### Step 1: Build declarations from the manifest

Add `build_index_from_manifest_sqlite()` in `builder.py`.

The function should:

- open the manifest read-only
- query approved atoms and their descriptions
- query `io_specs`
- construct a `type_signature` deterministically from input and output ports
- create `Declaration` instances with `source_lib` set to a manifest-prefixed
  origin

The resulting function should either return a `FAISSStore` or return the
declaration list plus a built store if that helps later fallback logic.

### Step 2: Introduce a composite index wrapper

`UnifiedIndex` today models one store only. That is not enough.

Preferred direction:

- add a thin composite `SemanticIndex` implementation that wraps multiple
  backends and merges results
- keep `SemanticIndexImpl` and `LexicalSemanticIndex` untouched where possible

The wrapper should:

- query each child index independently
- merge by declaration name
- keep the best score per declaration
- return globally ranked results

### Step 3: Respect embedding-space compatibility

This is a hard design rule:

- if a manifest FAISS store is built at runtime, it must use the same embedder
  family as the loaded local FAISS store
- if that cannot be guaranteed, do not merge FAISS scores

The simplest safe path is:

- load the local FAISS store first
- infer embedder backend/model from its metadata
- pass that embedder into manifest index construction

### Step 4: Define lexical fallback behavior explicitly

If `_load_semantic_index()` falls back to lexical mode, Phase C should not try
to bolt vector semantics onto it.

Safe options:

- either leave lexical fallback local-only for the first version
- or add a lexical manifest declaration index and compose lexical indexes only

Do not mix lexical scores and FAISS scores in one ranking function without a
separate normalization design.

### Step 5: Wire into runtime loading

Update `_load_semantic_index()` so it:

1. loads the local index as it does today
2. checks for `~/.sciona/manifest.sqlite`
3. if present and FAISS mode is active, builds a compatible manifest index
4. wraps local and manifest indexes in a composite index
5. returns a mode string that still makes debugging clear

## Testing Plan

Add or extend tests for:

- manifest declaration construction from SQLite fixtures
- composite index merge behavior and duplicate-name resolution
- `_load_semantic_index()` in FAISS mode with a manifest present
- lexical fallback path remaining intact when FAISS is unavailable
- no crash when the manifest is missing, partial, or lacks `io_specs`

## Worker Breakdown

Recommended split:

- Worker C1: `indexer/builder.py`, `indexer/unified.py`, and
  `tests/test_indexer.py`
- Worker C2: `runtime_helpers.py` and `tests/test_fallback_index.py`

Constraint:

- C2 must coordinate with Phase B because both eventually touch
  `runtime_helpers.py`.

Practical recommendation:

- let C1 land the indexer primitives first
- do one combined B/C runtime integration pass afterward

## Risks And Decisions

### Score semantics

The biggest risk in this phase is invalid ranking caused by mixed embedding
spaces. This needs to be treated as a correctness issue, not as a tuning issue.

### Type-signature quality

Manifest `io_specs` may be less expressive than native source declarations. The
constructed signature only needs to be stable and useful, not perfect.

### Duplicate declarations

The same atom may exist in the local index and the manifest. The composite
index should deduplicate by declaration name and prefer the stronger score or
the local declaration if tie-breaking is needed.

## Exit Criteria

Phase C is complete when:

- runtime semantic search can include manifest-backed declarations
- FAISS score merging is safe and deterministic
- lexical fallback still works
- runtime integration is covered by indexer and fallback tests
