# Catalog Completeness Phase 3: Concrete Macro Artifacts

## Goal

Publish concrete exemplar CDGs that are bindable, benchmarkable, and useful at
macro retrieval time, rather than relying on abstract family skeletons alone.

## Why This Phase Matters

The current generic family skeletons are too abstract to bind automatically.
That means the macro catalog is structurally present, but not yet complete
enough to improve solution quality in the way we want.

The right fix is not to force abstract families to masquerade as concrete
algorithms. The right fix is to publish concrete exemplars beneath those
families.

## Scope

This phase covers:

- concrete family exemplar CDG assets
- artifact publication/sync for those exemplars
- matcher macro retrieval ranking for those concrete exemplars

It does not cover artifact evidence parity. That belongs to Phase 4.

## Primary Repos

- `sciona-matcher`
- provider repos that own the underlying primitive atoms

## Worker Ownership

Recommended family split:

- Worker 1: signal/event-rate exemplars
- Worker 2: state-estimation exemplars
- Worker 3: inference exemplars
- Worker 4: matcher-side artifact sync/retrieval integration

## Tasks

1. Identify family-level gaps.
   - For each currently abstract macro family, determine the smallest concrete
     exemplar that should be catalog-published.

2. Author concrete skeleton/CDG assets.
   - Add or refine assets under
     [sciona/architect/assets/skeletons](</Users/conrad/personal/sciona-matcher/sciona/architect/assets/skeletons>).
   - Ensure stages carry concrete primitive hints where appropriate.

3. Sync exemplars into Supabase and Memgraph.
   - Use the existing artifact sync path.
   - Confirm artifact IDs, versions, CDG nodes, edges, and bindings are all
     coherent.

4. Update macro retrieval ranking.
   - Ensure concrete exemplars outrank abstract family scaffolds when the query
     matches the exemplar more specifically.

5. Add family-level retrieval tests.
   - The tests should prove that concrete exemplars, not only local skeleton
     fallbacks, are retrievable from the catalog.

## Validation

- focused matcher tests around:
  - artifact retrieval
  - skeleton artifact loading
  - skeleton catalog sync
  - single-agent planner behavior
- live sync against local Supabase + Memgraph
- query-level checks that concrete exemplars win for family-specific prompts

## Exit Criteria

- at least the key signal and state-estimation families have concrete published
  exemplars
- those exemplars can be retrieved from the catalog as macro candidates
- they carry real bindings to registered leaf atoms
