# Skeleton Proposal Phase 2

## Goal

Generate bounded `skeleton_proposal` candidates for eligible nodes.

This phase should make skeleton proposals *possible*, but still tightly gated.
It should not yet introduce broad acceptance behavior or full ranking-policy
changes beyond producing candidates with raw metadata.

## Scope

In scope:

- skeleton proposal generator
- family allowlist
- boundary compatibility checks
- hard size caps
- unit tests for candidate generation and rejection

Out of scope:

- unified ranking across primitive/template/skeleton
- acceptance-margin policy
- rollback behavior
- telemetry/dashboard changes

## Files To Update

- [sciona/architect/proposal_models.py](/Users/conrad/personal/ageo-matcher/sciona/architect/proposal_models.py)
- [sciona/architect/nodes.py](/Users/conrad/personal/ageo-matcher/sciona/architect/nodes.py)
- [sciona/architect/skeletons.py](/Users/conrad/personal/ageo-matcher/sciona/architect/skeletons.py)
- optionally a new helper like [sciona/architect/skeleton_proposals.py](/Users/conrad/personal/ageo-matcher/sciona/architect/skeleton_proposals.py)
- [tests/test_proposal_models.py](/Users/conrad/personal/ageo-matcher/tests/test_proposal_models.py)
- new tests such as [tests/test_skeleton_proposals.py](/Users/conrad/personal/ageo-matcher/tests/test_skeleton_proposals.py)

## Implementation Tasks

### 1. Add a skeleton proposal generator

Create a helper that takes:

- target node
- current graph context if needed
- candidate skeleton allowlist

Returns:

- zero or more `skeleton_proposal` candidates

### 2. Add the initial allowlist

Start with a very narrow allowlist of skeleton families or named skeletons.

Requirements:

- do not enumerate the full skeleton registry by default
- keep the initial list small and explicit
- document the allowlist in code comments or constants

### 3. Add boundary compatibility checks

For a skeleton to become a proposal:

- input arity must be compatible with the target node
- output arity must be compatible with the target node
- type-class compatibility must not obviously conflict

This phase can use conservative compatibility checks. If compatibility is
unclear, reject the skeleton candidate.

### 4. Add size caps

Every skeleton proposal must be filtered by:

- maximum inserted node count
- maximum inserted edge count

If a skeleton exceeds either cap, do not emit a proposal.

### 5. Add raw metadata for ranking later

Populate these fields on generated `skeleton_proposal` candidates:

- `source_family`
- `skeleton_name`
- `delta_nodes`
- `delta_edges`
- `delta_family_count`
- `delta_concept_type_count`
- `compatibility_score`

These are raw fields only. They are not yet used for final selection policy.

### 6. Thread generation into node enrichment without changing selection

In [sciona/architect/nodes.py](/Users/conrad/personal/ageo-matcher/sciona/architect/nodes.py):

- build `skeleton_proposal` candidates alongside the passive primitive/template
  proposal list
- do not let them override current behavior yet unless the existing code is
  already using proposals as a candidate surface

If proposal selection has not yet been introduced:

- keep the proposals passive and testable
- do not silently switch decompose behavior

## Guardrails

1. Do not broaden candidate generation to all skeletons.
2. Reject any skeleton with ambiguous IO compatibility.
3. Keep the allowlist small.
4. Do not add ranking heuristics yet beyond metadata collection.

## Acceptance Criteria

Phase 2 is complete when:

- eligible nodes can produce bounded `skeleton_proposal` candidates
- ineligible nodes produce none
- allowlist enforcement is covered by tests
- size caps are covered by tests
- boundary compatibility failures are covered by tests

## Recommended Test Command

```bash
pytest -q tests/test_proposal_models.py tests/test_skeleton_proposals.py tests/test_decomposition.py
```

## Notes For Planner Agent

- This is a generation phase, not a policy phase.
- Be conservative: false negatives are acceptable here, false positives are not.
