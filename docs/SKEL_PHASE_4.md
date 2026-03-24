# Skeleton Proposal Phase 4

## Goal

Make `skeleton_proposal` safe in the live enrichment path by adding explicit
acceptance rules, minimum-margin checks, and rollback-safe behavior.

This is the phase where skeleton proposals become a real live option, but only
under strict safety constraints.

## Scope

In scope:

- acceptance thresholds
- minimum-margin rule over lower-complexity alternatives
- live selection behavior
- rejection and rollback-safe semantics
- live acceptance tests

Out of scope:

- dashboard and telemetry polish
- global tuning loops for penalty weights

## Files To Update

- [sciona/architect/nodes.py](/Users/conrad/personal/ageo-matcher/sciona/architect/nodes.py)
- [sciona/principal/graph.py](/Users/conrad/personal/ageo-matcher/sciona/principal/graph.py) only if skeleton proposals participate in Principal-side proposal selection
- [tests/test_decomposition.py](/Users/conrad/personal/ageo-matcher/tests/test_decomposition.py)
- add focused live tests such as [tests/test_skeleton_proposal_acceptance.py](/Users/conrad/personal/ageo-matcher/tests/test_skeleton_proposal_acceptance.py)

## Implementation Tasks

### 1. Add explicit acceptance rule

A skeleton proposal should only be accepted if:

- it passes all gating checks
- it outranks primitive/template alternatives
- it clears the minimum-margin rule against the best lower-complexity candidate

### 2. Add the minimum-margin rule

Hard requirement:

- if a skeleton proposal is more complex than the best lower-complexity
  alternative, require a strictly positive margin above that alternative before
  acceptance

Keep this conservative.

### 3. Add live rejection behavior

If a skeleton proposal fails the acceptance rule:

- reject it cleanly
- fall back to the best lower-risk alternative or keep the current behavior

### 4. Keep rollback-safe semantics

If a chosen skeleton-based enrichment later harms the measured objective:

- the system must be able to reject or revert it
- do not allow unbounded structural drift from one bad skeleton decision

If rollback already exists elsewhere, integrate with it rather than duplicating it.

## Required Live Tests

Add live tests proving:

1. Skeleton proposal is considered but rejected when a simpler candidate is
   nearly as good.
2. Skeleton proposal is accepted only when it is materially better.
3. Harmful skeleton insertion is rejected or later reverted.
4. Cross-family skeleton proposal can win when compatibility is strong and the
   measured benefit clears the margin.

## Guardrails

1. Do not bypass the minimum-margin rule.
2. Do not let skeleton acceptance depend only on rank order; margin matters.
3. Do not mix broad telemetry work into this phase.

## Acceptance Criteria

Phase 4 is complete when:

- skeleton proposals can be accepted in live enrichment
- they are rejected when simpler alternatives are not meaningfully worse
- harmful skeleton insertions are rollback-safe
- live tests cover acceptance, rejection, and revert paths

## Recommended Test Command

```bash
pytest -q tests/test_decomposition.py tests/test_skeleton_proposal_acceptance.py tests/test_principal.py
```

## Notes For Planner Agent

- Treat this as the safety phase.
- Conservative rejection is better than aggressive acceptance.
