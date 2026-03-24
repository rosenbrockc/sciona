# Skeleton Proposal Phase 3

## Goal

Rank `primitive_proposal`, `template_proposal`, and `skeleton_proposal`
together under a single policy with an explicit complexity penalty.

This is the phase where skeleton proposals become comparable to simpler
alternatives, but they must still lose by default unless they are materially
better.

## Scope

In scope:

- unified proposal ranking
- complexity-penalty implementation
- risk penalty and prior bonus
- direct unit tests for ranking behavior

Out of scope:

- rollback behavior
- final safety margins under noisy execution
- dashboard/telemetry work

## Files To Update

- [sciona/architect/proposal_models.py](/Users/conrad/personal/ageo-matcher/sciona/architect/proposal_models.py)
- [sciona/architect/nodes.py](/Users/conrad/personal/ageo-matcher/sciona/architect/nodes.py)
- preferably a new ranking helper such as [sciona/architect/proposal_ranking.py](/Users/conrad/personal/ageo-matcher/sciona/architect/proposal_ranking.py)
- tests such as [tests/test_proposal_ranking.py](/Users/conrad/personal/ageo-matcher/tests/test_proposal_ranking.py)

## Implementation Tasks

### 1. Add a unified ranking function

Define a ranking helper that accepts a list of proposals and returns them
sorted by score.

Each proposal score should follow:

`proposal_score = objective_gain - complexity_penalty - risk_penalty + prior_bonus`

### 2. Implement the complexity penalty

This is the hard requirement for this phase.

Complexity penalty must increase with:

- `delta_nodes`
- `delta_edges`
- `delta_family_count`
- `delta_concept_type_count`

And `skeleton_proposal` must carry the highest base penalty.

Recommended starting heuristic:

`score = gain`
`- 0.20 * delta_nodes`
`- 0.10 * delta_edges`
`- 0.35 * delta_family_count`
`- 0.25 * delta_concept_type_count`
`- 0.50 * skeleton_base_penalty`

### 3. Implement risk penalty

Add a penalty term for:

- weak compatibility
- low template confidence
- uncertain boundary fit

It can be conservative and simple in this phase.

### 4. Implement prior bonus

Add a small positive prior for:

- same-family proposals
- historically stable proposal classes if such metadata already exists

Constraint:

- the prior must not outweigh the complexity term

### 5. Integrate ranking into node enrichment candidate handling

If the codebase already has a passive proposal list from earlier phases:

- rank primitive/template/skeleton candidates together
- keep the output of the ranking explicit and testable

Avoid hiding the ranking inside one large node function.

## Required Behavioral Tests

Add tests proving:

1. A primitive proposal beats a skeleton when objective gains are similar.
2. A template proposal beats a skeleton when the template is materially simpler
   and gains are comparable.
3. A skeleton can win only when its gain is materially better.
4. Same-family prior does not allow a more complex skeleton to beat a clearly
   better simpler proposal.

## Guardrails

1. Do not make family a hard gate.
2. Do not let prior bonus dominate complexity.
3. Do not hard-code a single family’s skeletons into the ranking logic.
4. Keep the ranking helper reusable and testable in isolation.

## Acceptance Criteria

Phase 3 is complete when:

- all three proposal classes are ranked together
- skeletons carry an explicit higher complexity penalty
- tests show simpler candidates win when gains are comparable
- tests show skeletons can still win when gains are materially stronger

## Recommended Test Command

```bash
pytest -q tests/test_proposal_models.py tests/test_proposal_ranking.py tests/test_decomposition.py
```

## Notes For Planner Agent

- This is the phase where local-minimum protection starts.
- Optimize for conservative ranking, not aggressive cross-family insertion.
