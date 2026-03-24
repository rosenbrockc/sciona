# Skeleton Proposal Phase 5

## Goal

Make skeleton proposal behavior observable, measurable, and tuneable through
telemetry and dashboard summaries.

This phase should let the team answer:

- how often skeleton proposals are considered
- how often they are accepted
- whether they improve the objective
- whether they survive later optimization
- whether complexity penalties are strong enough

## Scope

In scope:

- telemetry fields for skeleton proposals
- optimize summary integration
- dashboard summary integration
- tests for API/dashboard payloads

Out of scope:

- introducing new proposal-selection behavior
- retuning core acceptance logic unless required by a bug

## Files To Update

- [sciona/commands/optimize_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/optimize_cmds.py)
- [sciona/visualizer_api.py](/Users/conrad/personal/ageo-matcher/sciona/visualizer_api.py)
- [sciona/static/dashboard.html](/Users/conrad/personal/ageo-matcher/sciona/static/dashboard.html)
- [tests/test_visualizer_api.py](/Users/conrad/personal/ageo-matcher/tests/test_visualizer_api.py)

## Required Telemetry

Per run:

- `skeleton_proposal_trials`
- `accepted_skeleton_proposals`
- `rejected_skeleton_proposals`
- `mean_skeleton_complexity_penalty`
- `mean_skeleton_objective_gain`
- `skeleton_retention_rate`

Per trial row:

- target node
- proposal type
- source skeleton family
- inserted node count
- inserted edge count
- complexity penalty
- objective estimate or measured gain
- accepted / rejected
- retained / reverted

## Implementation Tasks

### 1. Extend optimize history summarization

In [optimize_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/optimize_cmds.py):

- summarize skeleton proposal counts
- summarize acceptance/rejection counts
- compute mean complexity penalty and objective gain
- compute retention rate if later trial history supports it

### 2. Extend API summary serialization

In [visualizer_api.py](/Users/conrad/personal/ageo-matcher/sciona/visualizer_api.py):

- expose run-level skeleton proposal summary fields
- expose per-trial skeleton proposal details

### 3. Extend dashboard rendering

In [dashboard.html](/Users/conrad/personal/ageo-matcher/sciona/static/dashboard.html):

- show skeleton proposal counts
- show acceptance vs rejection
- show average complexity penalty
- show objective gain / retention

Keep the presentation concise and operationally useful.

### 4. Add tests

In [tests/test_visualizer_api.py](/Users/conrad/personal/ageo-matcher/tests/test_visualizer_api.py):

- add payload examples with skeleton proposal summaries
- verify the API returns the new fields

## Guardrails

1. Do not invent telemetry fields that are not grounded in real trial history.
2. Do not overload the dashboard with low-signal details.
3. Keep run-level summaries distinct from per-trial raw rows.

## Acceptance Criteria

Phase 5 is complete when:

- skeleton proposal behavior is visible in optimize telemetry
- dashboard summaries expose both volume and quality of skeleton use
- tests verify the new API surface

## Recommended Test Command

```bash
pytest -q tests/test_visualizer_api.py
```

## Notes For Planner Agent

- This is the observability phase.
- Prefer fields that help tune the complexity penalty and acceptance margin.
