# Heuristic Usability Phase 4: Cohort And Benchmark Integration

## Purpose

Replace ad hoc usable-member checks with explicit usability-scope handling in
cohort selection, proposal guidance, scoring, and benchmark policy.

This phase is where the new usability model starts changing system behavior.

## Problem

The framework currently uses procedural proxies such as:

- run completed
- loss emitted
- heuristics present

Those checks are too weak. They cannot distinguish:

- guidance-useful but score-invalid members
- score-valid but benchmark-invalid members
- members that should be excluded entirely

## Scope

This phase covers:

- cohort member eligibility
- proposal-guidance aggregation
- benchmark admissibility policy
- reporting and explanation updates

## Non-Goals

This phase does not:

- design the canonical schema
- define family registries from scratch
- implement long-term storage
- enforce final audit policy

## Deliverables

1. Cohort selection driven by `usable_for_guidance`.
2. Scoring paths driven by `usable_for_scoring`.
3. Benchmark policy driven by `usable_for_final_benchmark`.
4. Reporting surfaces that explain exclusions and partial admissions.

## Required Behavioral Changes

The phase must ensure:

- cohort guidance can include informative but non-final members
- final benchmark conclusions exclude members lacking required evidence
- skipped members carry explicit usability reasons
- proposal weighting uses recurring issues only across guidance-usable members

## Implementation Work

### Workstream A: Cohort integration

- replace current usable-member checks
- update multi-night aggregation to use guidance scope explicitly

### Workstream B: Scoring integration

- prevent score comparison on members not usable for scoring
- distinguish hard exclusion from warning-only scoring

### Workstream C: Benchmark policy and reporting

- update policy summaries and tables
- expose usability reasons in benchmark outputs

## Testing Strategy

- cohort tests with mixed guidance/scoring usability cases
- benchmark-policy tests for partial and full exclusion scenarios
- proposal-guidance tests proving common issues are counted only across
  guidance-usable members
- reporting tests for explicit rationale visibility

## Risks

- policy changes may accidentally reduce benchmark coverage too aggressively
- guidance and scoring scopes may be confused in aggregation logic
- reporting may become opaque if reason handling is not standardized

## Exit Criteria

- cohort selection is driven by `usable_for_guidance`
- scoring and benchmark paths use the appropriate scopes
- outputs explain why members were included, excluded, or only partially
  admitted
- procedural usability checks are no longer the primary selection mechanism

## Dependencies

- depends on meaningful progress in Phases 2 and 3

## Parallelization Notes

- sits on the critical path after Phases 2 and 3
- should precede most Phase 5 work so memory reflects operational behavior

