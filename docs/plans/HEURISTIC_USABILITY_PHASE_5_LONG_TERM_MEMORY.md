# Heuristic Usability Phase 5: Long-Term Memory

## Purpose

Persist usability assessments, heuristic signatures, actions, and outcomes so
Sciona can learn deterministically over time what "usable" means in practice.

This phase turns runtime usability from an ephemeral artifact into platform
memory.

## Problem

Without long-term persistence, the framework cannot accumulate evidence about:

- which heuristic signatures predict unusable data
- which members are guidance-useful but benchmark-invalid
- which actions help under degraded usability
- how stable those patterns are across build contexts

## Scope

This phase covers:

- memory-record schema extensions
- persistence of usability assessments and context
- query/reporting surfaces for usability-aware historical learning

## Non-Goals

This phase does not:

- define canonical usability schema
- emit runtime assessments
- redesign benchmark policy from scratch
- implement audit enforcement

## Deliverables

1. Memory schema support for usability assessments.
2. Persistence of heuristic signatures, usability decisions, actions, and
   outcomes.
3. Reporting/query helpers for usability-aware historical analysis.
4. Compatibility behavior for legacy runs lacking explicit usability artifacts.

## Required Stored Fields

At minimum, store:

- heuristic signature
- usability assessment
- family and task context
- selected atoms and expansions
- action classes attempted
- outcome deltas
- build/runtime context identifiers
- provenance for the usability decision

## Implementation Work

### Workstream A: Memory schema

- extend outcome-memory structures for usability
- define compatibility behavior for historical runs

### Workstream B: Persistence

- write usability records during optimize/profile/benchmark flows
- ensure record identity is deterministic

### Workstream C: Reporting

- expose summaries such as common blockers, guidance-only rates, and
  action-performance conditioned on usability signatures

## Testing Strategy

- schema compatibility tests
- persistence tests on representative runs
- reporting tests proving usability-aware aggregation
- fallback tests for older runs without the new artifact

## Risks

- memory records may become too large or too tied to one runtime format
- poor record identity may fragment the same usability signature across runs
- reporting may drift into family-specific language

## Exit Criteria

- usability assessments are persisted in long-term memory
- historical reporting can group by heuristic signature and usability scope
- compatibility behavior exists for legacy runs
- the stored record is explicit enough to support future deterministic priors

## Dependencies

- depends on Phase 3
- should follow or closely track Phase 4

## Parallelization Notes

- can run in parallel with Phase 6 once the Phase 4 artifact shape stabilizes

