# Phase 6: Heuristic Outcome Memory And Benchmark Integration

## Purpose

Close the loop by recording which heuristic signatures led to which actions and
what happened afterward, so the framework can build deterministic experience
without hard-coding narrow expert rules.

## Problem Statement

Without outcome memory, heuristics can guide one run but cannot accumulate
evidence across runs. The framework will continue to rediscover the same
patterns instead of learning a stable prior about which actions tend to help
under which heuristic conditions.

## Scope

This phase covers:

- outcome-memory schema
- integration with benchmark artifacts
- heuristic signature recording
- action and improvement tracking
- replay and reporting support

## Non-Goals

This phase does not:

- redesign the whole benchmark harness
- replace constraint-first planning
- create a fully autonomous learning system

## Deliverables

1. A storage schema for heuristic signatures, actions, and observed outcomes.
2. Integration with benchmark and optimize artifacts.
3. Reporting that summarizes heuristic signatures and resulting actions.
4. A policy for how outcome memory influences ranking without overfitting.

## Required Design Decisions

### Memory schema

At minimum, persist:

- family
- skeleton
- benchmark context
- heuristic signature
- producing atoms or transforms
- selected action and action class
- outcome deltas
- uncertainty notes
- sample-count or evidence-strength indicators

### Influence policy

The phase must define:

- how outcome memory affects proposal ranking
- how to avoid overweighting a small number of runs
- how to treat conflicting historical evidence
- how to represent uncertainty explicitly

### Reporting policy

Benchmark artifacts should be able to show:

- which heuristics were present
- which actions were considered
- what the observed gains or regressions were
- whether the memory signal is mature or weak

## Implementation Considerations

- Outcome memory should support later migration or externalization.
- The design should be useful even before the memory corpus is large.
- The system should distinguish between anecdotal evidence and stable patterns.

## Testing Strategy

- storage round-trip tests
- ranking-integration tests with synthetic history
- benchmark-reporting tests
- safeguards against small-sample overconfidence

## Risks

- Early memory may overfit to one family or benchmark.
- Historical priors may become sticky and suppress exploration.
- Reporting may imply false confidence if uncertainty is not surfaced clearly.

## Exit Criteria

Phase 6 is complete when:

- benchmarks can report heuristic signatures and observed action outcomes
- proposal selection can consume outcome memory cautiously and deterministically
- the system distinguishes weak history from mature evidence
