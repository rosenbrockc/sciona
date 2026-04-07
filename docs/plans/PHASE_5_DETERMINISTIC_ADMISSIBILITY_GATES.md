# Phase 5: Deterministic Admissibility Gates

## Status

Drafted on April 7, 2026 as the fifth implementation phase of
[Constraint-Driven Synthesis And Refinement Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md).

This phase assumes:

- planning constraints exist
- skeleton and expansion family assets exist or have defined contracts
- the runtime graph model is semantically rich enough to express the relevant
  boundaries and edge semantics

## Purpose

The framework currently allows too many semantically weak candidates to survive
into expensive stages such as optimization, export, profiling, and manual
inspection.

That is the opposite of what determinism should provide.

This phase introduces deterministic admissibility gates so the system can prune
obviously bad candidates early and reproducibly.

## Goal

Make candidate admissibility a first-class runtime decision instead of an
implicit side effect of later evaluation failures.

The phase should move the system from:

- "can this candidate execute at all"

to:

- "is this candidate semantically admissible enough to deserve further search"

## Non-Goals

Phase 5 should not:

- replace full benchmark evaluation
- encode every family-specific notion of quality
- become a generic ranking model
- hard-code special cases in ways that bypass the planning and asset systems

Admissibility is an early pruning layer, not the whole search strategy.

## Problem Statement

A candidate can be:

- type-correct
- executable
- exportable

and still be obviously poor.

Examples of the kinds of failure this phase is meant to catch:

- implausibly low event count for the observed duration
- failure to use required context such as sampling metadata
- catastrophic output sparsity or density
- outputs that violate declared stage preconditions
- candidates that destroy information earlier than allowed by the planning
  constraints

Today these failures are often discovered too late. That wastes time and makes
search quality look worse than it should.

## Admissibility Concept

Admissibility should be treated as a deterministic, inspectable decision layer
that operates on:

- planning constraints
- semantic graph structure
- compact telemetry summaries
- family asset expectations

The system should be able to answer:

- what rule rejected this candidate
- what evidence triggered that rejection
- whether the rejection is hard or soft
- whether a refinement could make the candidate admissible

## Hard Vs Soft Admissibility

This phase should support two broad classes of decisions.

### Hard Rejection

Candidate should not continue.

Examples:

- required input semantics missing
- output cardinality below a declared minimum
- downstream precondition violated
- catastrophic information loss before an irreversible boundary

### Soft Penalty Or Warning

Candidate may continue but should be deprioritized or routed to refinement.

Examples:

- suspicious but not catastrophic density
- moderate instability in event intervals
- missing recommended correction stage

The framework should not reduce everything to one binary outcome.

## Sources Of Admissibility Rules

Admissibility should not be defined in one place only.

It should draw from:

- planning constraints from Phase 1
- skeleton-family expectations from Phase 2
- refinement-family expectations from Phase 3
- semantic graph and edge meaning from Phase 4
- runtime telemetry summaries from Phase 6

This phase should define how those sources are consumed, not rebuild them.

## Runtime Placement

Admissibility gates should be placed where they save the most time.

Likely checkpoints include:

- after initial candidate assembly
- after cheap execution or ghost-style validation
- after a refinement step
- before expensive optimization trials

The exact placement may differ by path, but the principle is:

- reject or redirect before expensive work when the evidence is already strong

## Explainability Requirement

Every admissibility decision should emit structured justification.

Useful fields include:

- rule id
- severity
- evidence summary
- observed metric
- threshold or expected bound
- hard vs soft disposition
- suggested refinement route if known

This is critical for:

- debugging search failures
- evaluating search quality
- preventing hidden family heuristics

## Family Extension Model

The framework should provide a generic admissibility system with a family
extension surface.

That means:

- some rules are family-agnostic
- some rules are family-level extensions
- family rules should be declared through explicit assets or registries rather
  than hidden local conditionals

The exact enforcement may remain local, but the rule inventory should still be
auditable.

## Suggested Deliverables

Phase 5 should produce:

1. A canonical admissibility rule contract.
2. Runtime support for hard rejection, soft warning, and refinement routing.
3. Structured admissibility decision artifacts in trial state.
4. Family extension points for additional admissibility rules.
5. Integration of admissibility checks into one or more core search paths.
6. Maintainer documentation on authoring and reviewing admissibility rules.

## Testing Strategy

### Unit Tests

Verify:

- rule evaluation is deterministic
- hard and soft decisions are distinguished correctly
- structured justifications are emitted

### Integration Tests

Verify:

- inadmissible candidates are pruned before expensive downstream work
- soft failures can still route to refinement
- admissibility decisions are preserved in trial history or benchmark artifacts

### Regression Tests

Verify:

- common failure patterns no longer survive to late-stage search
- family-specific admissibility logic remains explicit and reviewable

## Risks

### Risk: Over-pruning

If admissibility is too aggressive, the framework may reject candidates that
would have been recoverable through refinement.

Mitigation:

- support soft dispositions
- prefer refinement routing when evidence suggests recoverability

### Risk: Family-specific heuristics become hidden policy

Mitigation:

- require rule identifiers and structured justifications
- keep extension points explicit

### Risk: Admissibility duplicates evaluation

Mitigation:

- keep admissibility cheap and early
- leave final quality ranking to later evaluation

### Risk: Rules depend on telemetry that does not yet exist consistently

Mitigation:

- phase ordering should place full telemetry work alongside or ahead of the
  most telemetry-dependent gates

## Exit Criteria

Phase 5 should be considered complete when:

- the framework has a first-class admissibility decision layer
- candidates can be hard-rejected or soft-routed with explicit reasons
- admissibility decisions appear in runtime artifacts or trial history
- at least one expensive failure mode is pruned significantly earlier than
  before
- family-specific admissibility rules are explicit and inspectable

## Deferred To Later Phases

Phase 5 does not finish:

- telemetry standardization
- benchmark reform
- all possible family extensions

Those later phases should consume the admissibility system rather than replace
it.

## Recommended Sequencing

1. define the admissibility rule contract
2. implement hard/soft dispositions and reporting
3. wire checks into one core search path
4. add family-extension support
5. expand to other search paths
6. document maintainer expectations

## Relationship To Other Phases

Phase 5 is where determinism starts paying off directly in search quality.

It depends on semantic planning, family assets, and semantic graph structure,
and it will become much more effective once Phase 6 provides richer telemetry.
