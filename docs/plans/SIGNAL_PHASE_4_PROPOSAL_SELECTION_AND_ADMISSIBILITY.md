# Signal Phase 4: Proposal Selection And Admissibility

## Status

Drafted on April 7, 2026 as Phase 4 of the signal-processing expansion
implementation plan in
[SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md).

## Purpose

This phase turns signal-family expansion into deterministic proposal-based CDG
enrichment rather than first-match rewriting.

## Problem

Even with semantic contracts and real expansion assets, the framework still
needs a disciplined way to decide:

- which enrichments are actually applicable
- which enriched CDGs are admissible
- which enriched CDG should be preferred

Without this phase, expansion remains a mutation engine rather than a selection
engine.

## Goals

1. Generate enriched CDG proposals from the signal-family asset inventory.
2. Apply deterministic admissibility checks to those proposals.
3. Score or rank proposals using planning constraints and telemetry evidence.
4. Record why a proposal was selected, rejected, or deferred.

## Proposal Types

This phase should support signal-family proposal types such as:

- insert a signal conditioning stage
- insert a correction stage
- insert a quality gate
- replace a detector or estimator stage
- add a branch-and-compare structure
- insert a smoothing stage after support checks

## Admissibility Model

Signal-family proposals should be checked for:

- compatibility with planning constraints
- compatibility with semantic stage roles
- preservation of required provenance and timing context
- acceptable information-loss profile
- evidence that the enrichment addresses a real observed weakness
- no introduction of unjustified complexity or unsupported branches

## Deliverables

1. Proposal materialization logic for signal-family assets.
2. Deterministic admissibility rules for signal-family enriched CDGs.
3. Proposal ranking and selection logic.
4. Selection telemetry and explanation records.

## Implementation Work

### Workstream A: Proposal generation

- materialize enriched candidate CDGs from applicable assets
- define proposal identity and deduplication rules

### Workstream B: Deterministic admissibility

- implement signal-family admissibility checks
- reject proposals that violate the family contract or lack evidence

### Workstream C: Proposal ranking

- define deterministic proposal scoring
- combine planning constraints, telemetry evidence, and structural cost

### Workstream D: Decision traceability

- persist why a proposal was selected, rejected, or deferred
- expose selection rationale in trial history and diagnostics

## Testing Strategy

- proposal materialization tests
- admissibility tests for valid and invalid enrichments
- ranking tests where multiple proposals compete
- integration tests showing better enriched CDGs are selected over weaker ones

## Risks

### Risk: proposal ranking becomes opaque

Mitigation:

- require explanation payloads for selection outcomes
- keep a deterministic baseline score before any heuristic tie-breakers

### Risk: proposal space becomes too large

Mitigation:

- constrain proposals to asset-backed enrichments
- deduplicate aggressively
- rank only a small proposal set per refinement step

## Exit Criteria

- the framework can generate multiple signal-family enriched CDG proposals
- deterministic admissibility narrows that set
- the selected proposal is inspectable and justified
- expansion no longer behaves like a blind first-match rewrite
