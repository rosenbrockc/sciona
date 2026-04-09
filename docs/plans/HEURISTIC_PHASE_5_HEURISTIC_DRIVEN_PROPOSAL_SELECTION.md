# Phase 5: Heuristic-Driven Proposal Selection

## Purpose

Move proposal generation, admissibility, and ranking onto the new heuristic
layer so that structural refinement decisions are based on explicit evidence
rather than thin rule-local diagnostics.

## Problem Statement

The framework already has expansion assets and proposal-selection machinery, but
the decision path is still too close to:

- raw summaries
- family-local diagnostics
- topology matching

That makes the system brittle and limits reuse across families.

## Scope

This phase covers:

- heuristic-to-action-class mapping
- admissibility integration
- proposal generation triggers
- deterministic ranking policy
- explanation and reporting for selected and rejected proposals

## Non-Goals

This phase does not:

- define the canonical heuristic schema
- define atom metadata
- define long-term outcome memory storage

## Deliverables

1. A shared interface from heuristic outputs into proposal generation.
2. Policy for mapping heuristic signatures to candidate action classes.
3. Deterministic ranking logic that can cite heuristic evidence explicitly.
4. Proposal reports that explain why a candidate was generated, selected, or
   rejected.

## Required Design Decisions

### Trigger model

The phase must specify:

- when heuristics trigger proposal generation
- when they only affect ranking or admissibility
- how multiple heuristics combine
- how conflicts are handled

### Action mapping

The policy should support:

- insertion
- replacement
- split-stage refinement
- validation sidecars
- smoothing or aggregation responses
- branch-and-compare responses

### Deterministic ranking

Ranking should consider:

- heuristic support strength
- family registry guidance
- constraint preservation
- admissibility
- structural cost
- expected benefit from prior outcome evidence when available

## Implementation Considerations

- Proposal selection must remain cross-family and not collapse into one
  family’s jargon or heuristics.
- Reporting should explain heuristic evidence in de-jargonized terms.
- The policy should support multiple candidate proposals without requiring the
  family-specific logic to reimplement selection itself.

## Testing Strategy

- proposal-generation tests from synthetic heuristic signatures
- ranking tests with competing action classes
- conflict-resolution tests
- explanation/reporting tests

## Risks

- Too many heuristics may trigger too many candidates.
- The ranking policy may silently encode family bias.
- Weak heuristics may dominate stronger structural evidence.

## Exit Criteria

Phase 5 is complete when:

- proposal generation can cite heuristics directly
- ranking behaves deterministically under competing heuristic signatures
- cross-family tests show the policy remains generic
