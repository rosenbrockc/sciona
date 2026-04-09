# Phase 3: Family Heuristic Registries

## Purpose

Create the family-owned layer that interprets generic heuristics locally
without contaminating the shared heuristic interface with family jargon.

## Problem Statement

A shared heuristic schema alone is not enough. Different families need
different:

- heuristic producers
- action mappings
- admissibility implications
- confidence thresholds

Those specializations must be explicit and auditable, but they must stay local
to families or skeletons rather than leaking into the shared schema.

## Scope

This phase covers:

- family or skeleton heuristic registries
- sanctioned heuristic producers per family
- action-class mappings
- family-local interpretation notes
- escalation and admissibility hooks

## Non-Goals

This phase does not:

- implement runtime extraction
- implement proposal ranking
- decide the final memory schema

## Deliverables

1. A registry format for family-level heuristic configuration.
2. Registry examples for at least one reference family and one non-signal
   family candidate.
3. Validation rules connecting registries to canonical heuristic identifiers and
   action classes.
4. A compatibility story for skeleton-level overrides or narrowing.

## Required Design Decisions

### Registry contents

Each registry should be able to declare:

- relevant heuristic identifiers
- sanctioned producers
- expected evidence strength
- supported action classes
- optional action priorities
- admissibility notes
- escalation conditions
- family-local documentation

### Scope and ownership

The phase must define:

- whether registries attach at family level, skeleton level, or both
- how overrides work
- where long-term ownership should live

### Cross-family safeguards

The registry system should prevent:

- unknown heuristic identifiers
- family-specific jargon in shared fields
- direct bypass of the canonical action-class vocabulary

## Implementation Considerations

- Registries should be auditable assets, not hidden Python code.
- The design should support progressive refinement from broad family defaults to
  skeleton-specific specialization.
- It should remain possible to interpret one heuristic differently in different
  families without redefining the heuristic itself.

## Testing Strategy

- registry schema validation tests
- cross-reference tests against the canonical heuristic schema
- examples proving that two families can share a heuristic but map it to
  different action classes or priorities

## Risks

- Families may overfit their registries to a small benchmark set.
- Registry precedence rules may become hard to reason about.
- Family notes may quietly reintroduce jargon.

## Exit Criteria

Phase 3 is complete when:

- families can declare heuristic interpretation locally
- the shared heuristic model remains generic
- validation prevents incompatible or overly narrow registry definitions
