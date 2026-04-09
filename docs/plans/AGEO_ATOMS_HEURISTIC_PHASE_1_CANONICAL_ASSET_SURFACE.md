# AGEO Atoms Heuristic Phase 1: Canonical Asset Surface

## Purpose

Create the canonical declarative asset surface in `../ageo-atoms` for the
heuristic layer so that heuristic knowledge can be authored, reviewed, and
versioned there rather than remaining matcher-local.

## Problem

Today `ageo-matcher` owns the heuristic schema and local asset shapes. That was
useful for validating the abstraction, but it is the wrong long-term ownership
boundary. `ageo-atoms` needs a first-class place to store:

- canonical heuristic definitions
- per-atom heuristic metadata
- family heuristic registries

without depending on Principal internals.

## Scope

- define the canonical asset directories and file conventions in `ageo-atoms`
- define the high-level schemas for the three heuristic asset classes
- define repository ownership and versioning rules
- define transitional compatibility expectations for `ageo-matcher`

## Non-Goals

- migrating all existing heuristic assets immediately
- changing proposal policy
- changing runtime extraction logic

## Deliverables

- a canonical directory layout in `ageo-atoms` for heuristic assets
- schema definitions or equivalent typed asset contracts for:
  - canonical heuristics
  - atom heuristic metadata
  - family heuristic registries
- naming/versioning guidance for those assets
- migration-readiness notes for each asset class

## Key Design Constraints

- canonical heuristic IDs must remain cross-family and dejargonized
- asset contracts must be declarative and not encode matcher scheduling logic
- family registries may interpret shared heuristics locally but may not redefine
  them
- atom metadata must describe evidence, not global proposal policy

## Implementation Notes

- choose a stable on-disk layout that future family assets can share
- keep the asset contract narrow enough that new families can adopt it without
  signal-specific fields
- include explicit source-kind and review-status fields so migration state stays
  observable

## Testing And Validation

- schema validation tests for each asset class
- round-trip load tests from representative JSON assets
- negative tests for duplicate IDs, missing dejargonized summaries, and invalid
  action classes

## Risks

- recreating matcher-local assumptions in the `ageo-atoms` schema
- making the canonical schema too signal-oriented
- overfitting the directory layout to current heuristic families only

## Exit Criteria

- `ageo-atoms` can represent canonical heuristics, atom heuristic metadata, and
  family registries as auditable assets
- the asset contract is stable enough for loader work in `ageo-matcher`
- cross-family constraints are expressed in the contract itself, not only in
  prose

## Dependencies

- none; this is the foundation phase

