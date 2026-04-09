# AGEO Atoms Heuristic Phase 8: Cross-Family Rollout

## Purpose

Extend the externally owned heuristic asset model beyond signal processing so
the system proves it remains cross-family rather than becoming a signal-only
subsystem.

## Problem

A heuristic asset model that only works cleanly for one family has not yet
proven the repository-wide design. The rollout phase is where we validate that:

- the canonical vocabulary stays portable
- the asset contracts work for very different families
- family interpretation remains local

## Scope

- select additional families for adoption
- author family registries and atom metadata for those families
- validate that no schema changes are needed to support them
- update audit and migration tooling as needed for genuine cross-family gaps

## Non-Goals

- migrating every family in one pass
- reopening canonical heuristic meaning casually for one family’s needs

## Deliverables

- at least one non-signal family migrated onto the external heuristic asset
  model
- family-specific registries and selected atom metadata in `ageo-atoms`
- lessons-learned notes on any true cross-family schema pressure

## Rollout Guidance

- prefer families with clearly different dataflow character from signal
  processing
- validate that shared heuristic IDs still make sense without domain-specific
  reinterpretation
- only extend canonical vocabulary when there is a real cross-family need

## Testing And Validation

- family-specific loader tests
- audit tests for new family registries
- cross-family compatibility tests ensuring canonical heuristics are reused
  rather than cloned under new jargon

## Risks

- family-local pressure causing canonical heuristics to drift into jargon
- introducing new shared IDs prematurely instead of reusing generic ones

## Exit Criteria

- at least one additional family uses `ageo-atoms` heuristic assets cleanly
- the schema remains stable across materially different families
- the audit system continues to enforce cross-family discipline successfully

## Dependencies

- depends on the earlier phases being stable enough to support wider adoption

