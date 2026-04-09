# AGEO Atoms Heuristic Phase 7: Matcher Source Reduction

## Purpose

Reduce `ageo-matcher` from heuristic asset author to heuristic asset consumer by
removing or deprecating matcher-local source-of-truth definitions after enough
coverage exists in `../ageo-atoms`.

## Problem

The migration is not complete while matcher-local heuristic assets remain the
real authoring home in practice. The system needs a deliberate cleanup phase so
ownership becomes unambiguous.

## Scope

- deprecate matcher-local canonical heuristic definitions where possible
- deprecate matcher-local family registry assets where external ownership exists
- keep only compatibility adapters, runtime-derived heuristics, and policy
  consumption code
- document the supported transitional surfaces that remain

## Non-Goals

- removing runtime-derived heuristic logic
- removing all compatibility shims immediately if still needed for active
  migrations

## Deliverables

- deprecation/removal plan for matcher-local heuristic assets
- code cleanup reducing duplicate asset definitions
- tests proving external assets remain authoritative

## Key Design Constraints

- remove duplication without breaking active families still in migration
- keep compatibility code isolated and clearly marked transitional

## Testing And Validation

- tests proving matcher uses external assets as the authoritative source
- regression tests for remaining fallback paths
- negative tests ensuring removed local assets are not silently still used

## Risks

- removing local compatibility too early
- leaving dead matcher-local definitions that continue to confuse contributors

## Exit Criteria

- matcher-local heuristic assets are no longer the primary authoring home
- remaining local heuristic logic is clearly transitional or runtime-specific

## Dependencies

- should follow substantial completion of Phases 3 through 6

