# AGEO Atoms Heuristic Phase 3: Canonical Heuristic Migration

## Purpose

Move the canonical heuristic vocabulary out of `ageo-matcher` and into
`../ageo-atoms` so the shared heuristic language is no longer matcher-local.

## Problem

The canonical heuristic IDs and meanings are currently defined in matcher code.
That makes the execution engine the authoring home for shared vocabulary, which
is the wrong ownership model for audited ecosystem-wide assets.

## Scope

- migrate canonical heuristic definitions into `ageo-atoms`
- keep matcher-side runtime APIs stable during the transition
- preserve backward compatibility for existing heuristic IDs
- define deprecation behavior for matcher-local canonical definitions

## Non-Goals

- migrating family registries
- migrating atom metadata
- changing runtime heuristic derivation semantics

## Deliverables

- canonical heuristic asset files in `ageo-atoms`
- matcher adapters that load canonical heuristics from those assets
- compatibility mapping for any pre-existing legacy diagnostic names
- deprecation notes for matcher-local definitions

## Key Design Constraints

- heuristic IDs must remain dejargonized and portable
- canonical heuristics may not include family-local interpretation fields
- compatibility mappings may exist, but they must not become the new source of
  truth

## Implementation Notes

- preserve the existing shared action-class model unless there is a clear schema
  gap
- keep legacy compatibility hints narrow and explicitly transitional
- ensure canonical heuristics can be referenced equally by atom metadata and
  family registries

## Testing And Validation

- tests for loading canonical heuristics from `ageo-atoms`
- equality/compatibility tests for migrated matcher behavior
- negative tests for family-local jargon in canonical heuristic assets

## Risks

- keeping too much matcher-local heuristic definition logic alive
- confusing compatibility hints with canonical definitions

## Exit Criteria

- canonical heuristic definitions are authored in `ageo-atoms`
- matcher consumes them without changing downstream interfaces
- canonical heuristics remain family-neutral and dejargonized

## Dependencies

- depends on Phases 1 and 2

