# AGEO Atoms Heuristic Phase 2: Matcher Loader And Compatibility

## Purpose

Teach `ageo-matcher` to load heuristic assets from `../ageo-atoms` while
preserving a safe migration window for existing matcher-local assets.

## Problem

Even with a canonical asset surface in `ageo-atoms`, the matcher still needs a
clear policy for:

- locating external heuristic assets
- validating schema compatibility
- resolving precedence between local and external assets
- surfacing migration state in telemetry and tests

## Scope

- implement loaders for the three heuristic asset classes from `ageo-atoms`
- define precedence rules between external and local assets during migration
- add compatibility shims where current matcher code expects local assets
- surface loaded-source provenance in summaries and diagnostics

## Non-Goals

- deprecating local assets immediately
- migrating all family registries in this phase
- changing heuristic ranking logic

## Deliverables

- loader functions in `ageo-matcher` for external heuristic assets
- explicit precedence rules such as `ageo-atoms first, local fallback`
- compatibility summaries that show which source provided each asset
- regression tests for mixed-source loading

## Key Design Constraints

- the migration must be observable, not silent
- external asset loading must not require Principal-specific fields inside the
  asset files
- fallback behavior must be deterministic and easy to audit

## Implementation Notes

- prefer narrow loader adapters over invasive changes to downstream consumers
- keep transitional shims isolated so they can be removed later
- make missing or ambiguous ownership a surfaced state, not hidden behavior

## Testing And Validation

- loader tests for `ageo-atoms` assets
- precedence tests covering external-only, local-only, and dual-source cases
- version mismatch tests
- summary/provenance tests showing the selected source

## Risks

- silent drift between local and external assets during migration
- making external loading depend on local path assumptions that are hard to
  generalize later

## Exit Criteria

- matcher can consume heuristic assets from `ageo-atoms`
- dual-source behavior is deterministic and visible
- existing heuristic-driven flows still work with the compatibility layer

## Dependencies

- depends on Phase 1 canonical asset surface

