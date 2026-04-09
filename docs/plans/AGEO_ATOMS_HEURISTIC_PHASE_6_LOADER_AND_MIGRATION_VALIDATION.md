# AGEO Atoms Heuristic Phase 6: Loader And Migration Validation

## Purpose

Add explicit validation and observability around the mixed-source migration
window so `ageo-matcher` can prove which heuristic assets were loaded and warn
on ambiguous ownership or version drift.

## Problem

During migration, both repositories may temporarily contain overlapping
heuristic assets. Without dedicated validation, the system could silently load:

- stale matcher-local assets
- mismatched schema versions
- partial external migrations
- conflicting family registries

## Scope

- add matcher-side validation for heuristic asset provenance
- add reporting for external vs local asset selection
- add checks for version mismatches and ambiguous dual-source ownership
- surface migration state in summaries and benchmark/reporting hooks

## Non-Goals

- removing fallback behavior yet
- introducing new heuristic semantics

## Deliverables

- provenance summaries for loaded heuristic assets
- migration diagnostics and warnings
- validation tests for ambiguous or conflicting asset states
- benchmark/reporting fields that expose heuristic asset source state

## Key Design Constraints

- observability should not depend on manual inspection of local files
- warnings should be deterministic and actionable
- validation must remain generic across asset classes and families

## Testing And Validation

- tests for provenance summaries
- conflicting-source tests
- schema-version mismatch tests
- mixed migration state reporting tests

## Risks

- letting migration ambiguity persist too long without hardening
- overcomplicating the runtime with migration-only logic that is hard to remove

## Exit Criteria

- the matcher can prove which heuristic assets it loaded
- ambiguous ownership is surfaced automatically
- migration issues are visible in tests and reporting surfaces

## Dependencies

- depends on Phase 2
- can run in parallel with Phases 3, 4, and 5

