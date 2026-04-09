# AGEO Atoms Heuristic Phase 5: Audit System

## Purpose

Make heuristic assets subject to explicit audit enforcement in `../ageo-atoms`
so cross-family portability, dejargonization, and family-boundary discipline are
checked by tooling rather than left to reviewer memory.

## Problem

Without a dedicated audit phase, heuristic assets could drift toward:

- domain jargon in canonical IDs
- weak or missing dejargonized explanations
- family registries that redefine shared heuristic meaning
- unsupported producer or action declarations
- undocumented uncertainty and provenance assumptions

That would undermine the entire cross-family purpose of the heuristic layer.

## Scope

- extend `ageo-atoms` audit tooling to validate heuristic assets
- define heuristic-specific audit findings and severities
- add checks for cross-family naming discipline and family-boundary rules
- define audit coverage expectations for canonical heuristics, atom metadata,
  and family registries

## Non-Goals

- changing matcher proposal logic
- auditing runtime outcomes or benchmark quality in this phase

## Deliverables

- audit rules for heuristic definitions
- audit rules for atom heuristic metadata
- audit rules for family heuristic registries
- documented severity policy for heuristic findings
- regression tests covering representative good and bad assets

## Required Audit Checks

- schema validity
- duplicate IDs or conflicting declarations
- missing dejargonized summaries or weak placeholder descriptions
- banned or suspicious domain jargon in cross-family canonical fields
- family registry attempts to redefine shared heuristic meaning
- unsupported action classes or producer kinds
- missing references, uncertainty notes, or provenance requirements where
  required by the asset contract
- invalid applicability-scope claims

## Design Constraints

- audit rules must distinguish canonical shared assets from family-local notes
- family-local notes are allowed, but only in sanctioned fields
- cross-family checks must be strict enough to prevent ECG-style overfitting
  from leaking into shared vocabulary

## Testing And Validation

- unit tests for each new audit finding
- golden-asset tests for accepted heuristic assets
- deliberately invalid fixture assets triggering expected findings

## Risks

- audit rules that are too weak to protect the shared vocabulary
- audit rules that are so rigid they block reasonable family-local explanation

## Exit Criteria

- heuristic assets are audited with explicit cross-family and dejargonization
  enforcement
- failures are surfaced in the same review workflow as other atom-repo audits

## Dependencies

- depends on Phase 1
- can run in parallel with Phases 3 and 4 once the schema stabilizes

