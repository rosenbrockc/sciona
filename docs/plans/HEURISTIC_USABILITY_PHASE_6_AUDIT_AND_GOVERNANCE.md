# Heuristic Usability Phase 6: Audit And Governance

## Purpose

Add audit enforcement and governance rules for usability assets so the new
layer remains cross-family, de-jargonized, and evidence-backed.

This phase prevents usability from devolving into hidden family-local filter
logic.

## Problem

Without audit enforcement, usability assets could drift toward:

- domain jargon in shared reason codes
- weak or missing rationale for blocking decisions
- family registries redefining shared meaning
- implicit thresholds buried in notes instead of structured fields
- opaque memory records that cannot be audited

## Scope

This phase covers:

- audit rules for usability schemas and registries
- governance for shared reason codes and scope semantics
- validation of required rationale, provenance, and uncertainty notes

## Non-Goals

This phase does not:

- redesign runtime logic
- redesign benchmark ranking
- replace general heuristic auditing already covered elsewhere

## Deliverables

1. Audit rules for canonical usability assets.
2. Audit rules for family usability registries.
3. Governance checks for de-jargonization and cross-family portability.
4. Regression fixtures covering representative accepted and rejected assets.

## Required Audit Checks

- schema validity
- duplicate or conflicting reason codes
- banned or suspicious domain jargon in shared fields
- missing rationale for blocking or warning decisions
- family-local attempts to redefine shared usability semantics
- unsupported scope claims
- missing provenance or uncertainty notes where required

## Design Constraints

- canonical shared fields must be audited more strictly than family-local notes
- audit rules must allow local explanation while preventing shared-interface
  drift
- checks should align with eventual ownership in `../ageo-atoms`

## Testing Strategy

- unit tests for each audit rule
- golden accepted assets
- deliberately invalid fixtures for jargon leakage and semantic redefinition
- compatibility tests proving legacy assets fail clearly when migration is
  incomplete

## Risks

- audit rules may be too weak to stop domain leakage
- audit rules may become so rigid they block reasonable local explanation
- governance may focus on wording and miss semantic misuse

## Exit Criteria

- usability assets are audited with explicit cross-family enforcement
- shared reason codes and scope semantics are protected by tooling
- invalid family-local redefinitions are surfaced in the normal review workflow

## Dependencies

- depends on Phase 1
- benefits from concrete assets from Phases 2 through 5

## Parallelization Notes

- can begin after Phase 1 with fixture design
- should run in parallel with Phase 5 once real asset shapes are stable

