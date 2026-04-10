# Heuristic Usability Phase 2: Family Rule Registries

## Purpose

Add declarative family-level usability rules that map heuristic evidence and
required contracts into usability assessments.

This phase localizes interpretation without polluting the shared schema.

## Problem

Even with a canonical usability schema, the framework still needs a family-safe
way to express:

- which heuristics matter for a given family
- which contracts are mandatory for guidance, scoring, or benchmark acceptance
- which conditions are warnings versus blockers

Without a declarative rule layer, those choices will drift back into runtime
code and become hard to audit.

## Scope

This phase covers:

- family usability rule schema
- registry loading and validation
- representative rule assets for at least one signal family and one non-signal
  family or neutral fixture family

## Non-Goals

This phase does not:

- emit runtime usability artifacts
- change benchmark behavior yet
- persist long-term memory
- add audit enforcement beyond basic validation

## Deliverables

1. A family usability rule schema.
2. Loader and validation support in matcher.
3. Representative registry assets for initial families.
4. Compatibility notes explaining how current family logic maps to these rules.

## Required Rule Concepts

The registry model should support:

- required contracts
- blocking conditions
- warning conditions
- scope-specific decisions
- fallback behavior for incomplete evidence
- local rationale and notes

## Design Constraints

- Family registries may interpret heuristics locally but may not redefine
  shared heuristic or usability meaning.
- Rule IDs and fields must remain cross-family in shape even when thresholds are
  family-specific.
- The schema should support future asset ownership in `../ageo-atoms`.

## Implementation Work

### Workstream A: Registry schema

- define the declarative model for usability rules
- specify scope-specific rule outputs

### Workstream B: Loader integration

- load rule registries in matcher
- validate required schema and family compatibility

### Workstream C: Initial registry assets

- implement signal-family usability rules
- implement at least one contrasting non-signal or neutral registry fixture

## Testing Strategy

- schema validation tests
- registry loading tests
- tests proving family registries cannot redefine shared heuristic meaning
- example-driven tests for blocking, warning, and partial usability decisions

## Risks

- family rules may become a back door for domain jargon
- rules may overfit one benchmark rather than describing family behavior
- unsupported conditions may be encoded informally in notes instead of schema

## Exit Criteria

- matcher can load and validate family usability registries
- at least one real family registry exists
- at least one non-signal or neutral example proves the interface is portable
- rule evaluation inputs are clear enough for runtime integration

## Dependencies

- depends on Phase 1

## Parallelization Notes

- can run in parallel with Phase 3 once Phase 1 stabilizes

