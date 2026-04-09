# AGEO Atoms Heuristic Phase 4: Signal Family Metadata And Registries

## Purpose

Prove the ownership model end-to-end by moving the first concrete family
registry and the first wave of signal-processing atom heuristic metadata into
`../ageo-atoms`.

## Problem

The heuristic layer is only real once a family can use externally owned:

- canonical heuristics
- atom heuristic producers
- family-local heuristic interpretation

Signal processing is the first migration target because it already has the most
concrete heuristic evidence and refinement behavior.

## Scope

- migrate the signal-family heuristic registry into `ageo-atoms`
- add heuristic metadata to selected signal-processing atoms
- define the first sanctioned heuristic-producing signal atoms
- keep the resulting assets cross-family in naming and meaning

## Non-Goals

- migrating every signal-processing atom at once
- expanding to all non-signal families in this phase
- changing global proposal scheduling

## Deliverables

- signal-family heuristic registry asset in `ageo-atoms`
- atom heuristic metadata for an initial set of signal atoms
- references and dejargonized notes for those assets
- matcher integration proving the signal family loads externally

## Selection Guidelines For Initial Atoms

- prefer atoms whose outputs are already used or strongly implied as heuristic
  evidence
- prefer atoms with clear, reusable evidence semantics
- avoid atom metadata that only makes sense for one benchmark or one modality

## Key Design Constraints

- signal-family notes may explain local meaning, but canonical heuristic meaning
  must stay generic
- atom heuristic metadata must describe evidence contracts, not special-case ECG
  policy
- the initial producer set should include both direct atom outputs and room for
  runtime-transform producers

## Testing And Validation

- registry loading tests from `ageo-atoms`
- atom metadata loading tests
- matcher integration tests showing signal-family guidance still works
- negative tests for ECG-specific jargon leaking into canonical fields

## Risks

- overfitting the first family migration to ECG
- writing signal-family notes that redefine shared heuristic meaning
- treating runtime-derived heuristics as if every one requires a dedicated atom

## Exit Criteria

- signal-family registry is externally owned in `ageo-atoms`
- selected signal atoms expose heuristic metadata there
- matcher uses those assets successfully on existing signal-family flows

## Dependencies

- depends on Phases 1 and 2
- benefits from Phase 3 but can start in parallel once the schema is fixed

