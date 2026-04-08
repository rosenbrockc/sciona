# Signal Phase 1: Stage Semantics And Boundary Contracts

## Status

Drafted on April 7, 2026 as Phase 1 of the signal-processing expansion
implementation plan in
[SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md).

## Purpose

Signal-processing expansion will stay brittle unless the framework has a stable
semantic model for:

- stage roles
- boundary transitions
- edge meaning
- allowed information loss
- provenance preservation

This phase defines that model.

## Problem

The current signal-family skeletons are useful, but still too coarse as a
semantic contract for expansion. In particular:

- stage identity is still too dependent on naming
- boundary semantics are incomplete
- loss rules are not strongly encoded
- current graphs do not expose enough meaning for robust applicability checks

This is the phase that gives the signal family a real semantic substrate.

## Goals

1. Define the canonical stage taxonomy for signal-processing families.
2. Define boundary and edge semantics for signal-family CDGs.
3. Define signal-family loss classes and preservation rules.
4. Define the compatibility layer from the current skeleton assets to the new
   semantic model.

## Stage Taxonomy

This phase should define canonical signal-family stage roles such as:

- acquisition or normalization
- conditioning or denoising
- transformation
- event detection or segmentation
- event correction or reconciliation
- measure or rate estimation
- quality validation
- smoothing or aggregation
- comparison or branch selection

The exact family may use only a subset, but the vocabulary should be stable.

## Boundary Model

The phase should define first-class signal-family boundaries such as:

- root waveform input
- conditioned waveform boundary
- representation-change boundary
- event-sequence boundary
- measure-output boundary
- validation sidecar boundary

This model should be explicit enough that later phases can target:

- before first consumer of waveform input
- after detector but before estimator
- between lossy and irreversible transitions

## Edge Semantics

Each signal-family edge should be able to declare:

- data kind
- provenance
- time basis
- alignment requirements
- loss class
- admissibility notes

Example loss classes that should be explicitly modeled:

- preserving
- lossy-but-allowed
- irreversible
- corrective
- validation-only

## Deliverables

1. A signal-family semantic vocabulary document or schema.
2. Canonical stage-role definitions.
3. Canonical boundary definitions.
4. Canonical edge semantic definitions.
5. Compatibility rules for current signal skeleton assets.
6. Example mappings for existing signal-family skeletons.

## Implementation Work

### Workstream A: Taxonomy definition

- define the stage-role set
- define minimal required inputs and outputs for each role
- define common role confusions and disambiguation rules

### Workstream B: Boundary and edge contract model

- define boundary anchors and port semantics
- define edge metadata fields
- define how provenance and time basis should be represented

### Workstream C: Compatibility mapping

- map current `signal_detect_measure` style skeletons onto the new semantic
  model
- identify where compatibility shims are required

### Workstream D: Example family mappings

- provide worked examples for:
  - waveform to event-rate family
  - transform to detector family
  - multi-stage filtered estimation family

## Testing Strategy

- unit tests for stage-role inference
- serialization tests for signal-family semantic edges
- compatibility tests for current signal skeleton assets
- explainability tests showing why a node or edge was classified a certain way

## Risks

### Risk: taxonomy becomes too ECG-shaped

Mitigation:

- require examples from multiple signal modalities
- validate against non-ECG signal families before freezing the vocabulary

### Risk: semantics become too abstract to be executable

Mitigation:

- keep every semantic field tied to a concrete downstream consumer:
  expansion, telemetry, admissibility, or synthesis

## Exit Criteria

- the signal family has a stable semantic stage vocabulary
- the framework can classify current signal skeleton stages against that
  vocabulary
- boundary-aware reasoning is possible for signal-family graphs
- later phases can target semantic boundaries without inventing their own stage
  model
