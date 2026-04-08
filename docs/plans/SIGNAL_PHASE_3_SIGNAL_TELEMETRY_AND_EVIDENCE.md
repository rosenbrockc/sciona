# Signal Phase 3: Signal Telemetry And Evidence Contract

## Status

Drafted on April 7, 2026 as Phase 3 of the signal-processing expansion
implementation plan in
[SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md).

## Purpose

Signal-family expansion will only become useful if it can consume compact,
standardized evidence about what happened at intermediate stages.

This phase defines that evidence contract.

## Problem

The framework often knows too little about the intermediate behavior of a
signal-processing CDG:

- whether a detector collapsed
- whether a representation change introduced implausible loss
- whether a quality gate would have been justified
- whether a downstream estimator received enough support

Without stage-local evidence, expansion cannot distinguish:

- a plausible candidate
- an incomplete candidate
- a harmful candidate

## Goals

1. Define a compact telemetry model for signal-family stages.
2. Standardize summaries at stage and edge boundaries.
3. Make those summaries consumable by expansion assets and proposal selection.
4. Persist enough evidence to explain why an enrichment was or was not applied.

## Evidence Classes

This phase should define signal-family evidence classes such as:

- waveform summaries
- transformed representation summaries
- event-sequence summaries
- support and plausibility summaries
- measure-output summaries
- validation and quality summaries

The emphasis should be on compact, standardized summaries rather than large raw
artifacts.

## Likely Signal-Family Summary Types

Examples of useful summaries include:

- sample count, duration, and sampling context
- distribution summaries
- clipping, discontinuity, or saturation indicators
- local energy or band summaries
- event counts and densities
- interval median, spread, and outlier rate
- estimator support size
- plausibility metrics and missingness
- stage-specific threshold or gating summaries

Different modalities may use different subsets, but the contract should be
common.

## Deliverables

1. A signal-family telemetry vocabulary.
2. Standardized summary schemas by stage role.
3. Runtime collection hooks for signal-family pipelines.
4. Persistence rules for refinement-oriented evidence.
5. Integration points for expansion applicability and proposal ranking.

## Implementation Work

### Workstream A: Summary schema design

- define per-stage-role summaries
- define edge-transition summaries
- define naming and persistence conventions

### Workstream B: Runtime instrumentation

- add summary extraction for signal-family stages
- add persistence into runtime artifacts and trace outputs

### Workstream C: Consumption interfaces

- expose summary access to expansion applicability logic
- expose summary access to admissibility checks

### Workstream D: Evidence explainability

- ensure diagnostics can cite the specific summaries that justified an
  enrichment or rejection

## Testing Strategy

- unit tests for summary generation
- runtime artifact persistence tests
- integration tests showing expansion decisions reading telemetry summaries
- stability tests across multiple signal modalities

## Risks

### Risk: telemetry becomes too verbose

Mitigation:

- optimize for compact summaries first
- keep raw intermediate storage optional

### Risk: summaries become too ECG-specific

Mitigation:

- require at least one non-ECG signal family when finalizing the contract

## Exit Criteria

- signal-family stages emit standardized evidence summaries
- expansion and admissibility code can read those summaries
- the framework can explain enrichment decisions using persisted evidence
