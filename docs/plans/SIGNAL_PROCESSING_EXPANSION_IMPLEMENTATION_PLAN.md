# Signal Processing Expansion Implementation Plan

## Status

Drafted on April 7, 2026 as the signal-processing-specific implementation plan
derived from:

- [Constraint-Driven Synthesis And Refinement Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md)
- [Cross-Family Expansion And CDG Enrichment Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CROSS_FAMILY_EXPANSION_ENRICHMENT_PLAN.md)

This document is intentionally focused on the signal-processing family cluster
as the first serious proving ground for cross-family CDG enrichment.

## Purpose

Signal processing is the right first family for expansion hardening because it
already exposes the central failure modes clearly:

- minimal skeletons are often conceptually right but structurally too thin
- useful enrichments are common and easy to name
- stage boundaries are meaningful and reusable
- telemetry can be summarized compactly
- the family spans multiple modalities and objectives while still sharing a
  recognisable shape

The goal of this plan is to turn signal-processing expansion into a production
quality implementation of the broader expansion model, not to add ECG-specific
hacks.

## Scope

This plan covers the signal-processing family cluster, including but not limited
to:

- conditioning and filtering pipelines
- detect/measure pipelines
- event-rate pipelines
- multi-stage physiological signal pipelines
- time-domain, frequency-domain, and mixed-domain workflows
- validation, correction, and quality-control enrichments

The design should be reusable across ECG, PPG, EEG, EMG, PCG, motion, and
other sampled-signal families that fit the same high-level semantics.

## Non-Goals

This plan does not attempt to:

- finish all cross-family expansion work
- migrate every existing signal family asset into `../ageo-atoms` immediately
- solve every benchmark problem with one family-specific rewrite inventory
- define the final canonical schema for all families at once

The purpose is to establish the first strong, generalizable implementation.

## Why Signal Processing First

Signal processing is the best first implementation target because it has:

- strong and interpretable stage semantics
- clear information-loss boundaries
- common enrichment patterns
- rich but compact telemetry opportunities
- enough diversity to avoid overfitting to one task

Examples of recurring signal-processing stage patterns:

- acquire or normalize signal
- condition or denoise waveform
- transform representation
- detect or segment events
- correct or reconcile detections
- estimate a derived measure
- validate signal quality or plausibility
- smooth or aggregate downstream outputs

This makes the family ideal for proving:

- semantic stage models
- boundary-aware expansion matching
- asset-backed enrichment
- deterministic proposal selection

## Conceptual Implementation Goal

The desired end state is:

- the Architect emits a signal-family planning artifact with explicit stage and
  loss constraints
- a signal-family skeleton defines the minimal graph
- a signal-family expansion inventory defines sanctioned enrichments
- runtime telemetry provides compact evidence
- the Principal produces enriched candidate CDGs when warranted
- deterministic proposal selection chooses among them
- the best enriched CDG synthesizes and runs through the real benchmark path

## Phase Set

This implementation is broken into five signal-processing phases:

1. Signal Stage Semantics And Boundary Contracts
2. Signal Expansion Asset Library
3. Signal Telemetry And Evidence Contract
4. Signal Proposal Selection And Admissibility
5. Signal E2E Validation And Asset Migration

Each phase is documented separately:

- [SIGNAL_PHASE_1_STAGE_SEMANTICS_AND_BOUNDARIES.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PHASE_1_STAGE_SEMANTICS_AND_BOUNDARIES.md)
- [SIGNAL_PHASE_2_EXPANSION_ASSET_LIBRARY.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PHASE_2_EXPANSION_ASSET_LIBRARY.md)
- [SIGNAL_PHASE_3_SIGNAL_TELEMETRY_AND_EVIDENCE.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PHASE_3_SIGNAL_TELEMETRY_AND_EVIDENCE.md)
- [SIGNAL_PHASE_4_PROPOSAL_SELECTION_AND_ADMISSIBILITY.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PHASE_4_PROPOSAL_SELECTION_AND_ADMISSIBILITY.md)
- [SIGNAL_PHASE_5_E2E_VALIDATION_AND_ASSET_MIGRATION.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PHASE_5_E2E_VALIDATION_AND_ASSET_MIGRATION.md)

## Dependency Structure

The phases are not strictly linear.

### Hard dependencies

- Phase 1 is the semantic foundation for all later phases.
- Phase 2 depends on Phase 1 because expansion assets need stable stage and
  boundary semantics.
- Phase 4 depends on Phases 1, 2, and 3 because proposal selection needs
  semantic applicability, declared enrichments, and usable telemetry.
- Phase 5 depends on Phases 2, 3, and 4 for meaningful end-to-end validation.

### Soft dependencies

- Phase 3 benefits from Phase 1 but can start before Phase 2 is complete.
- Phase 5 can begin benchmark-harness work before all migration work is ready.

## Parallelization Analysis

The right way to execute this work is by dependency wave rather than by a
strict serial queue.

### Wave 0: Foundational alignment

Contains:

- Phase 1 kickoff and semantic contract definition

Reason:

- The other phases need a stable vocabulary for stages, edges, provenance, and
  loss semantics.

Parallelism:

- Internal subtasks inside Phase 1 can run in parallel:
  - stage taxonomy draft
  - boundary representation draft
  - signal-family contract examples
  - compatibility analysis with current skeleton assets

### Wave 1: Asset and telemetry build-out

Contains:

- Phase 2
- Phase 3

Reason:

- Once the semantic model is stable enough, asset authoring and telemetry
  standardization can proceed in parallel.

Why these can overlap:

- Phase 2 defines what enrichments exist.
- Phase 3 defines what evidence those enrichments can consume.

They are coupled conceptually, but the work is separable:

- asset authors can define applicability and expected evidence
- telemetry work can define the generic summaries and collection hooks

Integration point:

- the teams need a shared checklist so every new expansion asset declares the
  telemetry summaries it expects.

### Wave 2: Decision logic

Contains:

- Phase 4

Reason:

- Proposal selection only becomes meaningful once the family has:
  - a real enrichment inventory
  - telemetry summaries
  - a semantic applicability model

Parallelism:

- Some sub-work inside Phase 4 can run in parallel:
  - admissibility-gate implementation
  - proposal materialization
  - proposal scoring and selection telemetry
  - failure explanation/reporting

### Wave 3: Validation and migration

Contains:

- Phase 5

Reason:

- The final phase should validate the whole stack against real signal-family
  benchmarks and then move stable assets toward `../ageo-atoms`.

Parallelism:

- Phase 5 can split into two tracks after early validation is green:
  - benchmark and regression hardening
  - asset packaging and migration work

### Summary Table

| Phase | Can start after | Can run in parallel with |
| --- | --- | --- |
| Phase 1 | immediately | internal Phase 1 subtasks only |
| Phase 2 | Phase 1 semantic baseline | Phase 3 |
| Phase 3 | Phase 1 semantic baseline | Phase 2 |
| Phase 4 | substantial Phase 2 and Phase 3 progress | internal Phase 4 subtasks |
| Phase 5 | usable Phase 4 proposal flow | internal validation/migration tracks |

## Recommended Execution Strategy

The recommended work ordering is:

1. Establish a stable Phase 1 semantic contract.
2. Start Phase 2 and Phase 3 together.
3. Freeze a first candidate signal-family asset inventory and telemetry contract.
4. Build Phase 4 proposal selection around those contracts.
5. Use Phase 5 to harden the full path and prepare migration into
   `../ageo-atoms`.

This sequencing avoids the common failure mode of building proposal-selection
logic before the enrichment inventory and evidence model are mature enough.

## Success Criteria

This signal-processing implementation will be successful when:

- the framework can produce enriched signal-family CDGs beyond the minimal
  skeleton
- those enrichments are asset-backed and auditable
- they apply because of explicit evidence, not implicit heuristics
- proposal selection can explain why one enriched CDG was chosen over another
- synthesized artifacts run through the real benchmark/evaluation path
- stable family assets are ready to move toward `../ageo-atoms`

## Intended Use

This document should be used as:

- the top-level execution map for signal-processing expansion work
- a dependency and parallelization reference
- the coordination layer above the phase-specific signal-processing plans
