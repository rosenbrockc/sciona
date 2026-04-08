# Signal Phase 5: E2E Validation And Asset Migration

## Status

Drafted on April 7, 2026 as Phase 5 of the signal-processing expansion
implementation plan in
[SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md).

## Purpose

This phase validates the full signal-family expansion stack against real e2e
paths and prepares stable family assets for migration toward `../ageo-atoms`.

## Problem

A signal-family expansion system is not complete when:

- assets exist
- telemetry exists
- proposals can be selected

It is complete when enriched signal-family CDGs:

- are selected for the right reasons
- synthesize and export cleanly
- run through real datasets and evaluation paths
- are stable enough to be promoted into shared family ownership

## Goals

1. Validate the signal-family expansion stack on real benchmarks and datasets.
2. Confirm enriched CDGs outperform or out-robust the minimal skeleton path.
3. Harden regression tests and benchmark policy for enriched CDGs.
4. Prepare stable family assets for migration into `../ageo-atoms`.

## Deliverables

1. A signal-family benchmark and acceptance suite for enriched CDGs.
2. Regression coverage for proposal selection and runtime execution.
3. Asset migration criteria and packaging rules.
4. A migration shortlist of signal-family skeleton and expansion assets.

## Implementation Work

### Workstream A: Benchmark hardening

- add signal-family benchmark cases that require enrichment
- verify the benchmark uses the full framework path

### Workstream B: Execution-path validation

- validate run, synthesize, export, and profile paths for enriched CDGs
- validate runtime telemetry and proposal traces remain coherent

### Workstream C: Regression and policy

- add regression tests for harmful or missing enrichments
- define benchmark policies for acceptable enrichment behavior

### Workstream D: Asset migration readiness

- identify stable assets ready for shared ownership
- define packaging, review, and migration checklist

## Migration Target

The end of this phase should not require immediate migration of everything, but
it should leave the repository with a clear set of signal-family assets that are
ready to move into `../ageo-atoms`.

## Testing Strategy

- benchmark acceptance tests
- end-to-end run/synthesize/export/profile tests
- trial-history and telemetry-path tests
- migration-readiness validation for candidate assets

## Risks

### Risk: validation remains benchmark-fragile

Mitigation:

- use multiple signal-family cases, not one benchmark only
- validate both structure and executable behavior

### Risk: migration is attempted before assets are stable

Mitigation:

- require explicit readiness criteria
- keep unstable assets local until they satisfy that checklist

## Exit Criteria

- signal-family enriched CDGs are exercised in real e2e validation
- regression coverage exists for enrichment behavior
- stable family assets are identified and migration-ready
- the signal-processing family becomes the first strong implementation of the
  broader expansion strategy
