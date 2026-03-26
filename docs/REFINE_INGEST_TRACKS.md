# REFINE_INGEST Follow-Up Tracks

This document groups the post-phase-12 follow-up work into execution tracks and
spells out what can run in parallel.

## Phase Map

- phase 13: transitional-surface cleanup
- phase 14: real cache-enabled end-to-end ingest test
- phase 15: CI wiring for the regression corpus
- phase 16: maintainer architecture note
- phase 17: stable public contract definition

## Dependency Analysis

### Track A

- phase 13

This is cleanup work on canonical-first runtime structure. It can run now.

### Track B

- phase 14

This is a narrow cache-enabled integration confidence phase. It can run now and
does not need to wait for phase 13.

### Track C1

- phase 16

This is documentation work and can run now in parallel with phases 13 and 14.

### Track C2

- phase 17

This is contract-definition work and can run now in parallel with phases 13,
14, and 16.

### Track D

- phase 15

This should usually wait until phases 13 and 14 settle. The reason is simple:

- phase 13 may still reduce transitional churn in the protected runtime/test
  paths
- phase 14 adds the missing real cache-enabled integration confidence

CI wiring should reflect the stabilized suite rather than race ahead of those
two tracks.

## Recommended Execution Order

Start immediately in parallel:

- phase 13
- phase 14
- phase 16
- phase 17

Then start:

- phase 15 after phases 13 and 14 are in a good state

## Coordination Notes

- phase 13 owns `models.py`, `chunker.py`, `emitter.py`, and the related runtime
  tests
- phase 14 should stay focused on graph/cache/monitor integration tests
- phase 16 should mostly stay in docs
- phase 17 should mostly stay in docs plus small clarifying code notes if
  needed
- phase 15 should avoid landing until the runtime/test surface it wires into CI
  has settled
