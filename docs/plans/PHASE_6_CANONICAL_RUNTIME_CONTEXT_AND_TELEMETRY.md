# Phase 6: Canonical Runtime Context And Refinement Telemetry

## Status

Drafted on April 7, 2026 as the sixth implementation phase of
[Constraint-Driven Synthesis And Refinement Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md).

This phase assumes the earlier planning, asset, graph, and admissibility work
exists or is sufficiently defined to tell the telemetry system what evidence is
actually useful.

## Purpose

The framework cannot refine or reject candidates reliably if runtime evidence is
missing, inconsistent, or keyed incorrectly.

The ECG investigation exposed three concrete failures:

- important signals were present under dataset-specific aliases instead of
  canonical keys
- canonical sampling metadata was derived from the wrong stream
- reference-loss profiling bypassed the richer trace/evaluation path

Phase 6 makes runtime context canonical and telemetry refinement-oriented so the
rest of the framework can reason from reliable evidence.

## Goal

Build a canonical runtime context and telemetry surface that:

- preserves the right semantics across datasets
- emits compact but useful intermediate summaries
- supports refinement, admissibility, and attribution
- remains inspectable and stable across evaluation paths

The phase should move the system from:

- ad hoc runtime artifacts

to:

- canonical, structured, semantically meaningful runtime evidence

## Non-Goals

Phase 6 should not:

- store every intermediate tensor by default
- become a generic observability platform for all purposes
- replace benchmark evaluation with telemetry alone

Its purpose is targeted support for planning, refinement, and search discipline.

## Problem Statement

The current runtime evidence is too weak in several ways:

- key names depend too much on dataset-specific conventions
- canonical fields are populated opportunistically rather than correctly
- telemetry often captures only final outputs or timing
- reference-loss paths and trace-based paths are inconsistent
- downstream rule systems cannot depend on a stable evidence schema

As long as that remains true, the framework will keep making search and
refinement decisions with partial or misleading context.

## Canonical Runtime Context

This phase should define a canonical runtime context model that is distinct from
raw dataset frames and distinct from raw subprocess traces.

The context should include semantically stable concepts such as:

- canonical inputs by data kind
- per-stream provenance
- per-stream sampling or time basis
- canonical outputs by stage or edge role
- declared aliases and how they were resolved

The context should support multi-stream problems without accidentally collapsing
everything into one global `signal` or one global `sampling_rate`.

## Telemetry Philosophy

The telemetry system should optimize for:

- semantic usefulness
- compactness
- determinism
- explainability

It should generally prefer summary metrics over raw bulk persistence.

Examples of useful summaries:

- waveform quantiles and artifact measures
- discontinuity counts
- event counts and density
- interval median, MAD, and outlier fraction
- parameter choices used by detectors
- plausibility summaries of downstream outputs

These are the kinds of facts the refinement engine and admissibility layer can
consume directly.

## Standardization Requirements

Phase 6 should define:

- canonical field names
- required vs optional telemetry fields
- family-extension conventions
- provenance and stream identity rules
- how telemetry maps back to nodes, edges, or boundaries

The point is not to force every family into identical metrics. The point is to
create a common base that families can extend predictably.

## Evaluation-Path Unification

A major issue exposed by the ECG investigation is that some profiling and
reference-loss paths bypass richer trace collection and artifact persistence.

Phase 6 should define a unified evidence contract so that:

- evaluation paths produce comparable runtime artifacts
- reference-loss attribution can still access useful intermediate summaries
- profiling does not silently lose the telemetry needed for refinement

This does not necessarily mean all paths execute identically, but they should
produce compatible evidence surfaces.

## Relationship To Admissibility And Refinement

Phase 6 is the evidence layer beneath:

- admissibility gates
- expansion diagnostics
- attribution
- benchmark analysis

Those systems should not need to reconstruct meaning from raw arrays or
dataset-specific frame names. They should consume the canonical runtime context
and summary telemetry defined here.

## Suggested Deliverables

Phase 6 should produce:

1. A canonical runtime context contract.
2. Canonical aliasing and provenance-resolution logic for multi-stream inputs.
3. A structured telemetry summary contract for nodes, edges, or boundaries.
4. Unified runtime-artifact generation across evaluation and profiling paths.
5. Family-extension points for additional telemetry summaries.
6. Maintainer documentation for canonical evidence expectations.

## Testing Strategy

### Context Tests

Verify:

- canonical field resolution is stable across representative datasets
- provenance and sampling metadata are correctly assigned per stream
- dataset-specific aliases map into canonical keys predictably

### Telemetry Tests

Verify:

- compact summaries are emitted with stable schema
- node or edge mapping is preserved
- optional family-specific extensions do not break the base contract

### Evaluation Integration Tests

Verify:

- profiling and evaluation paths produce compatible runtime artifacts
- admissibility and refinement logic can consume the artifacts without
  dataset-specific hacks

## Risks

### Risk: Telemetry grows too large or too slow

Mitigation:

- prioritize summary metrics over raw values
- make raw capture opt-in when needed

### Risk: Canonical context becomes too lossy for complex families

Mitigation:

- keep explicit provenance and stream identity
- allow extension fields rather than flattening everything into one namespace

### Risk: Evaluation-path unification becomes invasive

Mitigation:

- unify artifact contracts before unifying every execution detail

### Risk: Family-specific metrics become a schema explosion

Mitigation:

- maintain a small canonical base
- allow extension namespaces with explicit ownership

## Exit Criteria

Phase 6 should be considered complete when:

- the framework has a canonical runtime context for multi-stream algorithm
  evaluation
- canonical keys are stable and semantically correct
- compact telemetry summaries are available for refinement-oriented reasoning
- profiling and evaluation paths produce compatible runtime evidence
- downstream refinement and admissibility systems can consume runtime artifacts
  without dataset-specific glue

## Deferred To Later Phases

Phase 6 does not finish:

- benchmark policy redesign
- all future family-specific telemetry extensions

Those efforts should build on the canonical evidence contract defined here.

## Recommended Sequencing

1. define the canonical runtime context contract
2. fix aliasing and provenance resolution
3. define the base telemetry summary schema
4. unify artifact contracts across execution paths
5. add family-extension hooks
6. document and regression-test the result

## Relationship To Other Phases

Phase 6 provides the evidence layer that makes Phase 5 admissibility and future
refinement behavior reliable instead of heuristic and dataset-fragile.
