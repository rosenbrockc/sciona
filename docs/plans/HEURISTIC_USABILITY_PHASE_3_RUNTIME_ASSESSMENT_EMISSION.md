# Heuristic Usability Phase 3: Runtime Assessment Emission

## Purpose

Emit deterministic `usability_assessment` artifacts during runtime evaluation
and cohort scoring.

This phase turns the model into operational evidence that other systems can
consume without re-deriving usability ad hoc.

## Problem

Right now, runtime artifacts may contain heuristics and telemetry, but they do
not yet contain a first-class usability decision. As a result:

- cohort selection relies on procedural checks
- scoring and benchmark logic must infer usability indirectly
- long-term memory cannot capture the usability decision directly

## Scope

This phase covers:

- runtime assessment builder
- persistence into runtime artifacts and cohort artifacts
- integration with current heuristic extraction surfaces

## Non-Goals

This phase does not:

- change proposal or benchmark policy yet
- implement long-term storage
- add final audit enforcement

## Deliverables

1. Runtime usability assessment builder logic.
2. `usability_assessment` persisted into runtime evidence.
3. Cohort artifact support for per-member usability records.
4. Compatibility handling for runs with incomplete heuristic evidence.

## Required Runtime Behaviors

The runtime path should:

- consume canonical heuristics plus family registry rules
- emit explicit scope decisions
- emit blocking and warning reasons
- preserve provenance and confidence
- tolerate incomplete evidence without crashing

## Implementation Work

### Workstream A: Assessment builder

- evaluate registry rules against runtime heuristic evidence
- compute scope decisions and reasons

### Workstream B: Artifact persistence

- write `usability_assessment` into runtime evidence
- include per-member assessment in cohort runs

### Workstream C: Compatibility and fallback

- define behavior for incomplete heuristics or missing family registries
- keep artifacts inspectable even when the assessment is uncertain

## Testing Strategy

- unit tests for rule-to-assessment evaluation
- runtime artifact tests proving the assessment is emitted
- cohort tests proving per-member assessments are preserved
- fixture tests for incomplete evidence and fallback behavior

## Risks

- runtime emission may silently collapse uncertainty into hard decisions
- missing heuristics may cause over-rejection
- artifact shape may become too family-specific

## Exit Criteria

- evaluated runs emit `usability_assessment`
- cohort member artifacts include per-member usability records
- incomplete evidence produces explicit uncertainty rather than hidden fallback
- the artifact is stable enough for downstream consumers

## Dependencies

- depends on Phase 1
- benefits from Phase 2 but can begin once the schema is stable

## Parallelization Notes

- can run in parallel with Phase 2 after Phase 1
- must be meaningfully complete before Phase 4 or Phase 5

