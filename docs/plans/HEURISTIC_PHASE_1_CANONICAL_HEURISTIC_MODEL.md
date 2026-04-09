# Phase 1: Canonical Heuristic Model

## Purpose

Define the first-class, de-jargonized heuristic schema that every later phase
will use.

This phase establishes the shared language for heuristic evidence across all
families. It is the main guardrail against the framework drifting into a
signal-specific or ECG-specific abstraction.

## Problem Statement

The framework currently mixes together:

- runtime summaries
- family-local diagnostics
- proposal applicability logic

That makes it difficult to reuse refinement reasoning across families. The same
underlying observation may be named or encoded differently in each family, and
some observations are hidden inside rule code instead of being explicit assets.

## Scope

This phase covers:

- canonical heuristic schema design
- de-jargonized naming conventions
- confidence and uncertainty model
- action-class vocabulary
- compatibility mapping from current diagnostics into the new model

## Non-Goals

This phase does not:

- implement runtime extraction
- modify benchmark policy
- introduce family-specific registries
- move assets into `../ageo-atoms`

## Deliverables

1. A canonical heuristic schema for first-class evidence artifacts.
2. A naming standard for shared heuristic identifiers.
3. A small action-class taxonomy that is family-neutral.
4. A compatibility document mapping current expansion diagnostics and runtime
   summaries into the new heuristic model.
5. Validation examples from multiple families, including but not limited to
   signal processing.

## Required Design Decisions

The phase must resolve:

### Shared heuristic fields

At minimum, define:

- `heuristic_id`
- `display_name`
- `dejargonized_meaning`
- `value_kind`
- `value_shape`
- `confidence`
- `uncertainty_notes`
- `producer_kind`
- `applicability_scope`
- `supported_action_classes`
- `provenance_requirements`

### Naming policy

The naming policy should prefer behavior-based identifiers such as:

- `interval_instability`
- `dominant_nuisance_structure`
- `density_collapse`
- `boundary_discontinuity`
- `confidence_instability`

It should explicitly reject family-jargon identifiers in the shared schema.

### Action classes

The schema should support a small generic set of action classes, such as:

- `precondition`
- `replace_stage`
- `split_stage`
- `insert_correction`
- `gate_or_validate`
- `smooth_or_aggregate`
- `branch_and_compare`

### Confidence and uncertainty

The schema must specify how to express:

- confidence in the heuristic value
- uncertainty in interpretation
- insufficient evidence
- conflict between heuristic producers

## Implementation Considerations

- The schema should be serializable into runtime evidence, benchmark artifacts,
  and asset metadata without forcing family-specific extension points.
- The compatibility mapping should identify which existing diagnostics are:
  - directly portable
  - family-jargon aliases
  - too narrow and should be retired
- The examples should include at least one non-signal family to test the
  vocabulary.

## Testing Strategy

This phase should be validated by:

- schema round-trip tests
- identifier linting tests
- compatibility tests for legacy diagnostics
- review examples that prove the terms remain meaningful across multiple
  families

## Risks

- The schema may become too broad to guide implementation.
- The naming layer may hide family meaning rather than clarifying it.
- Confidence semantics may be too vague to support deterministic use.

## Exit Criteria

Phase 1 is complete when:

- a canonical heuristic schema is documented and implemented
- the schema can represent existing diagnostic concepts cleanly
- the naming policy is explicit and enforceable
- at least two families can explain their local needs using the shared terms
