# Heuristic Usability Phase 1: Canonical Model

## Purpose

Define the first-class, cross-family schema for heuristic-derived usability
assessment.

This phase establishes the shared contract that all later runtime, registry,
benchmark, and memory work will use.

## Problem

The framework currently treats usability as an implicit procedural outcome of
evaluation rather than as an explicit evidence-backed artifact. That makes it
difficult to:

- distinguish guidance-only data from benchmark-valid data
- explain why a dataset member was included or excluded
- carry usability decisions into long-term memory
- keep family-specific logic from leaking into shared interfaces

## Scope

This phase covers:

- canonical usability schema design
- usability scope taxonomy
- canonical reason-code vocabulary
- compatibility mapping from current ad hoc usable-member checks

## Non-Goals

This phase does not:

- implement family rule registries
- emit runtime assessments
- change cohort or benchmark behavior
- persist new memory records

## Deliverables

1. A canonical `usability_assessment` schema.
2. A shared taxonomy for usability scopes:
   - `usable_for_guidance`
   - `usable_for_scoring`
   - `usable_for_final_benchmark`
3. A de-jargonized blocking/warning reason vocabulary.
4. Compatibility notes mapping current procedural checks into the new schema.
5. Validation examples spanning at least one non-signal family or neutral
   synthetic case.

## Required Design Decisions

### Shared fields

At minimum, define:

- `assessment_id`
- `family`
- `task_intent`
- `heuristic_signature`
- `required_contracts_checked`
- `usable_for_guidance`
- `usable_for_scoring`
- `usable_for_final_benchmark`
- `blocking_reasons`
- `warning_reasons`
- `confidence`
- `uncertainty_notes`
- `provenance`

### Reason vocabulary

The canonical vocabulary should prefer cross-family labels such as:

- `required_input_missing`
- `required_reference_missing`
- `coverage_insufficient`
- `timing_context_incoherent`
- `alignment_error`
- `quality_instability`
- `plausibility_fragmentation`
- `output_density_collapse`

It should explicitly reject domain-local labels in shared fields.

### Decision semantics

The phase must define:

- what counts as a blocking vs warning reason
- whether scopes may disagree
- how uncertainty should affect the three scope decisions
- how multiple reasons are ordered and serialized

## Implementation Considerations

- The schema should serialize cleanly into runtime artifacts, benchmark
  summaries, and long-term memory.
- The design should anticipate migration of canonical usability assets into
  `../ageo-atoms`.
- The compatibility mapping should identify which current checks can be retired
  once runtime assessment exists.

## Testing Strategy

- schema validation and round-trip tests
- reason-code linting tests
- compatibility tests for current usable-member logic
- fixture examples proving the model still reads sensibly outside signal
  processing

## Risks

- the model may collapse into a binary good/bad interface
- shared reasons may become too signal-flavored
- confidence semantics may be too vague to support deterministic use

## Exit Criteria

- a canonical usability schema is implemented and documented
- reason codes are explicit and lintable
- the three-scope model is represented cleanly
- compatibility mapping exists for current cohort/benchmark behavior

## Dependencies

- no upstream dependency
- blocks all later phases

## Parallelization Notes

- must complete before any parallel work begins

