# Heuristic Usability Implementation Plan

## Status

Drafted on April 9, 2026 as the execution-oriented companion to
[HEURISTIC_USABILITY_ASSESSMENT_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_USABILITY_ASSESSMENT_PLAN.md).

This document is for coding agents and implementation sequencing. The high-level
plan remains the ground truth for scope and intent.

## Purpose

Turn heuristic-based usability from a planning concept into an implemented,
auditable subsystem that:

- derives usability from first-class heuristic evidence
- distinguishes guidance, scoring, and final-benchmark usability
- persists usability decisions into long-term outcome memory
- keeps the shared interface cross-family and de-jargonized

## Phase Set

1. [HEURISTIC_USABILITY_PHASE_1_CANONICAL_MODEL.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_USABILITY_PHASE_1_CANONICAL_MODEL.md)
2. [HEURISTIC_USABILITY_PHASE_2_FAMILY_RULE_REGISTRIES.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_USABILITY_PHASE_2_FAMILY_RULE_REGISTRIES.md)
3. [HEURISTIC_USABILITY_PHASE_3_RUNTIME_ASSESSMENT_EMISSION.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_USABILITY_PHASE_3_RUNTIME_ASSESSMENT_EMISSION.md)
4. [HEURISTIC_USABILITY_PHASE_4_COHORT_AND_BENCHMARK_INTEGRATION.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_USABILITY_PHASE_4_COHORT_AND_BENCHMARK_INTEGRATION.md)
5. [HEURISTIC_USABILITY_PHASE_5_LONG_TERM_MEMORY.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_USABILITY_PHASE_5_LONG_TERM_MEMORY.md)
6. [HEURISTIC_USABILITY_PHASE_6_AUDIT_AND_GOVERNANCE.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_USABILITY_PHASE_6_AUDIT_AND_GOVERNANCE.md)

## Dependency Structure

### Critical path

- Phase 1 must land first.
- Phase 4 depends on meaningful progress in Phases 2 and 3.
- Phase 5 depends on Phase 3 and should ideally follow the first stable Phase 4
  integration so memory reflects real operational usage.

### Parallel wave after Phase 1

Phases 2 and 3 can proceed in parallel.

- Phase 2 defines the declarative family rule surface.
- Phase 3 implements runtime production and persistence of usability
  assessments.

### Parallel wave after Phase 4 begins to stabilize

Phases 5 and 6 can proceed in parallel.

- Phase 5 persists usability outcomes and integrates reporting.
- Phase 6 adds audit enforcement and governance for the new assets.

## Recommended Execution Order

1. Phase 1
2. Phases 2 and 3 in parallel
3. Phase 4
4. Phases 5 and 6 in parallel

## Cross-Family Guardrails

Every phase must preserve the following:

- shared usability fields remain de-jargonized
- family registries may interpret heuristics locally but may not redefine shared
  heuristic or usability meaning
- runtime artifacts remain family-neutral in shape
- memory records store explicit evidence and decisions, not opaque labels
- audit rules enforce portability rather than signal-specific expectations

## Coding-Agent Guidance

- Do not implement ECG-only usability labels or thresholds as shared concepts.
- Prefer reusable terms like `required_input_missing`, `coverage_insufficient`,
  and `timing_context_incoherent`.
- Treat `usable_for_guidance`, `usable_for_scoring`, and
  `usable_for_final_benchmark` as separate first-class outputs.
- Preserve compatibility where feasible, but do not let compatibility aliases
  become the new source of truth.
- Add regression tests in each phase proving the interface still works for at
  least one non-signal family or neutral fixture.

