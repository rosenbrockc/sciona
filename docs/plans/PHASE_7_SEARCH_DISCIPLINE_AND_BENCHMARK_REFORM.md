# Phase 7: Search Discipline And Benchmark Reform

## Status

Drafted on April 7, 2026 as the seventh implementation phase of
[Constraint-Driven Synthesis And Refinement Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md).

This phase assumes the earlier planning, asset, graph, admissibility, and
telemetry work is in place or at least stable enough to evaluate the framework
as a real synthesis system instead of a chain of ad hoc helpers.

## Purpose

The benchmark and search experience should validate the actual framework:

- real family knowledge
- real candidate pruning
- real refinement
- real telemetry
- real auditability

If benchmarks reward shortcut paths, exact scaffold matching, or tiny curated
inventories, they fail to measure what the framework is supposed to do.

Phase 7 reforms both the search discipline and the benchmark discipline so the
system is judged on robust synthesis behavior rather than superficial success.

## Goal

Make the framework behave like a disciplined search system and make the
benchmarks measure that behavior directly.

The phase should move the project from:

- "did this run happen to succeed"

to:

- "did the framework search, reject, refine, and justify its decisions in a
  robust and generalizable way"

## Non-Goals

Phase 7 should not:

- reintroduce benchmark-only shortcuts
- replace behavioral evaluation with brittle exact-structure assertions
- define every future benchmark for every family

It should establish the discipline and the evaluation policy, not complete every
future case.

## Problem Statement

A framework can appear to do well while still being weak if:

- it relies on narrow curated candidate sets
- it short-circuits real family retrieval
- it is judged by exact scaffold shape instead of semantic quality
- it does not preserve enough artifacts to evaluate search behavior

That kind of benchmark success is misleading. It encourages optimization around
the harness instead of the framework.

## Search Discipline Objectives

Phase 7 should define what good search behavior looks like.

Examples:

- candidate generation is guided by explicit constraints
- inadmissible candidates are pruned early
- refinement is applied for explicit reasons
- family assets are actually used
- decisions are explainable through persisted artifacts
- optimization is not wasted on structurally or semantically hopeless branches

Search quality should become a visible output of the system, not just a hidden
process behind the final score.

## Benchmark Reform Objectives

Benchmarks should measure:

- semantic correctness
- executable correctness
- search discipline
- refinement usage
- admissibility behavior
- auditability of chosen structures

Benchmarks should avoid overcommitting to:

- exact node counts unless intentionally required
- exact primitive identities when multiple semantically valid solutions exist
- harness-only shortcuts

The framework should be rewarded for finding good solutions through the real
stack, not for reproducing a brittle expected graph.

## Benchmark Artifact Expectations

Phase 7 should require benchmarks to preserve enough artifacts to evaluate:

- planning constraints
- chosen skeleton asset
- refinement attempts and outcomes
- admissibility decisions
- runtime telemetry summaries
- final candidate structure and quality

That makes it possible to distinguish:

- a high-quality search that found a weak final result for understandable
  reasons
- a weak search that simply never reasoned correctly

## Assertion Philosophy

The default benchmark philosophy should shift toward fuzzy and behavioral
assertions.

Examples:

- algorithm family structure is semantically appropriate
- outputs satisfy domain-level plausibility and quality criteria
- the run exercised real family assets and the real search stack
- inadmissible candidates were pruned when expected
- refinements were attempted or skipped for explicit reasons

Exact structural assertions should still exist when they protect truly canonical
requirements, but they should not be the default measure of success.

## Anti-Shortcut Policy

Phase 7 should codify an anti-shortcut policy for e2e and benchmark runs.

That policy should strongly discourage:

- curated family bypasses
- narrow hand-maintained candidate sets used only in benchmark mode
- benchmark-only logic that does not reflect real runtime behavior

If a specialized path is needed operationally, the benchmark policy should make
it explicit when that path is being exercised and when it is not.

## Suggested Deliverables

Phase 7 should produce:

1. A search-discipline evaluation policy.
2. A benchmark assertion policy favoring semantic and behavioral checks.
3. Benchmark artifact requirements covering planning, refinement, and
   admissibility behavior.
4. Anti-shortcut guidance for e2e and benchmark paths.
5. Updated representative benchmarks that exercise the real framework.
6. Maintainer documentation explaining how to judge search quality, not just
   final score.

## Testing Strategy

### Benchmark Contract Tests

Verify:

- required benchmark artifacts are present
- benchmarks run through the intended full framework path
- shortcut paths are either disabled or explicitly declared

### Behavioral Regression Tests

Verify:

- semantically valid alternative solutions can still pass
- inadmissible search behavior is surfaced as a benchmark concern
- refinement and admissibility decisions appear in artifacts

### Framework Evaluation Tests

Verify:

- the benchmark can distinguish a robust search from a lucky shortcut
- richer artifacts make diagnosis easier when runs fail

## Risks

### Risk: Benchmarks become too fuzzy

If assertions are too loose, the framework can regress while still passing.

Mitigation:

- keep strong semantic requirements
- preserve exact assertions only for true invariants

### Risk: Search-discipline metrics become noisy or subjective

Mitigation:

- require explicit structured artifacts
- judge behavior through surfaced decisions rather than intuition

### Risk: Benchmark runtime cost increases

Mitigation:

- use compact artifact summaries
- focus on representative full-stack cases rather than many shallow cases

### Risk: Teams reintroduce shortcuts under operational pressure

Mitigation:

- make benchmark policy explicit
- surface when a run used a shortcut path

## Exit Criteria

Phase 7 should be considered complete when:

- representative benchmarks exercise the real framework instead of shortcut
  paths
- benchmarks primarily use semantic and behavioral assertions
- search-discipline artifacts are available and inspectable
- benchmark results expose admissibility and refinement behavior, not only final
  scores
- maintainers can tell whether the framework is improving as a synthesis system,
  not only whether one case happened to pass

## Deferred Work

Phase 7 does not complete every possible benchmark family or metric. It defines
the durable benchmark and search-discipline policy that future cases should
follow.

## Recommended Sequencing

1. define the benchmark and search-discipline policy
2. define required artifacts and anti-shortcut rules
3. update a small representative set of benchmarks
4. revise assertions toward semantic and behavioral checks
5. document maintainer expectations for future benchmarks

## Relationship To Other Phases

Phase 7 is the evaluation layer that tells us whether the earlier phases
actually improved the framework as intended. Without this phase, the project can
still drift back toward harness-optimized behavior.
