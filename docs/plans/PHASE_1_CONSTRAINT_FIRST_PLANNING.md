# Phase 1: Constraint-First Planning Contract

## Status

Drafted on April 7, 2026 as the first implementation phase of
[Constraint-Driven Synthesis And Refinement Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md).

This document is intentionally more concrete than the parent plan, but still
stays above patch-level implementation detail. It is meant to define the first
architectural boundary to build, test, and review independently.

## Why Phase 1 Comes First

The framework currently decomposes a goal into a candidate skeleton before it
has a structured representation of what must be preserved, what loss is
acceptable, and what makes a candidate obviously invalid.

That causes several downstream problems:

- retrieval can choose semantically weak atoms too early
- expansions do not know enough about intended information flow
- evaluators discover obvious failures late
- telemetry has no stable target schema for what should be measured

Phase 1 establishes the missing contract: the system must plan under explicit
constraints before it builds or refines a candidate CDG.

## Goal

Introduce a first-class, persisted planning artifact that captures the
constraint system for a synthesis run and make that artifact part of the
Architect contract.

The system should move from:

- `goal -> skeleton -> candidate CDG`

to:

- `goal -> planning constraints -> skeleton intent -> candidate CDG`

The candidate CDG remains important, but it is no longer the first semantic
artifact produced by the planner.

## Intended Outcome

After Phase 1:

- every decomposition run emits a structured constraint artifact
- Architect prompts and outputs are organized around those constraints
- downstream components can consume the artifact without inferring intent from
  free text
- later phases can build typed skeletons, audited expansions, admissibility
  gates, and telemetry standards on top of a stable contract

Phase 1 does not attempt to solve all downstream issues by itself. Its purpose
is to create the canonical planning surface that later phases can depend on.

## Non-Goals

Phase 1 should not try to complete the rest of the architecture:

- it does not migrate skeletons into `../ageo-atoms`
- it does not redesign the full CDG type system
- it does not rewrite the expansion engine
- it does not add every admissibility gate
- it does not fully solve ECG heart-rate quality

Those belong to later phases. This phase should create the contract those
phases build on.

## Problem Statement

The current Architect runtime produces:

- a selected paradigm
- a skeleton template
- a decomposed CDG

What is missing is a stable planning representation of:

- required data kinds
- stage preconditions
- allowed and forbidden information loss
- provenance expectations
- admissibility expectations
- family-level structural obligations

Those expectations currently exist only implicitly in:

- prompt wording
- internal helper logic
- expansion rule assumptions
- evaluator heuristics
- manual reasoning during debugging

That is too weak. If the framework is supposed to be deterministic and
auditable, those expectations must be structured and inspectable.

## Proposed Artifact

Phase 1 introduces a new persisted planning artifact referred to here as the
constraint artifact.

At a high level, it should include:

- run identity:
  goal, thread id, timestamp, planner version

- planning intent:
  paradigm, family hint, skeleton intent, decomposition rationale

- input and output contracts:
  expected input kinds, output kinds, time basis expectations, provenance hints

- stage constraints:
  what each conceptual stage requires and guarantees

- edge or flow constraints:
  what information may be transformed, preserved, or not discarded across stage
  boundaries

- admissibility expectations:
  conditions that should cause a candidate or partial candidate to be rejected

- telemetry expectations:
  which kinds of runtime summaries should exist later to support refinement

- unresolved questions:
  ambiguities, assumptions, or known risks carried forward from planning

This artifact should be structured enough for downstream use, but compact enough
to remain readable and reviewable.

## Constraint Categories

The constraint artifact should support a small set of canonical categories in
Phase 1. This set should be deliberately limited so the contract is usable
early rather than theoretically complete.

### Data-Kind Constraints

These express what sort of values flow through the algorithm:

- waveform
- event sequence
- rate series
- feature vector
- state
- mask
- parameter
- scalar statistic

The main purpose is to stop later components from treating all arrays as
equivalent just because the Python type is `np.ndarray`.

### Provenance Constraints

These express where a value comes from and which context it must preserve:

- source stream or input binding
- sampling or time basis
- alignment expectations
- whether a downstream stage requires the same provenance

These constraints are critical for multi-stream datasets and any algorithm where
sampling context matters.

### Loss Constraints

These express whether a transformation is allowed to discard information:

- preserving
- lossy but allowed
- irreversible
- alignment-preserving
- monotonic-only

This gives the planner a vocabulary for statements like:

- keep waveform semantics intact until event extraction
- do not apply irreversible compression before detection
- event correction may move peaks locally but should preserve cardinality within
  bounded tolerance

### Stage Preconditions And Guarantees

These express what each conceptual stage expects and what it promises:

- requires conditioned waveform
- requires monotone event ordering
- guarantees event sequence
- guarantees bounded outlier fraction target

This is the bridge between high-level skeleton planning and later admissibility
checks.

### Admissibility Constraints

These express what must be true for a candidate to remain under consideration.

Examples:

- event density must remain plausible for the inferred duration
- a detector that ignores required sampling context is high risk
- a rate estimator must receive a sufficient number of events

Phase 1 does not need to implement all of these checks. It needs to define how
they are represented and passed forward.

### Telemetry Expectations

These express which runtime summaries would be useful for evaluating or
refining the candidate later.

Examples:

- event count
- threshold used
- interval outlier fraction
- discontinuity count
- output plausibility range

The point is not to require all telemetry immediately, but to allow the planner
to state what evidence later phases should collect.

## Architect Contract Changes

Phase 1 should update the Architect contract so the planner explicitly reasons
about constraints before finalizing the first candidate skeleton.

### Current Shape

Today the Architect sequence is roughly:

- select paradigm
- instantiate skeleton
- decompose pending nodes
- validate and critique decomposition
- emit CDG handoff

### Desired Shape

After Phase 1 the sequence should become:

1. interpret goal
2. identify family and planning intent
3. synthesize a constraint artifact
4. select or synthesize skeleton intent under that constraint set
5. decompose nodes while preserving the constraint context
6. emit both the CDG handoff and the constraint artifact

### Minimum Contract Additions

The Architect output should include, directly or through referenced artifacts:

- `planning_constraints`
- `skeleton_intent`
- `admissibility_expectations`
- `planning_assumptions`

These additions should be considered part of the canonical output surface for
decomposition, not optional debug metadata.

## Integration Surfaces

Phase 1 should identify, but not fully redesign, the surfaces that will consume
the new artifact.

### Architect

Primary producer of the constraint artifact.

Responsibilities:

- generate constraints from the goal and family context
- attach the artifact to decomposition outputs
- carry assumptions and unresolved ambiguity explicitly

### Principal

First downstream consumer.

Responsibilities:

- preserve the artifact in trial state
- pass relevant constraints to retrieval, mutation, and expansion paths
- avoid re-inferring semantics already declared by planning

### Expansion System

Future consumer.

Responsibilities after later phases:

- use the artifact as applicability context
- reason from declared loss and stage constraints rather than names alone

Phase 1 only needs to ensure the artifact shape is suitable for that future use.

### Evaluation And Telemetry

Future consumer.

Responsibilities after later phases:

- interpret telemetry expectations from the planning artifact
- measure candidate outputs against admissibility expectations

Phase 1 only needs to ensure those expectations have a place to live.

## Canonical-First Requirement

The constraint artifact should be treated the same way the ingest system treats
canonical IR:

- it becomes the semantic source of truth for planning intent
- later convenience projections may exist, but they must be derived from the
  canonical artifact
- no later component should silently invent stronger semantics than the planner
  declared without surfacing that change

This requirement matters because later phases will be tempted to reintroduce
implicit behavior. Phase 1 should define the canonical planning artifact as the
authoritative source instead.

## Suggested Data Shape

This section is descriptive, not a final schema.

The artifact should likely have top-level sections similar to:

- metadata
- goal interpretation
- family and paradigm intent
- input contracts
- output contracts
- stage constraints
- flow constraints
- admissibility expectations
- telemetry expectations
- assumptions and open questions

Important qualities:

- machine-readable
- stable enough to version
- easy to diff
- narrow enough to keep prompts grounded
- expressive enough to survive into later phases

## Prompting Implications

Phase 1 will require prompt changes, but they should remain disciplined.

The Architect LLM should not be asked to produce a full CDG and the full
constraint system in one uncontrolled step. The likely sequence is:

- prompt for family/paradigm intent
- prompt for constraint system
- prompt for skeleton intent under those constraints
- prompt for decomposition under those constraints

This separation improves auditability and makes it easier to identify where a
run went wrong.

## Review And Audit Expectations

Because this artifact is intended to become foundational, it should be reviewed
with the same seriousness as other semantic source-of-truth artifacts.

Review questions should include:

- are the declared constraints specific enough to guide synthesis
- are they generic enough to avoid one-off family hacks
- do they distinguish data semantics rather than only naming conventions
- do they surface ambiguity rather than hiding it
- do they provide enough structure for downstream automation

## Deliverables

Phase 1 should produce:

1. A canonical constraint artifact definition and versioning strategy.
2. An updated Architect output contract that includes the artifact.
3. A persisted decomposition artifact path for the constraint artifact.
4. Planner-state plumbing so downstream stages can retain the artifact.
5. Regression coverage for planning outputs and serialization stability.
6. Maintainer documentation describing the new contract and its boundaries.

## Testing Strategy

Phase 1 should be tested as a contract phase, not as a benchmark phase.

The primary tests should validate:

- artifact presence
- artifact schema or structural validity
- deterministic serialization
- compatibility with existing decomposition outputs
- preservation across Architect and Principal boundaries

Good tests for this phase include:

- unit tests for artifact construction and normalization
- golden-style tests for representative goal families
- integration tests that confirm the artifact survives full decomposition

This phase should avoid overcommitting to performance or benchmark-quality
assertions. Those belong later once the rest of the stack consumes the contract.

## Risks

### Risk: Overdesigning the schema too early

If the schema tries to represent every future semantic need now, Phase 1 will
stall.

Mitigation:

- keep categories small and canonical
- version early
- prefer extension points over theoretical completeness

### Risk: Prompt bloat

If every planning call carries the full artifact blindly, prompting may become
noisy and brittle.

Mitigation:

- keep a compact canonical form
- derive prompt-focused summaries where needed
- preserve the full artifact separately from prompt text

### Risk: Artifact ignored by downstream code

If the artifact is emitted but not treated as authoritative, this phase will
add overhead without changing behavior.

Mitigation:

- define explicit consumers from the start
- make preservation of the artifact part of the contract

### Risk: Family-specific leakage

The first implementation may accidentally encode ECG-specific concepts as if
they were universal.

Mitigation:

- define generic categories first
- treat family-specific semantics as instances of general constraint kinds

## Exit Criteria

Phase 1 should be considered complete when all of the following are true:

- decomposition emits a first-class constraint artifact for every run
- the artifact has a stable canonical representation and version
- the Architect contract formally includes the artifact
- the Principal preserves the artifact through trial state
- maintainers can inspect planning intent without reverse-engineering prompt text
- later phases can depend on the artifact without changing its fundamental role

## Deferred To Later Phases

The following are intentionally deferred:

- typed skeleton asset migration into `../ageo-atoms`
- audited expansion asset migration into `../ageo-atoms`
- semantic edge redesign for all CDGs
- full admissibility enforcement
- telemetry standardization and profile-path unification

Those efforts should use the Phase 1 artifact as input rather than inventing
their own planning semantics.

## Recommended Sequencing Inside Phase 1

The implementation work for this phase should likely be staged in this order:

1. define the canonical constraint artifact and versioning
2. update Architect prompts and output parsing to produce it
3. persist the artifact alongside decomposition outputs
4. thread the artifact through Principal state
5. add regression and contract tests
6. document the contract and boundaries for maintainers

This sequencing minimizes the risk of downstream integration work starting
before the canonical artifact exists.

## Relationship To Later Phases

Phase 1 is the contract phase.

Phase 2 and Phase 3 will externalize reusable family knowledge into auditable
assets.

Phase 4 will make graph semantics rich enough for the constraint system to bind
cleanly to executable topology.

Phase 5 and Phase 6 will finally turn those declared constraints into fast
candidate rejection and useful refinement telemetry.

Without Phase 1, those later changes would still be operating on implicit
assumptions.
