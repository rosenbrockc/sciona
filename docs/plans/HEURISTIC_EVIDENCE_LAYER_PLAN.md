# Heuristic Evidence Layer Plan

## Status

Drafted on April 8, 2026 after the ECG heart-rate refinement investigation,
the cross-family expansion work, and the realization that expansion quality is
now limited less by graph mechanics than by the weakness of the evidence used
to decide when and how to enrich a candidate CDG.

This document is intentionally high-level. It is the ground-truth planning
reference for the new heuristic-evidence abstraction layer. Detailed
implementation plans for each phase are documented separately and should be
audited against this document.

## Purpose

The framework now has better constraint-first planning, stronger semantic CDG
machinery, audited skeleton and expansion assets, and healthier runtime
telemetry. Those improvements were necessary, but they did not fully solve the
actual decision problem.

The remaining gap is that expansion and proposal selection still rely too much
on:

- thin runtime summaries
- ad hoc family-local diagnostics
- implicit expert assumptions hidden in rule code
- direct jumps from raw telemetry to structural decisions

That is not how strong algorithms are built in practice.

In practice, useful refinement usually comes from inspecting the behavior of the
current dataflow, identifying characteristic failure modes or quality patterns,
and then using those observations to choose the next structural move. This plan
defines a first-class, audited, cross-family heuristic-evidence layer to make
that process explicit inside Sciona.

## Why This Layer Is Needed

The ECG heart-rate investigation made the problem concrete.

The framework was eventually able to:

- select a plausible core atom chain
- generate enriched ECG candidates
- evaluate those candidates deterministically

But the best structural enrichment selected by the framework improved loss only
slightly, while direct data-first analysis showed that larger gains were
available from simple heuristic responses such as rejecting unstable event
intervals and smoothing the derived rate series.

That does not mean the solution is to encode more ECG jargon. It means the
framework is missing a reusable abstraction between:

- raw runtime telemetry
- family-aware structural actions

That missing abstraction is a heuristic evidence layer.

## Core Thesis

Heuristics should be first-class, audited, dejargonized evidence artifacts.

They should not be:

- hidden inside expansion rule code
- narrowly named in one domain’s terminology
- treated as an informal side effect of telemetry
- coupled directly to one benchmark or one modality

Instead, the framework should reason with heuristics such as:

- `interval_instability`
- `dominant_nuisance_structure`
- `boundary_discontinuity`
- `density_collapse`
- `plausibility_fragmentation`
- `confidence_instability`
- `residual_structure_after_transform`

These are cross-disciplinary observations about dataflow behavior. A specific
family may explain them in local terms, but the shared interface should remain
de-jargonized and portable.

## Design Intent

The heuristic-evidence layer should let the framework say:

- here are the relevant runtime observations for this candidate CDG
- here are the audited heuristics those observations support
- here are the family-sanctioned actions associated with those heuristics
- here is the prior evidence that those actions have or have not helped under
  similar heuristic signatures
- here is the deterministic proposal policy that should follow

That is the intended bridge between runtime behavior and structural refinement.

## Conceptual Model

The heuristic-evidence layer sits between runtime telemetry and proposal
selection.

### 1. Runtime telemetry produces evidence

Runtime execution emits compact, family-neutral summaries of intermediate node
outputs, edge behavior, and final outputs.

### 2. Heuristic producers interpret that evidence

Audited atoms or framework-level evidence transforms convert raw summaries into
typed heuristic outputs.

### 3. Family registries interpret heuristic meaning locally

Each family or skeleton declares which heuristics are relevant, how they should
be interpreted, and what families of structural actions they support.

### 4. Proposal selection consumes heuristics

Proposal selection uses the declared heuristic outputs, family registry, and
outcome memory to rank candidate enrichments deterministically.

### 5. Outcome memory closes the loop

The framework records which heuristic signatures were present, which actions
were tried, and what improvement or regression resulted.

## Cross-Family Principles

The heuristic layer must preserve the repository’s cross-family and
cross-disciplinary purpose.

### 1. De-jargonized shared vocabulary

Heuristics should be named for observable behavior, not one field’s local
terminology.

For example:

- prefer `interval_instability` over `RR irregularity`
- prefer `dominant_nuisance_structure` over `baseline wander`
- prefer `quality_gate_needed` over `signal quality index failed`

Family-specific documentation can explain how these abstractions appear in ECG,
EEG, optimization traces, graph search frontiers, or probabilistic samplers.

### 2. Heuristics are audited assets

Every heuristic needs:

- a definition
- a dejargonized explanation
- rationale
- uncertainty and failure modes
- applicability scope
- evidence requirements
- references when applicable

### 3. Atom outputs can be heuristic producers

Audited atoms should be allowed to declare that one or more of their outputs are
usable heuristics. This keeps heuristic production inside the same rigorous
asset ecosystem as executable atoms.

### 4. Family specialization happens through registries

The shared heuristic schema must remain generic. Family-specific interpretation
belongs in family or skeleton registries, not in the global interface.

### 5. Heuristics inform structure, not just ranking

Heuristics should support:

- expansion applicability
- substitution and replacement decisions
- admissibility gates
- confidence escalation
- benchmark analysis
- future planning hints

### 6. Determinism comes from evidence plus memory

Determinism should not mean repeating a weak default. It should mean applying a
stable policy over:

- current heuristic outputs
- family registry rules
- historical outcome evidence

## Architecture

The new layer should include the following architectural pieces.

### Canonical heuristic schema

A generic schema for first-class heuristics, including:

- `heuristic_id`
- `display_name`
- `dejargonized_meaning`
- `evidence_type`
- `value_shape`
- `confidence`
- `uncertainty_notes`
- `applicability_scope`
- `producer_kind`
- `supported_action_classes`

### Heuristic-producing atom metadata

Audited atoms should be able to declare:

- which outputs are heuristics
- the semantic kind of each heuristic
- the expected evidence contract
- whether the output is diagnostic, advisory, or gate-worthy

### Family heuristic registries

Families and skeletons should declare:

- which heuristics matter
- which heuristic producers are sanctioned
- which action classes each heuristic can support
- family-specific interpretation notes
- admissibility implications
- escalation rules

### Runtime heuristic extraction and persistence

The runtime must persist:

- raw summaries
- derived heuristic outputs
- provenance for how the heuristics were produced
- confidence and uncertainty metadata

### Heuristic-driven proposal selection

Proposal generation and ranking should use heuristic outputs directly instead of
forcing each family rule to rediscover them from raw runtime summaries.

### Heuristic-to-outcome memory

The framework should persist:

- family
- skeleton
- heuristic signature
- selected action
- action class
- loss delta
- confidence delta
- benchmark context
- success and failure notes

This becomes a deterministic prior for future proposal ranking.

## Relationship To Existing Plans

This plan does not replace the earlier cross-family expansion or signal-family
plans. It extends them by introducing the missing decision layer beneath
expansion applicability and proposal selection.

It is especially aligned with:

- [CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md)
- [CROSS_FAMILY_EXPANSION_ENRICHMENT_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/CROSS_FAMILY_EXPANSION_ENRICHMENT_PLAN.md)
- [SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/ageo-matcher/docs/plans/SIGNAL_PROCESSING_EXPANSION_IMPLEMENTATION_PLAN.md)

The heuristic-evidence layer should become the main bridge between:

- Phase 6 runtime telemetry
- family-owned expansion assets
- deterministic admissibility and proposal policies

## Phase Set

The work is split into six phases:

1. Canonical Heuristic Model
2. Heuristic-Producing Atom Metadata
3. Family Heuristic Registries
4. Runtime Heuristic Extraction And Persistence
5. Heuristic-Driven Proposal Selection
6. Heuristic Outcome Memory And Benchmark Integration

Each phase is documented separately:

- [HEURISTIC_PHASE_1_CANONICAL_HEURISTIC_MODEL.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_PHASE_1_CANONICAL_HEURISTIC_MODEL.md)
- [HEURISTIC_PHASE_2_HEURISTIC_ATOM_METADATA.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_PHASE_2_HEURISTIC_ATOM_METADATA.md)
- [HEURISTIC_PHASE_3_FAMILY_HEURISTIC_REGISTRIES.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_PHASE_3_FAMILY_HEURISTIC_REGISTRIES.md)
- [HEURISTIC_PHASE_4_RUNTIME_HEURISTIC_EXTRACTION_AND_PERSISTENCE.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_PHASE_4_RUNTIME_HEURISTIC_EXTRACTION_AND_PERSISTENCE.md)
- [HEURISTIC_PHASE_5_HEURISTIC_DRIVEN_PROPOSAL_SELECTION.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_PHASE_5_HEURISTIC_DRIVEN_PROPOSAL_SELECTION.md)
- [HEURISTIC_PHASE_6_HEURISTIC_OUTCOME_MEMORY_AND_BENCHMARK_INTEGRATION.md](/Users/conrad/personal/ageo-matcher/docs/plans/HEURISTIC_PHASE_6_HEURISTIC_OUTCOME_MEMORY_AND_BENCHMARK_INTEGRATION.md)

## Dependency Structure

### Hard dependencies

- Phase 1 is the vocabulary and schema foundation for all later phases.
- Phase 2 depends on Phase 1 because atom metadata must target the canonical
  heuristic model.
- Phase 3 depends on Phase 1 and benefits strongly from early Phase 2 progress.
- Phase 4 depends on Phase 1 and Phase 2; it benefits from Phase 3 because
  runtime extraction should line up with family needs.
- Phase 5 depends on Phases 1 through 4.
- Phase 6 depends on Phases 3 through 5 because outcome memory is only useful
  once heuristics, actions, and benchmark behaviors are linked coherently.

### Soft dependencies

- Phase 3 can start in parallel with Phase 2 once Phase 1 is stable enough.
- Phase 4 can begin before all family registries are complete if a reference
  family is chosen.
- Phase 6 can start with benchmark-contract work before ranking memory is fully
  integrated.

## Parallelization Analysis

The work should proceed in dependency waves, not as a fully serialized queue.

### Wave 0: Shared vocabulary

Contains:

- Phase 1

Reason:

- The rest of the system needs a stable, audited, de-jargonized schema before
  adding producers, registries, or memory.

Parallelism:

- Internal Phase 1 subtasks can run in parallel:
  - schema definition
  - naming and de-jargonization review
  - auditability requirements
  - compatibility mapping from current diagnostics

### Wave 1: Production and interpretation

Contains:

- Phase 2
- Phase 3

Reason:

- Once the schema is stable, the system can independently define:
  - how heuristics are produced
  - how families use them

Integration point:

- Atom metadata and family registries must agree on heuristic identifiers,
  evidence shapes, and action classes.

### Wave 2: Runtime activation

Contains:

- Phase 4

Reason:

- Runtime persistence becomes meaningful once the schema and initial producer
  contracts exist.

Parallelism:

- Internal Phase 4 subtasks can run in parallel:
  - runtime artifact schema updates
  - heuristic extraction hooks
  - provenance and uncertainty persistence
  - evaluation trace integration

### Wave 3: Decision logic

Contains:

- Phase 5

Reason:

- Proposal selection should only depend on heuristics once the heuristics are
  reliably produced and interpreted.

Parallelism:

- Internal Phase 5 subtasks can run in parallel:
  - action-class mapping
  - admissibility integration
  - ranking-policy integration
  - reporting and explanation surfaces

### Wave 4: Memory and validation

Contains:

- Phase 6

Reason:

- Outcome memory is most useful after the end-to-end heuristic-driven proposal
  flow is real.

Parallelism:

- Internal Phase 6 subtasks can run in parallel:
  - storage schema
  - benchmark-policy integration
  - historical replay analysis
  - migration-readiness reporting

## Recommended Execution Strategy

The recommended ordering is:

1. Freeze the de-jargonized heuristic schema and naming conventions.
2. Start atom metadata work and family registry work in parallel.
3. Bring runtime extraction online for a reference family while preserving the
   cross-family contract.
4. Move proposal selection to the heuristic layer.
5. Add outcome memory and benchmark reporting.
6. Use signal processing as the first proving ground without allowing signal
   terminology to dominate the shared interface.

## Expected Benefits

If this plan succeeds, the framework should become better at:

- selecting structurally meaningful enrichments
- explaining why a refinement was chosen
- reusing successful decision patterns across runs
- preserving cross-family abstraction while still benefiting from
  family-specific experience
- migrating more refinement knowledge into audited, reviewable assets

## Risks

The main risks are:

- reintroducing hidden family jargon under new names
- making the heuristic schema so abstract that it becomes unusable
- allowing heuristic producers to become an unreviewable ad hoc layer
- letting family registries become incompatible with one another
- prematurely overfitting outcome memory to a small number of benchmarks

These risks should be managed by keeping:

- the shared schema small and semantically clear
- family-specific interpretation local to registries
- heuristic producers auditable and asset-backed
- benchmark reporting explicit about uncertainty and sample size

## Exit Criteria

This abstraction layer should be considered established when:

- the framework has a stable canonical heuristic schema
- audited atoms can declare heuristic-producing outputs
- at least one reference family uses a heuristic registry end to end
- runtime artifacts persist heuristic outputs with provenance
- proposal selection can cite heuristic evidence directly
- benchmark reporting includes heuristic signatures and resulting actions
- the design remains understandable without relying on family-specific jargon
