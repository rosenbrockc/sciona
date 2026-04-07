# Phase 4: Semantic CDG And Boundary-Aware Rewriting

## Status

Drafted on April 7, 2026 as the fourth implementation phase of
[Constraint-Driven Synthesis And Refinement Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md).

This phase assumes the system already has:

- a canonical planning-constraint artifact from Phase 1
- auditable skeleton assets from Phase 2
- auditable expansion assets from Phase 3, or at least the corresponding asset
  model

## Purpose

The current runtime graph model is not semantically rich enough to support
reliable refinement.

The ECG jump-removal failure was the clearest example:

- the refinement was conceptually valid
- the diagnostic evidence existed
- but the rule could not apply because the graph did not represent the relevant
  boundary shape

Phase 4 fixes the representation mismatch by making the executable graph model
rich enough to express semantic boundaries, not just internal node-to-node
topology.

## Goal

Redesign the runtime CDG model and rewrite surface so refinements can target
semantic boundaries such as:

- root inputs
- root outputs
- waveform-to-event transitions
- event-to-measure transitions
- stateful or provenance-preserving edges

The phase should move the system from:

- topology-first matching over under-specified graphs

to:

- semantic matching over boundary-aware, typed graphs

## Non-Goals

Phase 4 should not:

- migrate all assets into `../ageo-atoms`
- implement every admissibility gate
- solve all telemetry collection concerns
- fully optimize runtime performance before semantics are correct

Semantics and correctness take precedence over micro-optimization in this phase.

## Problem Statement

The current CDG model is too weak in three ways:

1. Root inputs and outputs are not represented strongly enough as graph
   boundaries.
2. Edges do not carry enough semantic meaning.
3. Rewrites depend too heavily on exact internal graph shape.

As a result:

- semantically correct refinements can fail to match
- losses or provenance assumptions are invisible in the graph
- later stages must infer meaning from names and conventions

That is incompatible with the broader design goal of deterministic, auditable
planning and refinement.

## Representation Objectives

Phase 4 should redesign or extend the CDG model so it can express:

- boundary nodes or boundary ports for root inputs and outputs
- semantic edge typing
- provenance and time-basis associations
- information-loss class
- optionality and state flow where relevant

The representation should remain executable enough for the runtime, but much
closer to the real semantics of algorithm composition.

## Boundary Model

The key missing concept is the boundary.

The graph should support a first-class representation of:

- external inputs
- external outputs
- stage-local inputs and outputs
- transitions across conceptual data-kind boundaries

Examples of boundary-aware rewrite targets:

- insert preprocessing before the first consumer of root input `signal`
- insert correction after an event detector and before a rate estimator
- insert validation between a lossy transformation and a downstream consumer

Without boundary nodes or boundary ports, these operations remain fragile.

## Edge Semantics

Phase 4 should define richer semantic edge metadata.

At minimum, an edge should be able to express:

- data kind
- provenance
- time basis or sampling basis
- loss class
- alignment expectations
- whether downstream consumers require preservation of those properties

The edge should become the place where information-flow assumptions are made
visible rather than hidden in prompt text or stage names.

## Rewrite Semantics

The rewrite engine should be able to match on:

- stage roles
- input and output semantic types
- boundary relations
- flow constraints inherited from planning
- family structure derived from skeleton and expansion assets

Pure node-name or primitive-name matching should become a compatibility layer,
not the primary refinement mechanism.

## Compatibility Strategy

This phase will likely require a compatibility boundary because the current
runtime expects the existing CDG form.

Recommended approach:

- define the richer semantic form first
- add adapters between the current runtime CDG and the richer form
- gradually shift internal consumers onto the semantic form
- keep legacy projections derived rather than authoritative

The goal is to avoid a big-bang rewrite while still making the semantic form the
new source of truth.

## Interaction With Earlier Phases

Phase 4 is where the outputs of earlier phases become operational:

- Phase 1 constraints inform edge and boundary semantics
- Phase 2 skeleton assets describe family structure against the richer graph
- Phase 3 expansion assets target semantic boundaries rather than fragile local
  shapes

Without Phase 4, the earlier phases remain conceptually stronger but still hard
to execute reliably.

## Framework Changes Required

Phase 4 likely needs changes in:

- Architect handoff structures
- internal graph models
- rewrite matching logic
- graph serialization
- validation and critique paths
- runtime visualization or debugging tools

These changes should be introduced through a canonical representation rather
than ad hoc field growth on the current model.

## Suggested Deliverables

Phase 4 should produce:

1. A canonical semantic CDG model or extension layer.
2. First-class boundary representation for inputs and outputs.
3. Semantic edge metadata sufficient for refinement and admissibility.
4. A rewrite API that can target semantic boundaries.
5. Compatibility adapters from current runtime graph forms.
6. Maintainer documentation describing the new canonical graph surface.

## Testing Strategy

### Model Tests

Verify:

- boundary nodes or ports serialize and deserialize correctly
- semantic edge fields are preserved
- compatibility adapters are deterministic

### Rewrite Tests

Verify:

- boundary-aware refinements can match where topology-only rules previously
  failed
- semantic match explanations are inspectable
- non-applicable refinements fail for explicit reasons

### Integration Tests

Verify:

- Architect outputs can still flow into matching, synthesis, and evaluation
- migrated families continue to build executable artifacts

## Risks

### Risk: Overcomplicating the graph model

Mitigation:

- add only semantics that support planning, refinement, and admissibility
- avoid turning the graph into an all-purpose ontology

### Risk: Compatibility burden becomes permanent

Mitigation:

- make the semantic form canonical
- keep legacy projections derived and transitional

### Risk: Rewrite matching becomes slow or opaque

Mitigation:

- keep matching predicates typed and inspectable
- favor explainable rule-application diagnostics

### Risk: Boundary representation diverges from executable runtime needs

Mitigation:

- maintain an explicit contract between semantic graph form and executable graph
  lowering

## Exit Criteria

Phase 4 should be considered complete when:

- the framework has a canonical semantic graph form
- root inputs and outputs are represented as first-class boundaries
- semantic edge metadata is sufficient to express loss and provenance
- at least one previously brittle refinement can apply through boundary-aware
  matching
- downstream runtime components can still execute candidates via compatibility
  lowering

## Deferred To Later Phases

Phase 4 does not finish:

- full candidate admissibility policy
- full telemetry standardization
- benchmark reform

Those phases should rely on the semantic graph representation produced here.

## Recommended Sequencing

1. define the canonical semantic graph contract
2. add boundary representation
3. add semantic edge metadata
4. introduce compatibility adapters
5. update rewrite matching to use semantic boundaries
6. validate one or two high-value family refinements end to end

## Relationship To Other Phases

This is the phase that turns the earlier planning and asset work into something
the runtime can apply correctly. Later phases on admissibility, telemetry, and
benchmarking should treat this graph model as the canonical execution-time
semantic surface.
