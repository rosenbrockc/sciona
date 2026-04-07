# Phase 2: Skeletons As Auditable Family Assets

## Status

Drafted on April 7, 2026 as the second implementation phase of
[Constraint-Driven Synthesis And Refinement Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md).

This document assumes Phase 1 has established a canonical planning-constraint
artifact and that decomposition can already emit that artifact as part of the
Architect contract.

## Purpose

The framework currently relies on skeletons as internal scaffolds with weaker
semantics and weaker auditability than atoms in `../ageo-atoms`.

That is a structural mismatch.

If skeletons are reusable family knowledge, they should be:

- explicit
- typed
- versioned
- documented
- auditable
- reviewable

This phase promotes skeletons from framework-local templates to first-class
family assets that carry the same kind of semantic and documentation discipline
expected elsewhere in the ecosystem.

## Why This Phase Exists

The parent investigation showed that decomposition can produce a superficially
reasonable scaffold while still omitting important structure and constraints.
That happens because the skeleton layer is too weak:

- it does not declare enough semantic intent
- it does not express information-loss assumptions
- it does not encode structural obligations strongly enough
- it is not reviewed with the same rigor as reusable atoms

Until skeletons are first-class assets, the Architect will keep inventing or
reconstructing family structure from prompts and heuristics.

## Goal

Define and adopt an auditable skeleton-family asset model so that reusable
algorithm-family scaffolds can be stored, reviewed, versioned, and consumed as
canonical planning knowledge.

The phase should move the system from:

- framework-owned, weakly typed skeleton templates

to:

- family-owned, typed, documented skeleton assets with explicit constraints and
  semantic edge meaning

## Non-Goals

Phase 2 should not:

- migrate expansion operations yet
- redesign the rewrite engine
- solve all admissibility logic
- complete the entire semantic CDG redesign
- fully restructure `../ageo-atoms` for every family at once

Those concerns are handled in later phases.

## Problem Statement

Skeletons currently behave like an implementation convenience instead of a
semantic source of truth.

That causes several issues:

- family structure is under-specified
- downstream code cannot tell which edges are semantically important
- there is no shared review surface for family patterns
- audit expectations differ between atoms and skeletons
- framework prompts become the de facto source of family semantics

The system needs a representation where family-level algorithm structure can be
inspected independently of any single run.

## What A Skeleton Asset Should Represent

A skeleton asset is not an implementation of an algorithm. It is a reusable,
typed conceptual scaffold for a family of algorithms.

At a minimum, a skeleton asset should describe:

- family name and scope
- conceptual purpose
- inputs and outputs
- stage structure
- typed stage-to-stage flow
- loss and preservation assumptions
- planning constraints inherited or required by the family
- unresolved alternatives or optional branches
- references and provenance
- uncertainty or confidence notes
- dejargonized explanation of what the skeleton means

The artifact should be rich enough that a maintainer can review it without
needing to inspect planner prompts.

## Proposed Ownership Model

The long-term target is for family skeleton assets to live in `../ageo-atoms`.

The reasons are straightforward:

- they are reusable semantic assets
- they should be audited alongside related atoms
- they should share the same review culture for references and documentation
- they should not remain hidden inside one framework runtime

The framework in `ageo-matcher` can still consume these assets, but it should
not be their only source of truth.

## Artifact Requirements

Phase 2 should define a skeleton asset format with the following properties.

### Typed Structure

Each stage should carry:

- conceptual role
- input and output types at the semantic level, not only Python level
- preconditions and guarantees
- family-specific notes

### Edge Semantics

Edges should express:

- data kind
- provenance expectations
- loss class
- alignment or monotonicity expectations where relevant

### Constraint Binding

The skeleton asset should be able to reference or incorporate the planning
constraints established in Phase 1.

The intended relationship is:

- Phase 1 constraint artifact describes run-level planning intent
- Phase 2 skeleton asset describes family-level structural knowledge

The two should align, not compete.

### Audit Metadata

Each asset should support:

- provenance
- references
- rationale
- uncertainty and caveats
- maintainership or ownership metadata

### Human-Readable Documentation

Every skeleton should include or generate:

- a concise conceptual summary
- dejargonized stage descriptions
- explanation of the family pattern
- explanation of what the skeleton does not yet specify

## Framework Changes Required

Phase 2 will require changes in how the Architect selects and uses skeletons.

### Skeleton Resolution

The Architect should resolve a skeleton asset by family or planning intent
rather than instantiating only internal hard-coded templates.

This likely means adding:

- a skeleton asset registry
- resolution by family/paradigm/constraint match
- version or compatibility selection rules

### Skeleton Instantiation

The runtime should instantiate the chosen skeleton asset into a working CDG
representation while preserving the source asset identity.

Important consequence:

- a decomposition artifact should record which skeleton asset and version it was
  derived from

### Local Compatibility Layer

During rollout there will likely be legacy local skeleton templates and new
asset-backed skeletons at the same time.

Phase 2 should define a compatibility boundary so:

- new canonical semantics live in the asset model
- old local templates are transitional
- the runtime can still produce a CDG while migration is underway

## Migration Strategy

Do not attempt to migrate every family at once.

The recommended strategy is:

1. define the asset format and loader
2. choose one or two representative families
3. author audited skeleton assets for them
4. consume them from the Architect path
5. keep local fallback templates only where migration is incomplete

The first migrated families should be chosen for leverage, not completeness.
Ideal candidates:

- signal -> detect -> measure / rate families
- families already causing repeated refinement pain
- families with strong reusable structure across goals

## Suggested Deliverables

Phase 2 should produce:

1. A skeleton asset schema or canonical file contract.
2. A framework-side skeleton loader and resolution path.
3. Source attribution for skeleton selection in decomposition artifacts.
4. One or more migrated audited skeleton-family assets in `../ageo-atoms`.
5. Maintainer documentation for authoring and reviewing skeleton assets.
6. Regression coverage around skeleton selection and serialization.

## Testing Strategy

Phase 2 should be validated at three levels.

### Contract Tests

Verify:

- skeleton assets parse and validate
- required fields and documentation are present
- asset versioning and compatibility rules work

### Runtime Integration Tests

Verify:

- the Architect can resolve and instantiate a skeleton asset
- decomposition outputs preserve the source asset identity
- the instantiated skeleton still produces a valid CDG handoff

### Reviewability Tests

Verify:

- skeleton assets expose references and documentation in a stable way
- missing audit fields are surfaced as validation failures or warnings

## Risks

### Risk: Skeleton assets become too implementation-specific

If the skeleton artifact encodes atom-level details, it will stop being a
family scaffold and become another execution plan.

Mitigation:

- keep the asset conceptual
- reserve concrete primitive binding for later stages

### Risk: Asset format duplicates the entire CDG model too early

If the asset is forced to mirror every current CDG detail, migration will be
slow and brittle.

Mitigation:

- define only the semantic fields needed for family planning
- add adapter logic from asset form to runtime form

### Risk: Migration stalls because every family must be perfect before adoption

Mitigation:

- start with a narrow migrated subset
- preserve a compatibility path during transition

### Risk: Audit burden becomes too heavy for early adoption

Mitigation:

- define a minimum viable audit contract
- distinguish required metadata from optional enrichment

## Exit Criteria

Phase 2 should be considered complete when:

- the framework can resolve skeletons from canonical family assets
- at least one representative family uses asset-backed skeletons in practice
- the asset format includes typed structure, edge semantics, and audit metadata
- decomposition outputs retain traceability to the chosen skeleton asset
- maintainers can review skeleton assets independently of runtime code

## Deferred To Later Phases

Phase 2 does not finish:

- expansion asset migration
- semantic rewrite redesign
- admissibility enforcement
- telemetry reform

Those phases should build on asset-backed family skeletons rather than
reintroducing hidden framework-local family logic.

## Recommended Sequencing

1. define the skeleton asset contract
2. add framework loading and resolution support
3. migrate one high-value family into `../ageo-atoms`
4. make the Architect consume that asset path
5. add regression and review tooling
6. document authoring and maintenance expectations

## Relationship To Other Phases

Phase 1 defines planning constraints.

Phase 2 turns family structure into auditable assets.

Phase 3 will do the same for expansions.

Phase 4 will make the runtime graph rich enough that the semantics encoded in
these assets are executable in a principled way.
