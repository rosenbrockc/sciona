# Phase 3: Expansion Operations As Auditable Family Assets

## Status

Drafted on April 7, 2026 as the third implementation phase of
[Constraint-Driven Synthesis And Refinement Plan](/Users/conrad/personal/ageo-matcher/docs/plans/CONSTRAINT_DRIVEN_SYNTHESIS_PLAN.md).

This document assumes:

- Phase 1 has introduced canonical planning constraints
- Phase 2 has introduced auditable family skeleton assets or at least the
  associated asset model

## Purpose

The framework currently expresses expansion and refinement mostly as local code:

- diagnostic functions
- graph rewrite rules
- topology assumptions
- family-specific heuristics

That makes expansions:

- hard to audit
- hard to review outside the framework
- too tightly coupled to local graph mechanics
- inconsistent with the rigor expected of reusable atoms

Phase 3 turns reusable family-specific expansions into auditable assets, ideally
owned in `../ageo-atoms`, so refinement knowledge is explicit and shared.

## Goal

Represent family refinement operations as versioned, auditable graph-level
assets that the local rewrite engine can consume.

The phase should move the system from:

- framework-local expansion code as the only source of refinement truth

to:

- family-owned refinement assets plus a runtime engine that interprets them

## Non-Goals

Phase 3 should not:

- complete the full semantic CDG redesign
- rewrite every runtime matching primitive
- solve all admissibility logic
- make every expansion asset fully automatic on day one

Its purpose is to externalize and structure the refinement inventory, not to
finish every downstream consumer.

## Why This Phase Matters

The ECG investigation showed two problems at once:

1. important refinement operations were missing from the family inventory
2. the existing rules were hard to reason about because they lived as local
   imperative code

If refinement knowledge is reusable family knowledge, it should be treated like
other reusable assets:

- typed
- reviewable
- referenced
- uncertainty-aware
- documented

Otherwise the framework keeps encoding family wisdom in private code paths and
prompts.

## What Counts As An Expansion Asset

An expansion asset is a reusable refinement operation that transforms one
family-level conceptual structure into a richer one.

Examples of the kinds of operations this phase is meant to support:

- inserting a preprocessing stage
- inserting a correction stage
- inserting a quality gate
- inserting a validation or rejection stage
- replacing a coarse stage with a safer subgraph
- adding a smoothing or outlier-rejection branch

The asset is not the runtime engine. It is the declarative family knowledge that
describes what transformation is valid and why.

## Asset Content Requirements

Each expansion asset should be able to express:

- family scope
- intent and rationale
- applicability constraints
- before and after conceptual graph forms
- port and edge typing
- expected effect on information flow
- references and provenance
- uncertainty or caution notes
- human-readable documentation

The asset should be understandable without reading the runtime matching code.

## Applicability Model

Phase 3 should make applicability explicit rather than implicit.

An expansion asset should be able to declare:

- required data kinds
- required boundary or stage types
- required planning constraints
- required telemetry or diagnostic evidence
- prohibited contexts

This matters because many expansions are valid semantically but not universally.
The framework should be able to answer:

- when is this refinement appropriate
- when is it high risk
- what evidence justifies it

## Relationship To `../ageo-atoms`

These assets belong conceptually with family knowledge, not only with runtime
execution logic.

The long-term target is:

- audited family refinement assets live in `../ageo-atoms`
- the local framework consumes them
- maintainers can review refinement inventories with the same rigor as related
  atoms and skeletons

The local framework still owns:

- matching machinery
- orchestration
- runtime application logic

But it should not remain the only place where refinement semantics are encoded.

## Runtime Responsibilities

Phase 3 does not remove the need for a local runtime engine. It clarifies the
boundary.

### Asset Layer

Owns:

- what refinement exists
- when it is conceptually applicable
- what structure it introduces
- why it exists

### Runtime Engine

Owns:

- matching the current graph against asset applicability
- applying the transformation in executable form
- reporting success, failure, or incompatibility

This separation is important because it allows family refinement knowledge to be
reviewed independently of the mechanics of applying it.

## Migration Strategy

The recommended strategy is incremental.

1. Define the expansion asset format.
2. Convert one existing built-in family expansion inventory into the new format.
3. Add a loader and compatibility layer so local runtime rules can reference the
   asset-backed definitions.
4. Migrate the highest-value missing refinements into asset form.
5. De-emphasize hidden local-only family rewrite logic over time.

The migration should start with families where refinement already exists or is
obviously needed.

## Important Design Choices

### Keep assets declarative

Expansion assets should declare structure and applicability, not hide runtime
procedures or arbitrary code.

### Preserve auditability

Each refinement asset should explain:

- what problem it addresses
- what evidence justifies it
- what tradeoff it introduces

### Support uncertainty

Not every refinement should be treated as universally good. Assets should be
able to say:

- strong recommendation
- conditional recommendation
- risky fallback
- experimental

### Avoid family-specific hacks in the framework

If a refinement is reusable family knowledge, add it as an asset. Do not encode
it only as a one-off runtime branch.

## Deliverables

Phase 3 should produce:

1. A canonical expansion asset contract.
2. A framework-side loader and registry for expansion assets.
3. A compatibility path from asset-backed definitions to runtime rewrite
   application.
4. One or more migrated family refinement inventories in `../ageo-atoms`.
5. Documentation for authoring, reviewing, and versioning expansion assets.
6. Regression coverage for asset loading and applicability evaluation.

## Testing Strategy

### Contract Tests

Verify:

- expansion assets parse and validate
- applicability requirements are structurally well-formed
- before and after forms remain consistent

### Runtime Integration Tests

Verify:

- the local rewrite engine can load and evaluate an expansion asset
- compatibility translation into runtime form works
- explainable failure modes are surfaced when an asset cannot apply

### Audit Tests

Verify:

- required references and rationale are present
- documentation generation or retrieval is stable

## Risks

### Risk: Expansion assets become too tied to current runtime graph mechanics

Mitigation:

- define assets at the semantic graph level first
- use adapters to current runtime forms where necessary

### Risk: Applicability rules become another hidden programming language

Mitigation:

- keep applicability categories narrow and typed
- prefer structured predicates over arbitrary embedded code

### Risk: Asset migration duplicates skeleton semantics

Mitigation:

- keep skeletons focused on family structure
- keep expansions focused on family refinement operations

### Risk: Runtime and asset semantics drift apart

Mitigation:

- require compatibility tests between asset forms and runtime application
- keep local-only rewrite logic temporary and visible

## Exit Criteria

Phase 3 should be considered complete when:

- the framework can consume expansion assets as a canonical refinement source
- at least one representative family has asset-backed refinements
- expansion assets include applicability constraints, rationale, and audit
  metadata
- maintainers can review refinement inventories independently of runtime code
- local refinement logic can trace back to explicit asset definitions

## Deferred To Later Phases

Phase 3 does not finish:

- semantic boundary-aware matching for all runtime graphs
- full admissibility enforcement
- telemetry redesign

Those later phases should operate over asset-backed refinement inventories
rather than hard-coded family heuristics.

## Recommended Sequencing

1. define the expansion asset contract
2. add loader and registry support in the framework
3. migrate one existing family refinement inventory
4. add compatibility translation to runtime rewrite mechanics
5. add missing high-value refinements as assets
6. document authoring and review expectations

## Relationship To Other Phases

Phase 2 externalizes family skeletons.

Phase 3 externalizes family refinements.

Phase 4 will make the runtime graph model and rewrite engine rich enough to
apply those refinements reliably at semantic boundaries.
