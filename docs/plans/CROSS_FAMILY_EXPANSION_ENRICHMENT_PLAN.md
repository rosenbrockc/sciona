# Cross-Family Expansion And CDG Enrichment Plan

## Status

Drafted on April 7, 2026 after the ECG heart-rate e2e investigation and the
follow-on work on constraint-first planning, skeleton assets, and semantic
retrieval.

This document is intentionally high-level. It is meant to serve as a baseline
planning reference for future algorithm families, not as a family-specific
implementation spec.

## Purpose

The framework now has a clearer planning contract and better family-aware
matching, but expansion is still not doing enough useful work.

In practice, many synthesized candidates still remain close to their initial
family skeletons. The framework can often select a plausible core chain, but it
does not yet reliably enrich that chain into a stronger, more robust CDG using
family-specific structure.

This plan defines how expansion should evolve from:

- brittle graph rewrites
- local imperative family heuristics
- topology-only matching
- weak evidence for applicability
- narrow, family-specific thinking

to:

- constrained graph enrichment
- auditable, family-owned expansion assets
- semantic and boundary-aware applicability
- deterministic evidence-driven proposal selection
- a reusable model that works across many algorithm families

The motivating failure case was ECG heart-rate, but the design target is much
broader:

- signal processing
- probabilistic inference
- numerical optimization
- graph algorithms
- dynamic programming
- filtering and tracking
- geometry
- information theory
- and other future algorithm families in this repository

## Core Thesis

Expansion should be treated as constrained graph synthesis, not as a bag of
rewrites.

An expansion is valuable only when it can:

- enrich the current CDG with additional structure
- preserve the family’s planning constraints
- improve admissibility or robustness
- explain why the enrichment is appropriate
- remain auditable and reusable outside the local runtime engine

If expansion is only a collection of topology-sensitive local edits, it will
remain brittle and family knowledge will stay trapped in framework code.

## Problem Statement

The current expansion model is too weak for cross-family refinement.

### 1. Expansion matches graph shapes more than semantics

Many current rules effectively rely on exact topology or naming conventions.
That makes them fragile when the skeleton or executable graph does not expose
the precise shape the rule expects.

### 2. Expansion knowledge is too local

Family-specific refinement knowledge often lives in:

- runtime rule code
- diagnostics
- prompts
- one-off heuristics

instead of in assets that can be reviewed, audited, versioned, and reused.

### 3. Applicability is under-specified

A refinement should not apply just because a pattern can be matched. It should
apply because the current CDG, planning artifact, and telemetry together
justify it.

### 4. Expansion is too insertion-oriented

Useful enrichment is not limited to inserting one more node. In many families,
the right move is to:

- replace a stage
- split a stage
- branch and compare alternatives
- add a validation sidecar
- wrap a risky stage with a quality gate
- introduce a correction or reconciliation pass

### 5. Telemetry is not yet standardized enough to support refinement

Expansion needs compact evidence about what happened at intermediate stages.
Without that, it cannot reliably distinguish:

- a candidate that is structurally incomplete
- a candidate that is admissible but weak
- a candidate that is actively harmful

### 6. The model is still too family-local in how it is discussed

The ECG investigation made the gaps easy to see, but the underlying problem is
general. Other families will need the same expansion discipline:

- identify meaningful stage boundaries
- declare what information must be preserved
- detect when a stage underperforms
- enrich the CDG in a controlled, auditable way

## Design Intent

The expansion system should become the framework’s mechanism for introducing
family-specific sophistication without sacrificing determinism or auditability.

It should allow the framework to say:

- the current family skeleton is conceptually correct but too coarse
- here are the sanctioned enrichment operations for this family
- here is the evidence that one of them is warranted
- here are the enriched CDG candidates
- here is why one of them should be preferred

That is the role expansion should play across the repository.

## Conceptual Model

### Expansion operates on semantic stage contracts

Expansion should match and transform semantic structures, not merely local
graph shapes.

Examples of the kinds of semantic boundaries that should be first-class:

- raw input to conditioned representation
- representation to event sequence
- event sequence to summary or measure
- state estimate to correction stage
- search frontier to pruning or validation stage
- optimization iterate to convergence or constraint check

These are intentionally cross-family. Different families will instantiate them
with different data kinds and invariants, but the framework should reason about
them in a uniform way.

### Expansion should enrich the CDG, not just mutate it

An enriched CDG is a candidate graph that is more structured than the original
family skeleton while remaining faithful to the planning artifact.

Enrichment operations should include:

- insertion of a new stage
- replacement of one stage with a safer or richer stage
- expansion of one stage into a small subgraph
- attachment of a sidecar validation or telemetry stage
- introduction of a branch-and-select structure
- introduction of a quality-gating or correction structure

The key idea is that expansion produces a richer candidate graph, not merely a
patched local edge.

## Cross-Family Principles

The same high-level principles should govern expansion for all algorithm
families.

### 1. Family-owned knowledge

The inventory of sanctioned enrichment operations should belong to the family,
not only to the local framework runtime.

### 2. Constraint preservation

Every expansion must preserve or intentionally transform the planning
constraints, edge semantics, and admissibility expectations of the current
family instance.

### 3. Evidence-driven applicability

An expansion should require explicit evidence that it is appropriate.

### 4. Deterministic proposal selection

The framework should be able to compare multiple plausible enriched CDGs and
choose among them using deterministic checks before spending more compute.

### 5. Auditability

Every expansion should be inspectable and understandable without digging into
private framework control flow.

### 6. Compatibility with future families

The expansion model should not assume signal processing, ECG, or any single
modality. It should be framed around abstract stage roles, semantic ports, and
family contracts.

## Expansion Asset Model

Expansion knowledge should be represented as auditable assets, ideally with
long-term ownership in `../ageo-atoms`.

Each expansion asset should be able to express:

- family scope
- conceptual purpose
- applicability conditions
- before and after conceptual graph forms
- typed inputs and outputs
- semantic edge expectations
- preserved constraints
- expected benefit
- known risks
- uncertainty or maturity level
- references
- dejargonized documentation

An expansion asset is not the runtime engine. It is the family knowledge that
describes what enrichment is sanctioned and why.

## Runtime Responsibilities

The local framework still needs an expansion engine, but its role should become
clearer.

### Asset layer

Owns:

- what enrichment operations exist
- what semantic structures they target
- what evidence they require
- what they claim to improve

### Runtime layer

Owns:

- matching the current CDG and planning artifact to candidate expansion assets
- checking applicability against current telemetry and constraints
- materializing enriched candidate CDGs
- evaluating and ranking those enriched candidates
- recording why a proposal was accepted or rejected

This separation allows family knowledge to become reusable and reviewable.

## Applicability Model

Expansion must become more disciplined about when it is allowed to fire.

Applicability should be defined in terms of:

- planning artifact constraints
- semantic stage roles
- edge data-kind and provenance contracts
- admissibility expectations
- current runtime telemetry
- family-specific prohibitions

Examples of cross-family applicability questions:

- is a lossy transition happening too early
- is an expected validation or correction stage missing
- is a state transition unstable
- is a downstream stage receiving implausible support
- is a constraint check missing before an irreversible transformation
- is a branch needed because multiple structurally distinct enrichments are
  plausible

The answer should be explicit and inspectable.

## Deterministic Rejection And Proposal Selection

Expansion should not automatically apply the first rule that matches.

Instead, it should behave like proposal-based graph enrichment:

1. inspect the current CDG, planning artifact, and telemetry
2. identify all applicable sanctioned enrichments
3. materialize a small set of enriched candidate CDGs
4. run deterministic admissibility and compatibility checks
5. score or rank the surviving candidates
6. select one, or reject all if none are justified

This is important because many families will have multiple plausible
enrichments, and some of them will be:

- beneficial
- neutral
- harmful
- redundant

Proposal selection is where determinism should help the system move faster.

## Telemetry Requirements

Expansion will not become useful unless the runtime emits compact, standardized
evidence that refinements can consume.

The framework does not need to persist all intermediate tensors by default, but
it does need family-agnostic summaries that support enrichment decisions.

Examples of the kinds of summaries families may need:

- representation quality summaries
- event or detection count summaries
- interval or spacing summaries
- distribution plausibility summaries
- convergence summaries
- residual or error summaries
- constraint-violation summaries
- branch-selection or gate-rejection summaries

The exact metrics will vary by family, but the telemetry model should be
consistent:

- compact
- structured
- stage-local
- traceable back to a semantic edge or stage boundary

## Types Of Enrichment To Support

The framework should explicitly support multiple enrichment classes.

### Structural enrichments

- insert a new stage
- split a coarse stage into a subgraph
- replace a stage with a safer or richer equivalent

### Validation enrichments

- add a quality gate
- add a plausibility check
- add a constraint-validation step
- add a residual or support check

### Corrective enrichments

- add a correction or reconciliation stage
- add a cleanup stage after a lossy transform
- add an outlier or inconsistency filter

### Comparative enrichments

- add a branch-and-compare structure
- add a fallback branch
- add a multi-detector or multi-estimator comparison stage

### Adaptive enrichments

- introduce adaptation of thresholds, tolerances, or schedules
- introduce context-aware control logic
- introduce stabilization logic for iterative or stateful families

These classes are deliberately general so they can be instantiated across many
algorithm families.

## Relation To Skeletons

Skeletons and expansions should work together.

### Skeletons define the coarse family plan

They identify:

- the canonical stages
- the intended information-flow boundaries
- the initial constraints and invariants

### Expansions define sanctioned ways to enrich that plan

They identify:

- what extra structure may be introduced
- under what conditions
- with what tradeoffs

The system should not treat skeletons as the finished graph and expansions as
optional hacks. The intended relationship is:

- the skeleton is the minimal family scaffold
- the expansion inventory is the family’s sanctioned enrichment space

## Relation To `../ageo-atoms`

Long term, family skeletons and family expansion assets should be normalized
into `../ageo-atoms` and subject to the same general standards as atoms:

- references
- provenance
- rationale
- uncertainty notes
- dejargonized documentation
- maintainership
- versioning
- reviewability

The runtime engine can remain in `ageo-matcher`, but the family knowledge should
not stay trapped here permanently.

## Recommended Development Order

This is the recommended order for making expansion useful.

### 1. Finish semantic and boundary-aware matching

Expansion must be able to target semantic boundaries and stage roles, not only
current topology.

### 2. Define a canonical expansion asset format

The framework needs a family-neutral way to express before/after graph forms,
applicability, constraints, and audit metadata.

### 3. Build one strong family inventory end to end

Use one family as the proving ground for the model, but design the asset format
and runtime around cross-family reuse.

### 4. Standardize telemetry summaries that expansions can consume

Without compact evidence, proposal selection will remain guessy.

### 5. Introduce deterministic proposal-based selection

Expansion should create a small candidate set and choose among enriched CDGs.

### 6. Migrate family-owned assets into `../ageo-atoms`

Once the format and runtime are stable, move ownership of reusable family
knowledge out of local framework internals.

## Non-Goals

This document does not prescribe:

- exact schema fields for every asset
- the full runtime implementation of proposal ranking
- family-specific telemetry definitions for every family
- immediate migration of all existing expansions

Those should be handled in narrower phase or family-specific implementation
documents.

## Deliverables

This cross-family approach should eventually produce:

1. A canonical conceptual model for expansion as CDG enrichment.
2. An auditable expansion asset format.
3. A semantic applicability model tied to planning constraints.
4. A runtime engine that produces enriched candidate CDGs.
5. Deterministic proposal-selection logic.
6. A standardized telemetry contract for expansion evidence.
7. A migration path for skeleton and expansion ownership into `../ageo-atoms`.

## Exit Criteria

This approach can be considered successful when:

- expansion consistently produces richer CDGs than the minimal family skeleton
- enriched CDGs are selected because of explicit evidence, not opaque heuristics
- the same conceptual expansion machinery works across multiple algorithm
  families
- family enrichment inventories are auditable and reviewable
- the framework can explain why an enrichment was applied, rejected, or
  deferred
- family-specific refinement knowledge is no longer trapped exclusively inside
  local rewrite code

## Intended Use Of This Document

This document should be used as:

- a baseline planning reference for future family-specific expansion work
- a framing document for narrower implementation specs
- a source of conceptual consistency across phases and families

It should not be treated as a family-specific implementation plan. Instead, it
defines the repository-wide direction for making expansion genuinely useful as a
way to enrich CDGs.
