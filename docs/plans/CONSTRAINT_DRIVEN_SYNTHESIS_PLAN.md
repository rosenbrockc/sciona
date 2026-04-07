# Constraint-Driven Synthesis And Refinement Plan

## Status

Drafted on April 7, 2026 after the ECG heart-rate e2e investigation in `ageo-matcher`.

This document is intentionally high-level. It is meant to preserve the current context, explain the architectural direction, and define implementation phases that can be planned and coded independently without losing the core reasoning.

## Purpose

The current framework is capable of building executable candidate algorithms, but it is not yet robust enough to reliably converge on high-quality algorithms quickly for simple problems. The ECG heart-rate benchmark exposed that weakness clearly.

The goal of this plan is to shift the system from:

- decomposition first, constraints later
- weakly typed skeletons
- topology-only expansion rules
- late discovery of obviously bad candidates
- sparse runtime telemetry

to:

- constraint-first planning
- semantically typed skeletons and edges
- boundary-aware, audit-ready expansions
- deterministic early rejection of bad candidates
- compact but informative telemetry that supports refinement

This is not an ECG-specific plan. ECG heart-rate is only the motivating failure case because it is simple enough that the framework should have solved it quickly and correctly.

## Motivating Failure

The ECG heart-rate investigation exposed several framework-level problems:

1. The final verified CDG remained the plain 3-step scaffold:
   filter -> detect peaks -> compute rate

2. The selected atom chain was weak:
   `pronto.bandpass_filter` -> `pronto.r_peak_detection` -> `biosppy.heart_rate_computation`

3. No refinement rules were actually applied, even though the framework contains signal-event-rate expansion logic.

4. The selected detector collapsed on longer windows because it used brittle global-threshold logic and hard-coded sampling assumptions.

5. The runtime context used by refinement was not canonical enough:
   ECG data was available under `ecg`, not canonical `signal`, and canonical `sampling_rate` was populated from the wrong group in a multi-stream dataset.

6. The jump-removal rule existed, but it could not apply because the rewrite engine expected a graph shape with an explicit source node that the actual scaffold did not represent.

7. The reference-loss profiling path bypassed the richer subprocess trace path, so intermediate artifacts were not preserved in a form that could drive refinement.

The important conclusion is that the framework was deterministic about the wrong abstractions. It consistently repeated a weak decision instead of deterministically pruning it.

## Design Intent

The framework should behave like a disciplined synthesis system, not a loose collection of heuristics.

Before constructing a first candidate CDG, it should know:

- what information must be preserved
- what forms of loss are allowed between stages
- what invariants downstream stages require
- what kinds of outputs make a candidate inadmissible
- what refinement operators are legal for that algorithm family

The system should then represent those decisions in artifacts that are:

- explicit
- auditable
- reusable
- versioned
- typed
- subject to the same rigor as atoms in `../ageo-atoms`

## Core Architectural Direction

### 1. Constraints become first-class planning artifacts

The Architect should produce a constraint system before or alongside the first skeleton. These constraints should not be informal prompt-only ideas. They should be structured, typed, and persisted.

Examples of the kinds of constraints the framework should support:

- data-kind constraints:
  waveform, event-sequence, rate-series, state, feature-vector, mask

- provenance constraints:
  which input stream a value came from and at what sampling or time basis

- loss constraints:
  information-preserving, lossy-but-allowed, irreversible, monotone-only, alignment-preserving

- invariants:
  monotonic timestamps, minimum cardinality, plausible ranges, continuity, bounded outlier rate

- stage preconditions:
  peak detector expects conditioned waveform
  rate estimator expects plausible event sequence

- family constraints:
  signal-to-event-to-rate pipelines should preserve waveform integrity until the event-extraction boundary

### 2. Skeletons should be represented as typed, auditable CDGs

Skeletons should not remain special-case internal templates with weaker semantics than atoms.

They should become first-class artifacts with:

- explicit typed inputs and outputs
- edge semantics
- loss semantics
- constraints
- audit metadata
- references and provenance
- uncertainty or confidence annotations where appropriate
- dejargonized documentation

The long-term target is for skeletons to live in `../ageo-atoms` as auditable assets, using the same quality bar as any other reusable atom-family artifact.

### 3. Expansion operations should also be represented as auditable graph artifacts

Expansion rules currently mix:

- semantic intent
- topology matching
- rewrite mechanics

inside local framework code.

Instead, the reusable refinement operations for an algorithm family should be promoted into versioned, auditable assets, ideally in `../ageo-atoms`, with:

- explicit CDG-shaped before/after forms
- typed ports
- constraints on applicability
- references and rationale
- uncertainty and caveat metadata
- documentation suitable for review

The local framework can still host the runtime rewrite engine, but the family-specific expansion inventory should be treated as shared knowledge, not hidden local logic.

### 4. Rewrites must operate on semantic boundaries, not just internal graph topology

The ECG jump-removal failure showed that topology-only rewriting is too brittle. The rule was conceptually correct but could not match the graph because the scaffold did not model root inputs as explicit nodes.

The framework must support rewrite targets such as:

- before the first consumer of root input `signal`
- between a waveform-producing stage and an event-producing stage
- after an event-producing stage but before a rate estimator

This requires richer graph semantics and/or first-class boundary nodes or ports.

### 5. Determinism should prune bad candidates early

The purpose of determinism is not merely reproducibility. It is fast elimination of weak candidates.

Every candidate should face admissibility checks before expensive optimization or deep refinement continues.

Examples:

- event count too low for window duration
- implausible event density
- output range incompatible with downstream semantics
- use of atoms that ignore required context such as sampling rate
- evidence of catastrophic threshold collapse

These checks should be generic at the framework level, with family-specific extensions layered on top.

### 6. Telemetry must be compact, structured, and refinement-oriented

The framework does not need to persist every intermediate tensor by default, but it does need enough structured telemetry to support deterministic refinement.

At minimum, each stage should be able to emit compact summaries such as:

- waveform stats:
  quantiles, clipping, local energy, max spike, band energy, discontinuity counts

- event stats:
  count, density, interval median, interval MAD, outlier fraction

- rate stats:
  mean, spread, missing fraction, plausibility metrics

- detector stats:
  threshold used, refractory period, candidate count before pruning

These summaries should be standardized enough that rule engines and admissibility checks can depend on them.

## High-Level Issues To Address

### Representation gap

The current CDG does not fully represent boundary inputs, semantic edge meaning, or information-loss assumptions. This makes correct rewrites hard and brittle.

### Constraint gap

Constraints exist mostly as implicit expectations in prompts, code, or evaluator logic. They are not a first-class system that guides planning from the start.

### Audit gap

Skeletons and expansions are not yet held to the same auditability and documentation standard as atoms in `../ageo-atoms`.

### Refinement gap

Expansion rules are too syntactic and too dependent on exact graph shape. They do not robustly express semantic applicability.

### Telemetry gap

Runtime artifacts are not canonical or rich enough to support reliable refinement. Important intermediate facts are either missing or keyed inconsistently.

### Search-discipline gap

Weak candidates can survive too long because there are not enough deterministic admissibility gates before expensive optimization or export/profile work.

## Phased Plan

## Phase 1: Constraint-First Planning Contract

### Intention

Introduce a formal planning artifact that captures required invariants and information-flow constraints before candidate synthesis begins.

### Scope

- define a first-class constraint schema
- extend the Architect contract so decomposition emits:
  goal, skeleton intent, constraints, and admissibility expectations
- ensure the Principal and rewrite engine can consume these constraints

### Desired Outcome

Every synthesized candidate is generated under an explicit constraint set rather than inferred only from prompt text.

### Exit Criteria

- a persisted constraint artifact exists for every top-level synthesis run
- constraints can be inspected independently of the CDG
- downstream components consume the constraint artifact rather than duplicating assumptions

## Phase 2: Skeletons As Auditable Family Assets

### Intention

Promote skeletons from internal scaffolds to reusable, typed, auditable assets.

### Scope

- define a skeleton artifact format compatible with CDG-level semantics
- move skeleton-family ownership toward `../ageo-atoms`
- require typed inputs/outputs, edge semantics, constraints, provenance, uncertainty, and dejargonized documentation

### Desired Outcome

Skeletons become reviewable family knowledge rather than opaque framework internals.

### Exit Criteria

- at least one family skeleton inventory is versioned as auditable assets
- skeletons can be validated with the same rigor expected of atoms
- decomposition can target a named skeleton asset instead of inventing all structure ad hoc

## Phase 3: Expansion Operations As Auditable CDG Assets

### Intention

Treat expansions as reusable algorithm-family operations with the same quality bar as other reusable knowledge.

### Scope

- define an expansion artifact format with applicability constraints and before/after graph forms
- move family-specific expansion operations into `../ageo-atoms`
- attach references, rationale, uncertainty, and documentation

### Desired Outcome

Expansion logic becomes inspectable and sharable across benchmarks rather than buried in local imperative code.

### Exit Criteria

- expansion assets exist outside the local rewrite engine
- local rules can directly reference audited expansion assets
- family expansion inventories are reviewable independently of runtime code

## Phase 4: Semantic CDG And Boundary-Aware Rewriting

### Intention

Make the graph representation expressive enough that semantically correct rewrites are straightforward to apply.

### Scope

- represent root inputs and outputs as first-class boundary nodes or ports
- enrich edges with semantic data-kind and loss information
- support rewrite matching against semantic boundaries, not only literal internal node shapes

### Desired Outcome

A rewrite such as jump removal before the first signal-conditioning stage can apply because the graph actually represents that boundary.

### Exit Criteria

- rewrite matching can target root-input boundaries
- edge semantics can distinguish waveform, event sequence, rate series, masks, and state
- topological rewrites no longer depend on fake internal source nodes

## Phase 5: Deterministic Admissibility Gates

### Intention

Reject obviously bad candidates early and reproducibly.

### Scope

- add framework-level admissibility checks
- allow family-specific admissibility extensions
- wire these checks into retrieval, synthesis, and optimization loops

### Desired Outcome

Candidates that are structurally executable but semantically poor do not survive long enough to dominate iteration time.

### Exit Criteria

- candidate rejection happens before expensive optimization where possible
- admissibility decisions are logged and inspectable
- failures such as catastrophic event-count collapse are caught deterministically

## Phase 6: Canonical Runtime Context And Refinement Telemetry

### Intention

Ensure the refinement engine sees the right data, under canonical keys, with enough summary statistics to reason about failure.

### Scope

- canonicalize signal, events, rates, masks, and per-stream sampling metadata
- support multi-stream datasets without poisoning canonical context
- persist compact per-node and per-edge summaries
- unify profiling and evaluation telemetry paths

### Desired Outcome

Refinement rules and attribution logic operate on reliable, family-agnostic telemetry rather than accidental dataset-specific keys.

### Exit Criteria

- canonical runtime context is stable across datasets
- telemetry exposes enough summary information to trigger and justify refinements
- profiling does not bypass the useful artifact path

## Phase 7: Search Discipline And Benchmark Reform

### Intention

Update evaluation strategy so benchmarks measure the real framework and reward robust search behavior rather than shortcut success.

### Scope

- keep e2e benchmarks on rich atom inventories
- prefer behavioral and fuzzy assertions over brittle exact-shape assertions
- measure search quality, admissibility behavior, refinement usage, and audit coverage

### Desired Outcome

Benchmarks validate framework robustness, not special-cased success paths.

### Exit Criteria

- e2e tests exercise the real retrieval and refinement stack
- benchmark assertions focus on semantic success and auditability
- benchmark artifacts make poor search behavior easy to diagnose

## Cross-Cutting Principles

### Shared rigor across repos

Anything the framework relies on as reusable family knowledge should be eligible for the same rigor as ageo-atoms artifacts:

- references
- uncertainty
- auditability
- documentation quality
- provenance
- reviewability

### No hidden family magic

Algorithm-family improvements should be encoded as explicit assets and rules, not hidden one-off special cases.

### Semantics over names

The system should reason primarily from typed semantics and constraints, not from fragile naming conventions.

### Deterministic failure is valuable only if it is informative

When the system rejects or refines a candidate, it should leave behind enough context to explain why.

## Out Of Scope For This Plan

This document does not prescribe:

- the exact final schema for constraints, skeletons, or expansion assets
- the exact repo layout in `../ageo-atoms`
- the exact rollout order of individual code patches within a phase
- family-specific tuning for ECG only

Those should be handled in phase-specific implementation plans.

## Success Criteria

The framework should be considered materially improved when:

- simple algorithm families converge quickly without special shortcuts
- refinements apply because graph semantics support them, not by luck
- bad candidates are rejected early for principled reasons
- telemetry is sufficient to explain failure without manual forensic work
- skeletons and expansions are auditable shared assets rather than local hidden logic
- benchmark results provide confidence in the general framework, not just the current example

## Immediate Implications For Future Phase Plans

When writing implementation plans for the phases above, prioritize the following order:

1. constraint-first planning contract
2. semantic CDG boundary model
3. canonical runtime context and telemetry
4. admissibility gates
5. skeleton and expansion asset migration into `../ageo-atoms`

This order keeps the early work focused on representation and observability before migrating knowledge assets into their long-term home.

## Why This Matters

The framework is supposed to make algorithm synthesis:

- faster
- more reliable
- more auditable
- more deterministic

The ECG heart-rate benchmark showed that determinism without strong constraints and strong telemetry only makes failure repeatable.

This plan is intended to change that by making the system deterministic about the right abstractions.
