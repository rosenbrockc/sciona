# AGEO Atoms Heuristic Asset Ownership Plan

## Status

Drafted on April 9, 2026 after the first heuristic-evidence implementation in
`ageo-matcher`, the ECG heart-rate refinement investigation, and the follow-on
realization that the heuristic layer now exists conceptually but still has the
wrong repository ownership boundary.

This document is intentionally high-level. It is the ground-truth planning
reference for moving heuristic assets and metadata into `../ageo-atoms` without
losing the cross-family, cross-disciplinary design goals of the system.

Detailed implementation plans for each phase should be audited against this
document.

## Purpose

The heuristic layer now exists in `ageo-matcher` as:

- a canonical heuristic schema
- atom heuristic metadata models
- family heuristic registries
- runtime heuristic extraction
- heuristic-driven proposal guidance
- heuristic outcome memory

That was the right first step for validating the abstraction. It is no longer
the right long-term ownership model.

Heuristic definitions, heuristic-producing atom metadata, and family-level
heuristic assets should live alongside the audited atom ecosystem in
`../ageo-atoms`, not remain matcher-local framework state.

The goal of this plan is to make heuristics:

- first-class audited assets
- owned next to the atoms that produce or depend on them
- reusable across families and repositories
- documented in dejargonized language
- available to Sciona as data, not hidden framework code

## Current State

Today the canonical heuristic layer is implemented in `ageo-matcher`:

- shared schema in `sciona/heuristics.py`
- atom heuristic metadata model in `sciona/heuristic_metadata.py`
- family registry model in `sciona/heuristic_registries.py`
- registry assets under `sciona/principal/assets/heuristic_registries`

By contrast, `../ageo-atoms` currently has no canonical per-atom heuristic
metadata store and no canonical heuristic asset registry. It contains:

- ordinary audited atoms
- CDGs and witnesses
- audit manifests and references
- incidental domain outputs that could become heuristic producers
- documents that mention heuristics in other audit contexts

So this is not a migration of an already-canonical heuristic system. It is the
creation of that ownership surface in `ageo-atoms`, followed by a controlled
reduction of framework-local duplication in `ageo-matcher`.

## Why Ownership Must Move

### 1. Heuristic knowledge is algorithmic knowledge

Heuristics are not only runtime policy. They encode what kinds of intermediate
behavior are meaningful, which observations are worth preserving, and what
structural responses they justify. That belongs next to the audited atom and
family asset ecosystem.

### 2. Atoms need auditable heuristic contracts

If an atom output is usable as heuristic evidence, that fact should be reviewed
with the same rigor as:

- its signature
- its uncertainty notes
- its references
- its dejargonized documentation
- its witnesses and conceptual meaning

### 3. Family registries are family assets

A family registry is not just Principal policy. It is an auditable declaration
of which cross-family heuristics matter for a family, which producers are
sanctioned, and what structural actions are appropriate. That is family asset
knowledge.

### 4. The matcher should consume heuristic assets, not define them

`ageo-matcher` should remain the execution and search engine. It should load,
validate, and use heuristic assets from `ageo-atoms`, but it should not remain
the primary authoring home for those assets.

## Core Thesis

Heuristic assets should be promoted into the same audited asset universe as
atoms, skeletons, and expansions.

That means:

- canonical heuristic identifiers remain cross-family and dejargonized
- atoms may declare heuristic-producing outputs
- families may declare local heuristic interpretation and sanctioned actions
- Sciona runtime may derive additional heuristics from execution summaries
- proposal policy may consume all of the above deterministically

But the source of truth for the declarative assets should live in
`../ageo-atoms`.

## Design Intent

The future system should let the framework say:

- these are the canonical heuristics recognized across the ecosystem
- these atom outputs are valid heuristic evidence producers
- these family assets define how a family interprets those heuristics
- these runtime summaries or diagnostic atoms produced the current evidence
- these expansions or substitutions are sanctioned responses
- this historical evidence says how those responses have performed before

The intent is not to make `ageo-atoms` responsible for search policy. The
intent is to make it responsible for the auditable, reusable knowledge that the
search policy consumes.

## Cross-Family Principles

The ownership move must not collapse into signal-specific or ECG-specific
interfaces.

### 1. Canonical heuristics remain dejargonized

Shared heuristic IDs should describe observable behavior, not domain jargon.

Examples:

- `interval_instability`
- `boundary_discontinuity`
- `quality_instability`
- `density_collapse`
- `dominant_nuisance_structure`
- `residual_structure_after_transform`

Family docs can explain how a heuristic appears in ECG, EEG, optimization,
tracking, dynamic programming, graph search, or other domains.

### 2. Family interpretation stays local

Families should not redefine shared heuristic meaning. They may only specify:

- why the heuristic matters locally
- which producers are sanctioned
- which action classes are supported
- what admissibility or escalation consequences follow

### 3. Atoms declare evidence, not decision policy

An atom may say:

- this output is a usable heuristic
- this is the heuristic’s meaning and uncertainty
- this is the evidence shape and provenance

An atom should not directly encode global proposal scheduler behavior.

### 4. Runtime transforms remain valid producers

Not every heuristic should require a dedicated atom. Some are best derived from
runtime summaries. The system must continue to support:

- atom-output heuristic producers
- runtime-transform heuristic producers
- compatibility mappings during migration

### 5. Search policy remains matcher-owned

The ranking, budgeting, admissibility comparison, and trial routing logic stays
in `ageo-matcher`. Only the declarative heuristic assets move.

## Target Ownership Model

The long-term split should look like this.

### In `../ageo-atoms`

- canonical heuristic schema assets
- per-atom heuristic metadata
- family heuristic registries
- heuristic references and dejargonized documentation
- migration-ready family-level heuristic notes
- optional diagnostic atoms whose primary purpose is to produce heuristic
  evidence

### In `ageo-matcher`

- loaders and validators for heuristic assets
- runtime heuristic derivation for execution summaries
- proposal policy consuming heuristic assets
- benchmark/reporting surfaces
- temporary compatibility shims while assets migrate

## Asset Classes To Introduce In `ageo-atoms`

### Canonical heuristic definitions

A repository-level asset class for canonical heuristics should define:

- `heuristic_id`
- `display_name`
- `dejargonized_meaning`
- `applicability_scope`
- `evidence_type`
- `value_shape`
- `supported_action_classes`
- `uncertainty_notes`
- `references`

These assets are shared vocabulary, not family-local policy.

### Atom heuristic metadata

Each audited atom that emits usable heuristic evidence should be able to declare:

- which output is heuristic-bearing
- which canonical heuristic it supports
- producer kind
- expected evidence contract
- confidence interpretation notes
- provenance requirements
- uncertainty/failure notes
- references

This should be auditable next to the atom itself.

### Family heuristic registries

Each algorithm family should be able to declare:

- relevant canonical heuristics
- sanctioned producer kinds or producer atoms
- supported action classes
- action priority
- admissibility notes
- escalation conditions
- family-local interpretation notes

These remain family assets, not global schema.

### Diagnostic heuristic atoms

Some heuristics may be best produced by dedicated diagnostic atoms rather than
by repurposing a primary executable atom. The asset model should support that
cleanly.

## Architectural Challenges

### 1. Avoid duplicating schema logic across repositories

We should not end up with two drifting definitions of canonical heuristic
schema. The plan must choose a stable contract and one repository as the
authoring source of truth.

### 2. Avoid coupling `ageo-atoms` to matcher internals

Heuristic assets should not depend on Principal-specific bookkeeping fields or
proposal engine details. They must stay declarative.

### 3. Preserve transitional compatibility

`ageo-matcher` already consumes local heuristic assets. The migration should
allow both sources temporarily, with explicit precedence and validation, until
the canonical `ageo-atoms` versions are complete.

### 4. Keep families cross-disciplinary

The migration must not create family registries that are little more than
domain slang tables. The shared vocabulary and action classes have to remain
portable.

### 5. Govern quality and auditability

Heuristic assets need review requirements comparable to atoms:

- dejargonized explanation
- uncertainty and failure modes
- references where appropriate
- migration readiness
- review status

## Recommended Phase Structure

### Phase 1. Canonical Asset Surface In `ageo-atoms`

Create the canonical storage model in `ageo-atoms` for:

- heuristic definitions
- atom heuristic metadata
- family heuristic registries

Outcome:

`ageo-atoms` can represent the full heuristic layer declaratively, even before
all current matcher-local assets are migrated.

### Phase 2. Loader And Compatibility Layer In `ageo-matcher`

Teach `ageo-matcher` to load heuristic assets from `ageo-atoms`, with temporary
fallback to local assets during migration.

Outcome:

The matcher can consume external heuristic assets without breaking current runs.

### Phase 3. Migrate Canonical Heuristics

Move the canonical heuristic definitions out of `ageo-matcher` and into
`ageo-atoms`, while preserving a stable runtime API for the matcher.

Outcome:

Shared heuristic vocabulary is no longer matcher-local.

### Phase 4. Migrate Signal-Family Registries And Metadata

Move the current signal-family heuristic registry and the first wave of
signal-processing atom heuristic metadata into `ageo-atoms`.

Outcome:

The first concrete family proves the asset model end-to-end.

### Phase 5. Heuristic Metadata Audit System

Add a dedicated audit phase for heuristic definitions, atom heuristic metadata,
and family registries inside `ageo-atoms`.

This phase should extend the audit system so heuristic assets are checked for:

- schema validity
- duplicate or conflicting heuristic definitions
- missing dejargonized summaries or weak explanatory text
- banned or suspicious domain-jargon leakage in cross-family heuristic IDs and
  meanings
- family registries that redefine shared heuristic meaning instead of locally
  interpreting it
- invalid action classes, producer declarations, or unsupported scope claims
- missing references, uncertainty notes, or provenance requirements where the
  asset contract requires them
- cross-family violations such as family-local slang appearing in shared
  canonical assets

This phase should also define the review posture for heuristic assets:

- which findings are errors vs warnings
- when a family registry is allowed to introduce family notes
- how to validate that canonical heuristics stay portable across disciplines

Outcome:

Heuristic assets are governed by explicit audit rules that enforce
cross-family portability, dejargonization, and family-boundary discipline
rather than relying only on reviewer judgment.

### Phase 6. Loader And Compatibility Validation Tooling

Extend `ageo-matcher` validation and compatibility checks so the runtime can:

- prove which heuristic assets were loaded from `ageo-atoms`
- detect mismatched schema versions or missing migrated assets
- surface ambiguous dual-source ownership during the migration window

Outcome:

The migration remains observable and safe while both repositories may
temporarily contain overlapping heuristic definitions.

### Phase 7. Reduce Matcher-Local Source Of Truth

Deprecate matcher-local heuristic assets and keep only:

- compatibility adapters
- runtime-derived heuristics
- search policy consumption code

Outcome:

Ownership is clear and duplication is minimized.

### Phase 8. Cross-Family Rollout

Apply the same model to additional families beyond signal processing.

Outcome:

The heuristic asset model proves it can stay cross-family rather than becoming
a signal-only subsystem.

## Parallelization Analysis

Some phases can proceed in parallel if interfaces are fixed early.

### Wave A: Sequential foundation

These should happen in order:

- Phase 1
- Phase 2

The asset surface and loader contract need to stabilize before broader
migration.

### Wave B: Parallel asset authoring and audit tooling

Once the canonical asset shape is stable, these can run in parallel:

- Phase 3 canonical heuristic migration
- Phase 4 signal-family asset migration
- Phase 5 heuristic metadata audit system
- Phase 6 loader and compatibility validation tooling

These depend on the same schema but do not have to block each other if the
contract is fixed.

### Wave C: Sequential cleanup

- Phase 7 should follow after enough migration coverage exists.

### Wave D: Parallel family rollout

After the signal-family migration is stable, Phase 8 can be split by family and
run in parallel so long as each family owns a disjoint set of registry assets.

## Success Criteria

This plan should be considered successful when:

- `ageo-atoms` is the canonical authoring home for heuristic assets
- canonical heuristics are defined once and consumed everywhere
- signal-family heuristic metadata is loaded from `ageo-atoms`
- matcher-local heuristic assets are transitional or removed
- the audit system can validate heuristic assets with the same rigor as other
  auditable assets, including explicit cross-family and dejargonization checks
- cross-family naming remains dejargonized and portable
- new families can adopt heuristics without changing the shared schema

## Non-Goals

This plan does not propose:

- moving Principal search policy into `ageo-atoms`
- turning every runtime summary into a dedicated atom
- freezing family-specific behavior before the asset model stabilizes
- replacing runtime-derived heuristics with atom metadata only
- overfitting the schema around ECG or signal processing

## Immediate Next Step

The next document should be a phased implementation plan set for this ownership
move, starting with:

- the canonical `ageo-atoms` heuristic asset schema
- the matcher loader and compatibility contract
- the first signal-family migration slice

That phased set should treat this document as the audit baseline.
