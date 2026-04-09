# PEP 420 Federated Atom Providers Plan

## Purpose
Define the target architecture for moving from the current single-repo `ageoa` package model toward a federated `sciona.atoms.*` ecosystem built from multiple repositories, while keeping `ageo-matcher` cross-family, cross-disciplinary, and operational during the transition.

This document is the high-level architectural ground truth for the namespace and provider migration. Detailed implementation phases can be derived from it later, but the core decisions and invariants should remain stable.

## Why This Plan Exists
The current repository state is already moving toward a stronger abstraction boundary:
- canonical heuristics are now first-class and cross-family
- family registries and atom metadata are moving into `../ageo-atoms`
- proposal selection and runtime evidence are becoming more asset-driven

That direction is good. The risk is elsewhere: much of the system still treats `ageoa` as:
- the import namespace
- the filesystem layout
- the persisted FQDN prefix
- the single source of truth for atoms and atom assets

That model does not scale to the intended future, where different research communities contribute independently through separate repositories such as:
- `sciona.atoms.physics`
- `sciona.atoms.fintech`
- other future `sciona.atoms.<provider_or_domain>` packages

So the problem is no longer just “rename a package.” The real problem is designing a federated atom-provider architecture that:
- supports PEP 420 namespace packaging
- supports multiple independently versioned atom repositories
- keeps logical atom identity stable across packaging changes
- allows matcher/runtime systems to discover and use atoms and assets without hard-coding one repository or package prefix

## Desired End State
`ageo-matcher` should operate against a federation of atom providers rather than a single hard-coded package tree.

In the desired end state:
- atoms may be supplied by multiple repos
- those repos may publish into a shared namespace such as `sciona.atoms.*`
- matcher code does not need to know which repo physically owns an atom family
- logical atom identity is stable even if import paths or repo boundaries change
- assets such as heuristics, family registries, skeletons, expansions, references, uncertainty, and matches can be discovered across providers
- provider-local specialization is allowed, but shared abstractions remain canonical and cross-family

## Core Architectural Principles

### 1. Separate Logical Identity From Import Identity
Today many surfaces use `ageoa.<family>.<module>.<atom>` as both:
- the logical identifier
- the import path
- the persistence key

That needs to be split.

The architecture should distinguish:
- `logical_atom_id`
  - stable semantic identifier, independent of packaging or repo layout
  - example: `biosppy.ecg.r_peak_detection`
- `import_fqdn`
  - concrete Python import target for the currently active provider package
  - example today: `ageoa.biosppy.ecg.r_peak_detection`
  - example later: `sciona.atoms.signal.biosppy.ecg.r_peak_detection`
- `provider_id`
  - identifier for the supplying package/repo
  - example: `core.ageo_atoms`, `physics.default`, `fintech.lab_a`
- `asset_uri` or `asset_source`
  - where metadata and auditable assets came from

This split is the most important migration-reducing move. Without it, every package transition becomes a data migration and every new provider becomes a special case.

### 2. Make Providers First-Class
The system should treat atom providers as registered sources with declared capabilities.

A provider contract should include:
- provider ID
- import namespace roots
- asset roots
- supported asset classes
- version and compatibility metadata
- optional priority/precedence policy

This replaces implicit assumptions like:
- “the atoms live in `../ageo-atoms`”
- “the package is called `ageoa`”
- “the asset glob starts with `ageoa/**`”

### 3. Treat PEP 420 As a Packaging Strategy, Not the Whole Architecture
PEP 420 enables a shared namespace across multiple distributions. It does not solve:
- logical identity stability
- provider precedence
- asset discovery
- compatibility reporting
- migration semantics

So the system should not be designed around “a namespace package exists.” It should be designed around provider registration and logical identity. PEP 420 is then one packaging mechanism that helps implement the shared namespace.

### 4. Keep Shared Semantics Central, Keep Family Interpretation Local
The heuristic work already points in the right direction:
- canonical shared meaning should remain cross-family
- family-level interpretation should stay local
- provider-specific explanation should not leak into shared identifiers

The same rule should hold during federation:
- canonical abstractions remain global
- provider-local and family-local interpretation remain local assets
- a provider may contribute assets, but should not silently redefine shared meaning

### 5. Support Multi-Repo Contributions Without Central Monorepo Friction
The provider model should assume that different scientific or engineering communities contribute independently.

That means the architecture must tolerate:
- uneven release cadence
- overlapping capabilities
- partial provider availability
- provider-local optional dependencies
- provider-local audit maturity

The system should degrade gracefully when some providers are absent, outdated, or incompatible.

## What Must Change

## A. Atom Identity Model
All key systems should stop assuming that the persisted identifier is also the import target.

Affected surfaces include:
- heuristic metadata
- matches and references
- source catalog entries
- proposal models
- expansion assets
- benchmark and telemetry records

Target shape:
- persisted records should prefer `logical_atom_id`
- import-specific surfaces should store `import_fqdn` as a resolvable attribute, not the primary identity
- compatibility layers may continue to accept `ageoa.*` during migration

This identity model is also the right place to fix poor historical atom names.

Some current atom names are degraded artifacts of ingestion or normalization mistakes, including:
- collapsed camel-case names rendered as single lowercase words
- unreadable or overly compressed tokens
- inconsistent segmentation across related families

Those should not become the permanent canonical surface of the federated system.

So the migration should treat:
- `logical_atom_id` as the canonical rename target
- `import_fqdn` as a compatibility/runtime target
- legacy names as aliases, not as the enduring source of truth

The long-term goal is:
- readable canonical logical names
- stable compatibility aliases for older names
- provider-local packaging freedom without forcing ugly historical names to survive forever

### Naming Normalization Requirements
The migration should include a naming-normalization policy for canonical logical IDs.

That policy should enforce:
- correct word segmentation from historical camel-case or collapsed forms
- stable snake_case naming for canonical IDs
- readable but not over-dejargonized atom names
- consistency across sibling atoms in a family
- separation between canonical names, display labels, and legacy aliases

The policy should also distinguish between:
- `logical_atom_id`
  - canonical, readable, stable
- `display_name`
  - human-facing label that can be richer
- `legacy_aliases`
  - old atom names, old FQDNs, old import spellings, and migration shims

This is important because a federated ecosystem with multiple repos will otherwise multiply name debt instead of containing it.

## B. Provider Registry
Add a provider registry abstraction that owns:
- provider discovery
- package roots
- asset roots
- version/compatibility metadata
- precedence rules

The registry should be able to answer:
- which providers are installed
- which assets each provider contributes
- which provider currently resolves a given logical atom ID
- which providers are compatible with the current matcher/runtime version

## C. Asset Discovery
Asset discovery should move from fixed path conventions to provider-declared locations.

This includes:
- heuristics
- family registries
- skeleton assets
- expansion assets
- audit metadata
- references
- uncertainty
- matches

Each asset class should define:
- canonical schema
- provider discovery contract
- precedence/override policy
- audit requirements

## D. Import Resolution Layer
All direct imports of `ageoa.*` need to be isolated behind a resolver or runtime bridge.

The import resolution layer should support:
- current `ageoa` packages
- future `sciona.atoms.*` packages
- multiple installed providers
- provider fallback rules

This is especially important for:
- ghost simulation
- emitted/generated wrappers
- ingester output templates
- runtime verification environments
- source catalog import scanning

## E. Packaging Strategy
The eventual shared namespace likely requires turning the top-level `sciona` package boundary into something compatible with federated packaging.

That is a separate migration from the data/model changes above.

Important implication:
- this is not a simple find/replace of `ageoa` to `sciona.atoms`
- the existing `sciona` package in `ageo-matcher` means namespace packaging needs an explicit packaging plan

The packaging plan should be designed after the provider and identity model are stabilized, not before.

## Migration Strategy
The migration should be staged so the system remains operational throughout.

### Stage 1. Provider-Agnostic Identity
Introduce `logical_atom_id`, `provider_id`, and `import_fqdn` side by side with current `ageoa.*` fields.

Goal:
- all new systems can reason in provider-neutral terms
- old systems still work

This stage should also introduce canonical naming normalization fields:
- canonical `logical_atom_id`
- optional `display_name`
- `legacy_aliases`
- `legacy_import_fqdns`

This is the cleanest point to fix ugly historical names without breaking runtime compatibility.

### Stage 2. Provider Registry And External Asset Discovery
Replace direct sibling-repo assumptions with a provider registry and provider-scoped asset roots.

Goal:
- one matcher install can consume assets from multiple provider repos

### Stage 3. Import Resolution Abstraction
Replace direct `ageoa.*` imports with resolver-based imports.

Goal:
- runtime/synthesis/import systems no longer care whether the package is `ageoa` or `sciona.atoms.<x>`

### Stage 4. Shared Namespace Packaging
Move providers toward publishing under `sciona.atoms.*` once the logical and runtime contracts are stable enough.

Goal:
- packaging transition without requiring the whole system to relearn atom identity

### Stage 5. Source Reduction
Remove old `ageoa` assumptions from matcher code, docs, tests, and persisted asset generation.

Goal:
- `ageoa` becomes compatibility history, not a live architectural dependency
- naming cleanup becomes canonical rather than patchwork

## Parallelization Opportunities
Some parts of this can proceed in parallel once the identity model is defined.

### Wave 1: Must Start First
- logical identity model
- naming normalization policy
- provider registry contract
- asset discovery contract

These define the architectural boundaries. They should be designed before large implementation waves proceed.

### Wave 2: Can Run In Parallel After Wave 1
- heuristic asset loader migration
- skeleton and expansion asset provider loading
- source catalog/provider registration
- runtime import-resolution abstraction
- compatibility schema changes in telemetry and benchmark records

These share the same core contract but touch mostly different subsystems.

### Wave 3: Should Follow Once Wave 2 Has Meaningful Progress
- generated wrapper/emitter migration
- ghost simulation import migration
- benchmark and audit compatibility reporting
- rollout of additional external providers

These depend on the contracts from the earlier waves being real and tested.

### Wave 4: Final Packaging Transition
- PEP 420 namespace rollout
- deprecation/removal of old `ageoa` assumptions
- cleanup of compatibility aliases

This should happen after the runtime model already supports federation, not before.

## Risks

### Risk: Treating Import Path As Identity
If import FQDN remains the primary identity, every provider split or namespace move becomes a large migration with high breakage risk.

### Risk: Freezing Bad Historical Names Into Canonical IDs
If collapsed, ugly, or inconsistent atom names are carried directly into canonical logical IDs, the federated namespace will preserve low-quality naming across every future provider.

That would create long-term readability and governance debt even if the packaging migration succeeds.

The system should therefore normalize canonical names once, early, and preserve old names only as compatibility aliases.

### Risk: Over-Centralizing Provider Logic
If the matcher bakes in provider-specific exceptions, federation will collapse back into hard-coded special cases.

### Risk: Shared Meaning Drift
If providers can redefine canonical heuristic or asset meaning, cross-family reasoning will fragment.

### Risk: Packaging First, Contracts Later
If the system tries to jump to `sciona.atoms.*` packaging before logical identity and provider loading are stabilized, the migration will be brittle.

### Risk: Asset-Class Fragmentation
If each asset class invents its own provider/discovery logic, the system will accumulate hidden integration debt.

## Non-Goals
- This plan does not define the exact final Python package names for every provider.
- It does not yet specify the exact packaging mechanics for converting `sciona` into a namespace-compatible surface.
- It does not define the detailed schema migration for every persisted record.
- It does not require immediate deprecation of `ageoa`.

## Success Criteria
This plan is succeeding when:
- matcher can consume atoms and assets from more than one provider repository
- logical atom identity remains stable across provider and namespace changes
- canonical logical names are readable even when legacy import names were not
- runtime/import systems resolve provider atoms without hard-coded `ageoa` assumptions
- shared heuristic and other cross-family abstractions remain canonical and portable
- new providers can join without forcing matcher code changes
- the eventual PEP 420 move becomes a packaging step on top of an already-federated architecture, not a full-stack rewrite

## Immediate Recommendation
Do not treat the future as a one-time rename. Treat it as a federated provider migration.

The next planning layer should turn this into detailed phases around:
- logical atom identity and compatibility fields
- naming normalization and alias governance
- provider registry contract
- provider-scoped asset discovery
- import resolution abstraction
- packaging and rollout strategy for `sciona.atoms.*`
