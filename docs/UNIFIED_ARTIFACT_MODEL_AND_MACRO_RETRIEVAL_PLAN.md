# Unified Artifact Model And Macro Retrieval Plan

This document proposes a concrete path to make published CDGs first-class
reusable solutions without forcing them to masquerade as leaf atoms.

Worker-facing execution phases for this design live in:

- [UNIFIED_ARTIFACT_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/sciona-matcher/docs/plans/UNIFIED_ARTIFACT_IMPLEMENTATION_PLAN.md)
- the linked phase docs under `docs/plans/`

The target outcome is:

- a user can ask for a high-level capability such as "find rate from signal"
- the system can directly select a published composite algorithm when one
  already exists and matches the contract well
- the system only falls back to fresh decomposition and leaf grounding when no
  strong macro candidate exists

## Why This Is Needed

The current runtime is leaf-centric:

- direct grounding asks Hunter for one match to the whole goal in
  [sciona/services/planner_service.py](/Users/conrad/personal/sciona-matcher/sciona/services/planner_service.py)
  and [sciona/services/hunter_service.py](/Users/conrad/personal/sciona-matcher/sciona/services/hunter_service.py)
- structured grounding exports only atomic CDG leaves through
  [to_pdg_nodes()](/Users/conrad/personal/sciona-matcher/sciona/architect/handoff.py:357)
- Hunter retrieval searches declaration candidates per leaf in
  [sciona/hunter/nodes.py](/Users/conrad/personal/sciona-matcher/sciona/hunter/nodes.py)
- template retrieval exists, but it is used as a decomposition aid inside the
  Architect in [sciona/architect/nodes.py](/Users/conrad/personal/sciona-matcher/sciona/architect/nodes.py),
  [sciona/architect/template_retriever.py](/Users/conrad/personal/sciona-matcher/sciona/architect/template_retriever.py),
  and [sciona/architect/graph_retrieval.py](/Users/conrad/personal/sciona-matcher/sciona/architect/graph_retrieval.py)

That means a published CDG can help only if:

- the Architect independently rediscovers a similar graph, or
- the whole goal happens to look like one leaf-sized declaration

This is the wrong bias for "best solution" retrieval. Composite algorithms
should be directly retrievable as composite artifacts.

## Design Rule

Do not treat CDGs as fake atoms inside the current Hunter declaration index.

Instead:

- atoms remain leaf-capable executable primitives
- CDGs become composite executable artifacts
- both share the same metadata and quality model
- retrieval chooses the right granularity before leaf grounding starts

## Proposed Artifact Model

Introduce a unified artifact layer with `artifact_kind`:

- `atom`
- `cdg`

Keep UUID primary keys and keep `content_hash` as the immutable artifact
identity. Use `(fqdn, semver)` as the public version lookup key.

### Core Tables

Add:

- `artifacts`
  - `artifact_id UUID PRIMARY KEY`
  - `artifact_kind TEXT CHECK (artifact_kind IN ('atom', 'cdg'))`
  - `fqdn TEXT UNIQUE NOT NULL`
  - `owner_id UUID`
  - `source_repo_id UUID`
  - `namespace_root TEXT`
  - `namespace_path TEXT`
  - `source_package TEXT`
  - `source_module_path TEXT`
  - `source_symbol TEXT`
  - `status TEXT`
  - `visibility_tier TEXT`
  - `description TEXT`
  - `source_kind TEXT`
  - `stateful_kind TEXT`
  - `is_stochastic BOOLEAN`
  - `is_ffi BOOLEAN`
  - `is_publishable BOOLEAN`
  - `topo_hash TEXT NOT NULL DEFAULT ''`
  - `top_level_input_arity INTEGER NOT NULL DEFAULT 0`
  - `top_level_output_arity INTEGER NOT NULL DEFAULT 0`
  - `leaf_count INTEGER NOT NULL DEFAULT 0`
  - `verified_leaf_coverage DOUBLE PRECISION NOT NULL DEFAULT 0`

- `artifact_versions`
  - same shape as `atom_versions`, but keyed by `artifact_id`
  - unique `(artifact_id, semver)`
  - unique `content_hash`

### Shared Metadata Tables

Mirror the current atom tables with `artifact_id` / `version_id`:

- `artifact_io_specs`
- `artifact_parameters`
- `artifact_descriptions`
- `artifact_references`
- `artifact_audit_rollups`
- `artifact_audit_evidence`
- `artifact_uncertainty_estimates`
- `artifact_verification_matches`
- `artifact_hyperparams`
- `artifact_benchmarks`

The point is not to duplicate business rules. The point is to let both atoms and
CDGs pass through the same metadata and publishability gates.

### CDG-Specific Structure Tables

Add:

- `artifact_cdg_nodes`
  - `version_id`
  - `node_id`
  - `name`
  - `description`
  - `concept_type`
  - `status`
  - `type_signature`
  - `matched_primitive`
  - `parent_node_id`

- `artifact_cdg_edges`
  - `version_id`
  - `source_id`
  - `target_id`
  - `output_name`
  - `input_name`

- `artifact_cdg_bindings`
  - binds CDG nodes to reusable artifacts
  - minimum useful columns:
    - `version_id`
    - `node_id`
    - `bound_artifact_fqdn`
    - `bound_version_content_hash`
    - `binding_confidence`
    - `binding_source`

This lets a published CDG remain executable and inspectable as a graph, not just
as a blob.

## Publishability

Replace `atom_is_publishable()` with a generic `artifact_is_publishable()`.

For both `atom` and `cdg`, publishability should require:

- IO specs
- parameters or top-level callable contract
- a dejargonized description
- an audit rollup
- references

For `cdg` specifically, add:

- at least one `artifact_cdg_nodes` row
- at least one terminal binding or verified leaf
- acyclic graph validation
- valid top-level contract derivation from the graph

This preserves the current atom quality bar while making macro artifacts obey
the same trust model.

## Runtime Changes

### 1. Direct Macro Retrieval Before Hunter

Add a new retrieval stage before the current direct Hunter path in:

- [sciona/services/planner_service.py](/Users/conrad/personal/sciona-matcher/sciona/services/planner_service.py)
- [sciona/runtime_paths.py](/Users/conrad/personal/sciona-matcher/sciona/runtime_paths.py)

New order:

1. direct macro retrieval against published `cdg` artifacts
2. if a strong macro match exists, return a one-node "artifact selected" CDG
3. otherwise run the current direct Hunter path
4. otherwise decompose and continue as today

The retrieval query should use:

- top-level type signature / IO contract
- goal text
- dejargonized description
- domain tags
- data-modality hints such as signal / image / graph / time series

This is the main change that improves "best solution" probability.

### 2. Architect Reuse Of Published CDGs

Keep and extend the existing template flow in:

- [sciona/architect/nodes.py](/Users/conrad/personal/sciona-matcher/sciona/architect/nodes.py)
- [sciona/architect/template_retriever.py](/Users/conrad/personal/sciona-matcher/sciona/architect/template_retriever.py)
- [sciona/architect/graph_retrieval.py](/Users/conrad/personal/sciona-matcher/sciona/architect/graph_retrieval.py)

Required changes:

- index published CDGs as reusable artifact exemplars, not only Memgraph-inserted
  decomposition traces
- boost exemplars with high `verified_leaf_coverage`
- let a template match terminate decomposition early when the graph-level match
  is strong enough

That turns published CDGs from "historical traces" into reusable macro assets.

### 3. Subgoal-Level Macro Retrieval During Refinement

In the orchestrator refinement loop in
[sciona/orchestrator.py](/Users/conrad/personal/sciona-matcher/sciona/orchestrator.py),
add a macro retrieval check before splitting a failed node into smaller leaves.

New rule:

- if a failed node has a published `cdg` artifact with matching top-level
  contract and acceptable confidence, substitute the macro artifact instead of
  decomposing further

This is how the system stops re-inventing a known algorithm mid-run.

### 4. Hunter Stays Leaf-Oriented

Do not refactor Hunter to verify arbitrary CDGs as if they were declarations.

Hunter should continue to handle:

- declaration retrieval
- candidate ranking
- candidate verification

Published CDGs should enter before Hunter or around Hunter, not inside the
declaration candidate loop.

## API And Catalog Changes

The current Supabase-facing catalog is atom-only through:

- `catalog_atoms_served`
- `get_atom_document(...)`

Add parallel artifact-facing surfaces:

- `catalog_artifacts_served`
- `get_artifact_document(request_fqdn TEXT)`

Then keep compatibility views:

- `catalog_atoms_served` becomes a filtered view over `catalog_artifacts_served`
  where `artifact_kind = 'atom'`
- existing atom endpoints stay stable during migration

Add a new API surface for macro search:

- `/catalog/search-artifacts`
- `/catalog/artifact/{fqdn}`

This keeps the old atom UX intact while enabling composite artifact retrieval.

## Indexing

The current semantic index is declaration-focused. Add an artifact index layer
for macro retrieval.

Index text for a `cdg` artifact should include:

- artifact fqdn
- top-level type signature
- technical description
- dejargonized description
- audit summary
- topological summary
- leaf primitive names
- concept-type multiset

This index is separate from the declaration index Hunter uses today.

## Rollout Plan

### Phase A: Additive Schema

- add `artifacts`, `artifact_versions`, shared `artifact_*` metadata tables,
  and `artifact_cdg_*` structure tables
- add generic `artifact_is_publishable()`
- keep all current `atom_*` tables and views working

### Phase B: Atom Back-Compat Population

- dual-write current atom seed/backfill paths into `artifacts` with
  `artifact_kind = 'atom'`
- build compatibility views or migration shims so current atom APIs still work

### Phase C: CDG Publication And Backfill

- ingest historical/published CDGs as `artifact_kind = 'cdg'`
- backfill top-level contracts, descriptions, audit summaries, references,
  uncertainty, and verification metadata
- populate `artifact_cdg_nodes` and `artifact_cdg_edges`

### Phase D: Macro Retrieval Runtime

- add direct macro retrieval before direct Hunter matching
- add subgoal macro substitution during orchestration refinement
- add artifact catalog endpoints and internal service layer

### Phase E: Architect Template Integration

- repoint template retrieval to prefer published `cdg` artifacts with strong
  verification coverage
- let high-confidence macro matches short-circuit fresh decomposition

### Phase F: Cutover And Simplification

- migrate `catalog_atoms_served` / `get_atom_document` internals to the unified
  artifact tables
- keep atom-only compatibility views for old clients
- deprecate atom-only assumptions in new code

## Decisions Still Needed

- Whether `fqdn` should remain globally unique across all artifact kinds, or be
  unique only per `artifact_kind`
- Whether a published CDG must always expose a top-level wrapper callable, or
  whether graph-level contract derivation is sufficient
- Whether CDG verification should require full executable replay, or whether
  verified leaf coverage plus structural validation is enough for early
  publishability
- Whether macro retrieval should be allowed to return a `cdg` artifact when a
  leaf atom also matches, or whether atoms should win unless the confidence gap
  exceeds a threshold

## Recommendation

Adopt:

- shared artifact model
- `artifact_kind in ('atom', 'cdg')`
- macro retrieval before leaf retrieval
- published CDGs as Architect template exemplars
- Hunter remaining declaration-focused

That is the smallest architecture change that materially improves the chance of
selecting an already-good composite solution instead of repeatedly decomposing
toward it.
