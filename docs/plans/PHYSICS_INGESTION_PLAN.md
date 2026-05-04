# Physics Equation Ingestion Implementation Plan

## Purpose

This plan makes physics equations, dimensionality, derivation relationships, and
physics data sources first-class Sciona artifacts. The goal is exhaustive
external-knowledge capture with staged trust: equations should not be ignored
just because their immediate cross-disciplinary utility is unclear.

Priority controls review and publication order, not whether an equation is
retained. A narrow equation today may become useful later when its physical
mechanism, dimensional signature, validity regime, or symbolic topology matches
a novel synthesis problem.

## Core Policy

Every external equation should enter the system as one or more of these states:

- `raw_equation_candidate`: source-preserved, not trusted for execution.
- `math_atom`: abstract symbolic topology without physical grounding.
- `physics_atom`: named physical grounding with dimensions, variables,
  assumptions, regimes, references, and mechanism descriptors.
- `data_artifact`: constants, spectra, material tables, empirical ranges, or
  benchmark data used by equations.
- `derivation_cdg`: executable or inspectable derivation graph.

High-value means "publish first", not "ingest exclusively".

## Mechanism Descriptors

Each equation should carry mechanism tags where applicable. These tags are what
let apparently domain-specific equations become reusable external knowledge.

Initial controlled vocabulary:

- conservation
- transport
- diffusion
- wave propagation
- relaxation
- resonance
- attenuation
- impedance
- constitutive response
- equilibrium
- symmetry
- scaling law
- variational principle
- stochastic process
- field interaction

Where useful, also record effort-flow archetypes:

- effort
- flow
- storage
- resistance
- source
- sink
- coupling
- transformer

## Phase 0: Schema And Model Contracts

Owner: infra/schema worker.

Add additive schema support for:

- raw source snapshots,
- raw equation candidates,
- version-scoped symbolic expressions,
- symbolic variables,
- validity bounds and regimes,
- typed relationships between artifacts, versions, expressions, and data
  artifacts,
- dimensional IO parity on `artifact_io_specs`.

Required tables:

- `physics_ingest_snapshots`
- `physics_equation_candidates`
- `artifact_symbolic_expressions`
- `artifact_symbolic_variables`
- `artifact_validity_bounds`
- `artifact_relationships`

Required surfaces:

- `catalog_symbolic_artifacts` view for symbolic retrieval and coverage
  dashboards.
- `get_artifact_document(...)` should include symbolic expressions, variables,
  validity bounds, and relationships.

Acceptance:

- Migrations are additive and idempotent.
- Existing atom, CDG, and state artifact behavior is unchanged.
- Later workers have stable tables for raw capture, normalization, review, and
  publication.

## Phase 1: Source Snapshot And Adapter Layer

Owner: ingestion/source workers.

Build source-specific adapters that write immutable raw snapshots first:

- Wikidata: equation discovery, aliases, use relationships, references, Q ids,
  and P ids.
- QUDT: quantity kinds, units, dimension vectors, and unit conversions.
- Physics Derivation Graph: equations, inference rules, derivation edges, and
  assumptions.
- NIST CODATA and DLMF: constants and special mathematical functions.
- HITRAN and Materials Project: spectra and material property data artifacts.
- OPB: biological and physiological physical-process mappings.
- TheorIA and Phy-SRBench: seed and evaluation corpora, pending license and
  provenance checks.

Each adapter emits:

- source id,
- source version,
- retrieval timestamp,
- license and provenance metadata,
- raw payload hash,
- formula payloads,
- variable hints,
- references,
- relationship hints.

Acceptance:

- Adapters are idempotent.
- Re-running an adapter produces deterministic candidate manifests for unchanged
  source payloads.
- Raw candidates are retained even when parse or normalization fails.

## Phase 2: Symbolic Normalization Pipeline

Owner: symbolic pipeline worker.

Pipeline steps:

1. Parse source formula into SymPy.
2. Store `sympy.srepr`; never rely only on presentation LaTeX.
3. Canonicalize symbol names while preserving source aliases.
4. Compute exact expression hash, topology hash, and dimensional hash.
5. Resolve variables through QUDT and local `DimensionalSignature`.
6. Classify mechanism and effort-flow archetypes.
7. Attach references and confidence evidence.
8. Run SymPy round-trip and dimensional checks.
9. Emit review tasks for ambiguity instead of guessing.

Matcher fixes required:

- Support rational dimensional exponents. Current dimensional signatures are
  integer-only, and symbolic power inference currently casts numeric exponents
  to `int`.
- Extend unit parsing from local string heuristics to QUDT vector ingestion.
- Make unknown dimensions explicit instead of silently treating externally
  sourced unknown symbols as dimensionless.

Acceptance:

- Normalized equations can instantiate `SymbolicExpression`.
- AOT NumPy code generation succeeds for supported expression classes.
- Dimensional ambiguity fails closed and leaves a reviewable candidate.

## Phase 3: Existing Physics Atom Migration

Owner: physics repo worker.

Implement the current physics SymPy migration in
`sciona-atoms-physics/docs/PHASE4_SYMPY_MIGRATION_PLAN.md` using the new
symbolic store.

Order:

1. Pulsar and dispersion atoms.
2. Particle tracking geometry.
3. Astro and Skyfield atoms.
4. Tempo and time conversion atoms.
5. jFOF and Pasqal special cases.

For each atom:

- Add `expressions.py`.
- Replace `@register_atom` with `@symbolic_atom`.
- Add dimensions, constants, validity bounds, references, and mechanism
  descriptors.
- Publish matching artifact symbolic rows.

Acceptance:

- Existing physics tests pass.
- Matcher symbolic tests pass.
- A small dimensional CDG compiles end to end.

## Phase 4: PDG Relationship Ingestion And CDG Extraction

Owner: derivation graph worker.

Extract from the Physics Derivation Graph in two layers.

Relationship graph:

- Each PDG equation node becomes a symbolic candidate or artifact.
- Each inference edge becomes an `artifact_relationships` row.
- Inference rules become reusable operation labels such as substitute, solve,
  differentiate, integrate, limit, nondimensionalize, approximate, and simplify.

Derivation CDGs:

- Extract small verified derivation chains as Sciona CDGs when they have typed
  ports and clear operations.
- Initial targets:
  - alternate solved forms of the same law,
  - limiting-case derivations,
  - constitutive substitution chains,
  - nondimensionalization flows,
  - conservation-law-to-PDE derivations.

CDG shape:

- Nodes are symbolic operations or physics atoms.
- Edges carry variable binding, dimensions, assumptions, and source PDG
  inference ids.
- Outputs are equations or compiled functions.

Acceptance:

- At least one PDG-derived CDG publishes as `artifact_kind = 'cdg'`.
- The CDG passes graph validation.
- The generated audit graph includes PDG provenance and bibliography.

## Phase 5: Review, Trust, And Audit Workflow

Owner: audit/review worker.

Review states:

- `raw_imported`
- `parsed`
- `dimension_resolved`
- `symbolically_validated`
- `source_verified`
- `human_reviewed`
- `published`

Automated gates:

- SymPy parse round-trip.
- Dimensional consistency.
- Constants and data artifact references resolve.
- Validity bounds are present or explicitly unknown.
- References are registered.
- Generated NumPy has no runtime SymPy dependency.

Human review focuses on:

- ambiguous notation,
- source reliability,
- mechanism classification,
- validity regimes,
- multiple incompatible conventions.

Acceptance:

- Publishability distinguishes trusted symbolic atoms from raw candidates.
- Internal search can still surface raw candidates as external-knowledge
  suggestions requiring review.

## Phase 6: Retrieval And Synthesis Integration

Owner: matcher/runtime worker.

Extend retrieval to use:

- topology hash,
- mechanism descriptors,
- dimensions,
- validity bounds,
- source domains,
- relationship edges,
- known analogues,
- data artifact dependencies.

Planner behavior:

- Search all candidate equations for inspiration.
- Prefer published and reviewed artifacts for executable synthesis.
- Allow raw candidates only as external-knowledge suggestions requiring review
  before production code.
- Use dimensional checking as the compiler contract.

Acceptance:

- Retrieval can find famous cross-domain equations and obscure domain-specific
  equations when a problem matches mechanism, dimensions, or topology.

## Phase 7: Bulk Backfill Strategy

Owner: batch ingestion workers.

Run ingestion in expanding rings:

1. Foundational mechanics, thermodynamics, electromagnetism, waves, and
   transport.
2. Existing Sciona domains: biosignals, imaging, particle tracking,
   astrophysics, and materials.
3. Full Wikidata physical equation corpus.
4. Full Physics Derivation Graph equation and derivation corpus.
5. Constants, spectra, materials, and property datasets.
6. Long-tail equations with lower metadata quality.

Nothing is dropped. Lower-quality items remain in raw or candidate state until
they can be normalized.

Acceptance:

- Coverage dashboards show discovered, parsed, dimensioned, reviewed, and
  published counts by source and physics family.

## Phase 8: Offline Validation And CI Gate

Owner: validation/tooling worker.

Add a deterministic validation script, analogous to the existing atom and CDG
validators, so symbolic atoms, publication fixtures, PDG relationship
extraction, and derived CDG rows can be checked before any database writes.

Validator scope:

- load every checked-in physics publication fixture,
- optionally compare each fixture to a live
  `build_symbolic_publication_manifest(...)` render from `sciona-atoms-physics`,
- run matcher publication loading for symbolic expressions, variables, and
  validity bounds,
- enforce symbolic metadata standards: mechanism tags, behavioral archetypes,
  bibliography, dimensions, review status, validation status, and JSON-safe row
  payloads,
- parse PDG payload fixtures,
- build relationship rows and PDG CDG publication rows,
- run CDG graph validation on nodes, edges, bindings, and artifact-version
  envelopes,
- emit a machine-readable report for CI and dashboards.

Script modes:

- default: fast, offline, deterministic validation over local fixtures,
- `--strict`: fail when expected fixture families are absent,
- `--json`: emit the validation report as JSON,
- future `--changed-only`: restrict checks to touched fixtures and payloads for
  fast developer loops.

Acceptance:

- The validator exits nonzero on missing symbolic metadata, fixture drift,
  publication loader errors, malformed PDG edges, CDG graph errors, or
  nondeterministic publication rows.
- The report identifies each failing fixture or PDG payload with stable reason
  codes.
- CI can run the validator without Supabase, network access, or mutable local
  state.

## Parallelization Analysis

Safe parallel work after Phase 0 lands:

- Worker A: QUDT adapter and dimension-vector mapping.
- Worker B: Wikidata adapter and raw candidate envelopes.
- Worker C: PDG adapter and derivation-edge model.
- Worker D: CODATA/DLMF constants and data artifact publishing.
- Worker E: existing `sciona-atoms-physics` migration.
- Worker F: review workflow and audit evidence.
- Worker G: offline validation script, fixture inventory, and CI report wiring.

Work that should stay single-owner:

- schema migrations in `sciona-infra/supabase/migrations`,
- core `DimensionalSignature` changes,
- symbolic parser/canonicalizer interfaces,
- publishability rules,
- runtime planner ranking semantics.

Recommended dependency waves:

1. Wave 0: schema and model contracts. One worker.
2. Wave 1: source adapters in parallel. Workers A-D.
3. Wave 2: symbolic normalization plus physics atom migration.
4. Wave 3: PDG CDG extraction and audit workflow in parallel.
5. Wave 4: validation gate plus retrieval/runtime integration.
6. Wave 5: bulk backfill and dashboards.

## Exit Criteria

The work is complete when:

- all discovered equations have durable raw candidate records,
- reviewed equations publish as symbolic atom or CDG artifacts,
- dimensional signatures are available at artifact IO and symbolic variable
  levels,
- derivation relationships and selected PDG subgraphs are represented as
  inspectable Sciona artifacts,
- synthesis can use topology, mechanism, dimensions, validity, and provenance
  when selecting physics knowledge,
- audit documents can explain exactly which equations, constants, data sources,
  and derivations grounded a generated algorithm.

## Current Landed Publication Pipeline

The first publication slice is now implemented as a side-effect-free pipeline
with a caller-owned storage boundary. The concise maintainer reference is
`docs/physics_ingest_publication_pipeline.md`.

Current modules:

- `sciona.physics_ingest.ids`: deterministic UUIDv5 snapshot and candidate ID
  planning.
- `sciona.physics_ingest.staging`: Wave 0 row validation for source snapshots,
  equation candidates, symbolic expressions, and relationship-adjacent rows.
- `sciona.physics_ingest.normalization`: side-effect-free symbolic expression
  draft normalization with QUDT-assisted dimension resolution, conservative
  unit and quantity-kind alias matching, round-trip diagnostics, and
  reviewable ambiguity handling.
- `sciona.physics_ingest.publication`: publication manifest loading for
  symbolic expressions, variables, and validity bounds.
- `sciona.physics_ingest.orchestration`: combines source bundles and symbolic
  manifests into validated insert rows and diagnostics.
- `sciona.physics_ingest.write_plan`: builds dependency-ordered inert write
  plans with per-table insert/upsert modes.
- `sciona.physics_ingest.writer`: applies plans through an injected
  `PublicationTableClient` and supports dry runs.
- `sciona.physics_ingest.supabase_adapter`: adapts injected PostgREST/Supabase
  clients, preflights planned writes, and applies writes through the shared
  writer without constructing storage clients.
- `sciona.physics_ingest.deployment`: composes publication rows, PDG catalog
  projections, review queue rows, and audit artifact rows into one deterministic
  production storage bundle with an inert write plan and injected-client
  preflight/apply boundary.
- `sciona.physics_ingest.deployment_runtime`: composes side-effect-free
  production deployment preflight reports across full source runtime execution
  readiness and optional storage bundle preflight summaries.
- `sciona.physics_ingest.pdg_deployment`: merges PDG-derived CDG publication
  rows, catalog projection rows, storage preflight summaries, and optional
  injected-client storage apply results into one deterministic deployment plan.
- `sciona.physics_ingest.backfill_deployment`: wraps bulk backfill reports,
  persistable audit/dashboard artifact rows, storage preflight, runtime
  preflight, and optional injected-client storage apply results into a
  JSON-safe deployment report.
- `sciona.physics_ingest.planner_runtime`: batches symbolic retrieval planner
  service requests against an injected runtime planner client, preserving
  replay hashes, blocker counts, diagnostics, and dry-run/preflight state.
- `sciona.physics_ingest.sources.retrieval_plan`: emits deterministic
  executor-facing request envelopes for source jobs.
- `sciona.physics_ingest.sources.executor`: executes retrieval envelopes only
  through injected HTTP clients and snapshot sinks; dry runs and manual sources
  perform no IO.
- `sciona.physics_ingest.sources.runtime_adapters`: wraps injected HTTP/session
  objects and snapshot sinks into executor-ready adapters with capability
  reports, preflight metadata, and normalized snapshot receipts.
- `sciona.physics_ingest.sources.runtime_execution`: builds deterministic
  runtime execution/preflight reports around source run plans, adapter bundles,
  and optional injected execution.
- `sciona.physics_ingest.pipeline`: composes ID planning, orchestration, write
  planning, and optional execution.
- `sciona.physics_ingest.cli`: builds JSON-serializable dry-run reports from
  decoded payloads, including opt-in production-boundary sections for source
  runtime execution preflight, audit artifact write-plan rows, and review queue
  write-plan rows.
- `sciona.physics_ingest.validation`: offline validation for symbolic
  publication fixtures and PDG-derived CDG publication rows.
- `sciona.physics_ingest.backfill`: builds deterministic bulk backfill reports,
  including opt-in source request-envelope and publication write preflight
  sections for production execution review, plus opt-in persistable
  audit/dashboard artifact manifests and source runtime execution preflight
  sections.
- `sciona.physics_ingest.audit_artifacts`: turns backfill audit/dashboard
  artifact manifests into deterministic storage rows and optional inert write
  plans.
- `sciona.physics_ingest.pdg_cdg`: builds PDG relationship rows, validates
  derived CDG publication graphs, and projects PDG-derived CDGs into
  deterministic catalog/search rows and catalog-aware inert write plans.
- `sciona.physics_ingest.review`: materializes deterministic review queue task
  rows for `needs_human`, `blocked`, and human-reviewed audit completion states,
  and shapes those tasks into inert write plans for production review queues.
- `sciona.physics_ingest.retrieval_io`: plans and executes catalog/RPC fetches
  through injected clients for runtime retrieval and synthesis ranking, and
  wraps those fetches in planner request/response envelopes with replay hashes,
  compiler expectations, trust-policy blockers, and injected planner-service
  invocation envelopes.

Dry-run usage:

```python
from sciona.physics_ingest.cli import build_publication_dry_run_report_from_payload

report = build_publication_dry_run_report_from_payload(
    {
        "source_bundles": [source_bundle],
        "publication_manifests": [publication_manifest],
        "artifact_bindings": artifact_bindings,
        "table_modes": {"artifact_symbolic_expressions": "upsert"},
        "plan_ids": True,
    },
    include_rows=True,
)
```

The write-plan/writer contract is deliberately small: callers pass a mapping of
table names to row mappings, the planner orders known publication tables, and
the writer calls `client.insert(table, rows)` or `client.upsert(table, rows)`.
The writer never creates a Supabase client; production storage is an adapter at
the application boundary.
