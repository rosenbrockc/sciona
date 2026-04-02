# Sciona Infra Split Plan

## Goal

Create a new sibling repository named `sciona-infra` and move all website- and
infrastructure-facing code there, while keeping this repository usable for
algorithm generation, matching, ingestion, expansion, synthesis, and related
offline tooling.

The split should preserve two properties:

1. `ageo-matcher` remains installable and runnable without frontend, Docker,
   Supabase, FastAPI, Authentik, OPA, Temporal, or Sentry setup.
2. `sciona-infra` can depend on the published or path-installed core package
   from this repository instead of duplicating shared algorithmic logic.

## Current Boundary

### Keep in this repository

These are core library and CLI capabilities tied to algorithm generation and
atom reasoning:

- `sciona/architect/`
- `sciona/benchmarks/`
- `sciona/commands/` except web-only commands
- `sciona/data/`
- `sciona/ecosystem/` for shared pure models/helpers
- `sciona/expansion_atoms/`
- `sciona/hunter/`
- `sciona/indexer/`
- `sciona/ingester/`
- `sciona/judge/`
- `sciona/principal/`
- `sciona/provenance/`
- `sciona/services/` if not web/infra bound
- `sciona/synthesizer/`
- `sciona/visualizer/`
- `sciona/visualizer_api.py`
- `sciona/static/`
- `sciona/commands/visualize_cmds.py`
- `sciona/commands/telemetry_cmds.py`
- core docs for matching, ingest, expansion, and synthesis
- visualizer/telemetry docs for CDG inspection and pipeline debugging
- tests that exercise the above

### Move to `sciona-infra`

These are web product, hosting, database, and ops concerns:

- `frontend/`
- `docker/`
- `supabase/`
- `sciona/api/`
- `sciona/workflows/`
- infra-specific docs:
  - `docs/supabase/`
  - infra/runbook docs
  - web/API deployment docs
- infra-specific tests:
  - `tests/test_supabase_*`
  - `tests/test_api_models.py`
  - `tests/test_bounty_*`
  - `tests/test_enterprise_auth.py`
  - `tests/test_opa_policies.py`
- `tests/test_snapshot_generation.py`
  - any frontend static/build tests
- infra/backfill/migration scripts that require Supabase or product tables:
  - `scripts/backfill_*`
  - `scripts/migrate_phase1_*`
  - `scripts/validate_supabase_phase0.sh`
  - `scripts/generate_embeddings.py`
  - similar database publication scripts

### Hybrid code that should stay in core

Do not move pure domain logic just because the API currently uses it:

- `sciona/clearinghouse/` settlement logic
- `sciona/ecosystem/dashboard.py`
- `sciona/ecosystem/models.py`
- Memgraph-backed CDG visualizer and telemetry dashboard code
- any pure data models or deterministic helpers with no web/db/runtime coupling

The new infra repo should import these from the core package.

### Why the visualizer stays in core

The visualizer is tightly coupled to algorithm-generation internals rather than
the product website:

- [app.py](/Users/conrad/personal/ageo-matcher/sciona/visualizer/app.py)
  opens a Memgraph driver via `AgeomConfig.memgraph_*` and wires in the core
  telemetry store.
- [cdg.py](/Users/conrad/personal/ageo-matcher/sciona/visualizer/cdg.py)
  queries `Atom`, `InputPort`, `OutputPort`, and `DATA_FLOW` nodes/edges from
  the repo’s CDG graph schema.
- [isomorphisms.py](/Users/conrad/personal/ageo-matcher/sciona/visualizer/isomorphisms.py)
  performs structural similarity search over decomposed CDGs in Memgraph.
- [dashboard.py](/Users/conrad/personal/ageo-matcher/sciona/visualizer/dashboard.py)
  reads runtime pipeline telemetry from `sciona.telemetry`.
- [visualize_cmds.py](/Users/conrad/personal/ageo-matcher/sciona/commands/visualize_cmds.py)
  is a developer/operator CLI for local JSON CDGs and Memgraph-backed browsing.
- [docs/VISUALIZE.md](/Users/conrad/personal/ageo-matcher/docs/VISUALIZE.md)
  documents it as a CDG/telemetry inspection tool, not product UI.

So the visualizer should remain in the core repo as an optional developer tool,
even if the public website moves out.

## New Repository Shape

Recommended sibling layout:

```text
../ageo-matcher      # core algorithms + CLI
../sciona-infra      # web app, API, DB, infra, deployment
```

Recommended `sciona-infra` top level:

```text
sciona-infra/
  frontend/
  docker/
  supabase/
  sciona_infra/
    api/
    workflows/
    visualizer/
  tests/
  docs/
  pyproject.toml
  README.md
```

Important: do not keep the moved code under the `sciona` import path in both
repos. The core package should own `sciona.*`. The infra repo should use a new
package namespace such as `sciona_infra.*`.

## Dependency Strategy

`sciona-infra` should depend on core `sciona`, initially via editable sibling
install during development and later via versioned package release.

Recommended transition:

1. In `sciona-infra`, use a local editable dependency during development.
2. After the split stabilizes, publish/version the core package and pin
   `sciona-infra` to released versions.

This avoids code duplication for:

- settlement logic
- dashboard computations
- registry/catalog pure models
- any future algorithmic helpers used by the API

## Phase Plan

### Phase 1: Define stable ownership boundaries

Goal: identify exactly what is core versus infra and make imports explicit.

Tasks:

- Freeze a move list for directories, tests, scripts, and docs.
- Mark web-only commands in `sciona/commands/`.
- Audit imports from infra code back into core code.
- Audit imports from core code into `sciona.api`, `sciona.visualizer`, and
  `sciona.workflows`.

Deliverables:

- approved move inventory
- list of pure shared modules that remain in core
- list of code that must be renamed from `sciona.*` to `sciona_infra.*`

### Phase 2: Remove infra from core package metadata

Goal: make this repo installable without web/infra extras by default.

Tasks:

- Remove the `api` extra from this repo’s `pyproject.toml`.
- Remove or stub platform-facing CLI commands so they point users to
  `sciona-infra`.
- Keep the visualizer extra and visualizer CLI in core.
- Move or delete references in README that describe this repo as the home of
  the website/API stack.
- Keep only algorithm-generation-facing install instructions here.

Deliverables:

- slimmer `pyproject.toml`
- updated core README

### Phase 3: Create `sciona-infra` skeleton

Goal: create the sibling repo before moving code.

Tasks:

- initialize `../sciona-infra`
- add `pyproject.toml`
- add package namespace `sciona_infra/`
- add basic README explaining dependency on core `sciona`
- set up frontend/build/test tooling there

Deliverables:

- empty but runnable infra repo
- editable dependency on sibling core repo

### Phase 4: Move web and runtime packages

Goal: relocate Python web/runtime modules into the new namespace.

Tasks:

- move `sciona/api/` -> `sciona_infra/api/`
- move `sciona/workflows/` -> `sciona_infra/workflows/`
- rewrite imports from `sciona.api.*` to `sciona_infra.api.*`
- rewrite imports from `sciona.workflows.*` to `sciona_infra.workflows.*`

Required decoupling:

- keep imports from infra into core one-way
- eliminate core imports that require infra modules

Success criterion:

- core package imports without FastAPI or Temporal installed
- infra package imports and runs against sibling core install

### Phase 5: Move frontend and infrastructure assets

Goal: move the deployable product surface into the infra repo.

Tasks:

- move `frontend/`
- move `docker/`
- move `supabase/`
- move relevant GitHub Actions or recreate them in the new repo
- move infra docs and runbooks

Special note:

- the new Docker secret generator script belongs here, not in core
- all Sentry/Auth/OPA/Temporal bootstrap logic belongs here

### Phase 6: Move infra tests and scripts

Goal: keep test ownership aligned with runtime ownership.

Tasks:

- move API/infra tests into `sciona-infra/tests/`
- move Supabase fixtures and local-integration fixtures
- move backfill/migration scripts that operate on Supabase product tables
- keep only core/unit tests in this repository

Keep in core:

- tests for architect, hunter, ingester, expansion, synthesizer, and pure
  library helpers

### Phase 7: Repair interfaces

Goal: establish clean contracts between repos.

Tasks:

- define which models are imported from core versus duplicated as API DTOs
- replace any accidental file-path coupling between repos
- remove imports from core into frontend/db/runtime code
- add compatibility docs for local development with two sibling repos

Recommended rule:

- core exports deterministic logic and shared models
- infra owns transport DTOs, HTTP routes, DB migrations, and operational code

### Phase 8: Validate both repositories independently

Goal: prove the split worked.

For core repo:

- run core test suite only
- verify package install without `fastapi`, `uvicorn`, `supabase`, `temporalio`
- verify CLI flows for ingestion, matching, and synthesis

For infra repo:

- run frontend build
- run API/unit/integration tests
- run Supabase local integration tests
- validate Docker compose files

## File-Level Move Checklist

### Python modules to move

- `sciona/api/__init__.py`
- `sciona/api/app.py`
- `sciona/api/bounty_state.py`
- `sciona/api/deps.py`
- `sciona/api/models.py`
- `sciona/api/policy.py`
- `sciona/api/routers/*`
- `sciona/api/schema.sql`
- `sciona/api/snapshot.py`
- `sciona/api/telemetry.py`
- `sciona/workflows/*`

### Python modules to reevaluate before moving

- `sciona/commands/catalog_cmds.py`
- `sciona/commands/visualize_cmds.py`
- any script or command that imports `sciona.api` or `sciona.visualizer`

Recommended treatment:

- keep `visualize_cmds.py` in core
- reevaluate `catalog_cmds.py`: if it is primarily Supabase publication
  tooling, move it; if it is needed as a core read-only export utility, keep it
  but remove imports on moved modules

### Non-Python assets to move

- `frontend/*`
- `docker/*`
- `supabase/*`
- infra docs under `docs/supabase/`

### Tests to move

- `tests/test_api_models.py`
- `tests/test_bounty_state_machine.py`
- `tests/test_bounty_workflow.py`
- `tests/test_enterprise_auth.py`
- `tests/test_opa_policies.py`
- `tests/test_snapshot_generation.py`
- `tests/test_supabase_auth.py`
- `tests/test_supabase_endpoints.py`
- `tests/test_supabase_local_integration.py`
- `tests/test_supabase_snapshot.py`
- frontend static tests under `frontend/tests/`

### Tests to keep in core

- `tests/test_visualizer_api.py`
- any tests for Memgraph-backed retrieval/graph visualization helpers
- telemetry dashboard tests tied to `sciona.telemetry`

## Compatibility Risks

### Risk 1: core imports infra

If any core module imports `sciona.api` or
`sciona.workflows`, the split will fail immediately.

Mitigation:

- make this a hard gating check before code movement

### Risk 2: namespace collision

If both repos define `sciona.api`, editable installs will
be brittle and import resolution will be ambiguous.

Mitigation:

- rename infra package namespace to `sciona_infra`

### Risk 3: CLI regressions in core

Commands that boot API servers will break once moved.

Mitigation:

- remove or relocate API-only commands as part of the split, not after
- keep local CDG visualizer commands in core

### Risk 4: shared model duplication drift

API DTOs and core models may fork accidentally after the split.

Mitigation:

- keep deterministic domain logic in core
- keep transport/request models in infra
- document ownership per model family

## Recommended Execution Order

1. Approve ownership list and namespace strategy.
2. Create sibling `sciona-infra`.
3. Move Python web/runtime modules and fix imports.
4. Move frontend/docker/supabase/assets.
5. Move tests/scripts/docs.
6. Remove web/infra extras and commands from core.
7. Run independent validation in both repos.

## Immediate Follow-Up Work

After plan approval, implement in this order:

1. Add the Docker secret generator script, but place it in the new
   `sciona-infra` repo during the actual split rather than deepening infra
   ownership in core.
2. Create the sibling repo scaffold.
3. Perform the Python namespace migration first.

## Success Definition

The split is complete when:

- this repo installs and runs core algorithm-generation workflows without any
  web/infra dependencies
- `sciona-infra` contains the website, API, DB, workflow, and deployment stack
- `sciona-infra` imports shared deterministic logic from the core package rather
  than copying it
- both repos have their own focused test suites and READMEs
