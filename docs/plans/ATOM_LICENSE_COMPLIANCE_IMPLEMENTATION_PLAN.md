# Atom License Compliance Implementation Plan

## Goal

Add first-class license support for atoms and artifacts so that:

- each published or developer-visible atom version has a recorded software license basis
- principals can be configured to allow only specific license sets
- manifest export can enforce license policy by tier
- we stop shipping or selecting atoms whose license status is unknown or incompatible

## Current Gap

As of the current local replay:

- there is no first-class atom or artifact license model in the Supabase schema
- there is no manifest field for atom or artifact licenses
- there is no matcher-side runtime filter for license policy
- the split provider repos do not currently expose a committed top-level `LICENSE` or `COPYING` file in the local checkouts
- provider seed/backfill paths do not ingest any SPDX-like license metadata

That means we cannot yet make a strong compliance claim for the ingested catalog beyond repo-by-repo manual inspection and reference-level inference.

## Design Rules

1. License must be version-scoped, not only atom-scoped.
   A license applies to a specific published artifact or wrapper version.

2. Store both normalized and source-of-truth forms.
   We need a normalized SPDX expression for filtering, plus the exact source path or provenance used to derive it.

3. Fail closed for public publication.
   Unknown or unresolved license status should block public publication.

4. Allow explicit developer-mode override.
   Developer manifests may include unresolved-license atoms only behind an explicit local feature flag.

5. Distinguish local wrapper license from upstream dependency license.
   Some atoms wrap or derive from upstream code under a different license than the local repo.

## Target Data Model

Short term, add version-level license metadata:

- `atom_versions.license_expression`
- `atom_versions.license_status`
  - `approved`
  - `restricted`
  - `unknown`
  - `needs_legal_review`
- `atom_versions.license_family`
  - `permissive`
  - `weak_copyleft`
  - `strong_copyleft`
  - `proprietary`
  - `unknown`
- `atom_versions.license_source_kind`
  - `repo_root_license`
  - `per_atom_manifest`
  - `upstream_vendor_license`
  - `manual_override`
- `atom_versions.license_source_path`
- `atom_versions.upstream_license_expression`
- `atom_versions.license_notes`

Unified artifact parity should mirror this on `artifact_versions`.

## Provider-Side Source Format

Add provider-owned license manifests under each repo, for example:

- `data/licenses/provider_license.json`
- optional per-family overrides:
  - `src/.../license.json`
  - `data/licenses/<family>.json`

Minimum fields:

- `scope`
- `scope_key`
- `license_expression`
- `license_status`
- `source_kind`
- `source_path`
- `upstream_license_expression`
- `notes`

Rules:

- repo-level defaults may apply broadly
- family- or atom-level entries override repo defaults
- missing entries remain `unknown`

## Runtime Policy

Add a matcher policy input such as:

- `SCIONA_ALLOWED_LICENSES=MIT,Apache-2.0,BSD-3-Clause`
- `SCIONA_ALLOWED_LICENSE_FAMILIES=permissive`
- `SCIONA_ALLOW_UNKNOWN_LICENSES=0`

Use it in:

- manifest sync selection
- catalog candidate filtering
- planner / hunter retrieval filtering
- artifact macro retrieval filtering

## Manifest Policy

Public manifests:

- must exclude atoms or artifacts whose `license_status != approved`
- must exclude atoms or artifacts with unknown license expression
- should optionally record tier-level license summary metadata

Developer manifest:

- may include unresolved-license rows only when developer mode is enabled
- should mark those rows explicitly as unresolved

## Compliance Sweep

Phase 1. Inventory and provenance

- enumerate provider repo roots and current top-level license files
- enumerate vendored upstream repos referenced by audit manifests
- classify atoms by likely license provenance source

Phase 2. Schema and seed/runtime scaffolding

- add version-level license fields for atoms and artifacts
- add compatibility views or RPC fields used by matcher and export

Phase 3. Provider manifests and family overrides

- add repo-level license manifests to each provider repo
- add family overrides where upstream licensing differs from repo default

Phase 4. Seed / backfill and one-off catalog enrichment

- ingest provider license manifests into Supabase
- derive version-level license rows deterministically
- mark unresolved atoms as `unknown`

Phase 5. Runtime and manifest enforcement

- add matcher-side license filtering for principal configuration
- add manifest export enforcement

Phase 6. Blessed compliance replay and baseline

- produce a catalog report:
  - approved licenses
  - unknown licenses
  - restricted licenses
  - atoms blocked from public publication on license grounds

## Parallelization

## Worker Execution Waves

Parallel wave A:

- Phase 1 integrator builds the canonical license manifest format and provenance report
- Phase 2 schema worker prepares additive migration + seed/runtime scaffolding
- provider workers inventory repo/family license sources in parallel from sibling repos

Parallel wave B after schema:

- provider manifest authoring by repo owner/family slice
- seeder/backfill implementation
- matcher manifest/runtime filtering

Final single-integrator wave:

- full replay
- compliance report
- public manifest gate enablement

## Phase Docs

- `ATOM_LICENSE_COMPLIANCE_PHASE_1_INVENTORY_AND_PROVENANCE.md`
- `ATOM_LICENSE_COMPLIANCE_PHASE_2_SCHEMA_AND_SEED_SCAFFOLDING.md`
- `ATOM_LICENSE_COMPLIANCE_PHASE_3_PROVIDER_LICENSE_MANIFESTS.md`
- `ATOM_LICENSE_COMPLIANCE_PHASE_4_BACKFILL_AND_CATALOG_ENRICHMENT.md`
- `ATOM_LICENSE_COMPLIANCE_PHASE_5_RUNTIME_AND_MANIFEST_ENFORCEMENT.md`
- `ATOM_LICENSE_COMPLIANCE_PHASE_6_BLESSED_COMPLIANCE_BASELINE.md`

## Immediate Recommendation

Before a full license baseline exists:

- keep public publication limited to the already-audited slice
- do not expand public publishability based on missing or inferred license state
- treat unresolved-license atoms as developer-only until the compliance sweep is complete
