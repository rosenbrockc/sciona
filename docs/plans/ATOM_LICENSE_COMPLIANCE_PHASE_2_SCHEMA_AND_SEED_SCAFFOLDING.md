# Atom License Compliance Phase 2: Schema And Seed Scaffolding

## Goal

Add the additive schema and provider-owned ingestion scaffolding required for version-scoped license metadata.

## Scope

- additive Supabase migration in `../sciona-infra`
- provider-owned parsing and normalization helpers in `../sciona-atoms`
- matcher-side read models only where needed for later filtering

## Schema Targets

- `atom_versions.license_expression`
- `atom_versions.license_status`
- `atom_versions.license_family`
- `atom_versions.license_source_kind`
- `atom_versions.license_source_path`
- `atom_versions.upstream_license_expression`
- `atom_versions.license_notes`
- parity fields for `artifact_versions`

## Tasks

1. Create additive migration for atom/artifact version license metadata.
2. Add provider-owned SPDX normalization helpers.
3. Add provider inventory helpers to discover license manifests/files.
4. Add seed scaffolding that can load license metadata without yet enforcing it.
5. Add focused tests for normalization, precedence, and null handling.

## Parallelization

- schema work should be single-owner in `../sciona-infra`
- parsing/helper work can proceed in parallel in `../sciona-atoms`

## Acceptance

- local schema reset succeeds with new license fields
- provider seed code can parse and stage license metadata
- no runtime filtering is enabled yet
