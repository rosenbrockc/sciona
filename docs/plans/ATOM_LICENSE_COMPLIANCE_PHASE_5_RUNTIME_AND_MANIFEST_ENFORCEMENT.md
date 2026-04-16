# Atom License Compliance Phase 5: Runtime And Manifest Enforcement

## Goal

Allow principals and manifest export to enforce license policy.

## Scope

- matcher runtime filtering
- manifest export filtering
- developer-mode override semantics

## Tasks

1. Add principal/runtime configuration for allowed licenses or license families.
2. Filter atom and artifact retrieval paths by license policy.
3. Enforce public-manifest exclusion for unknown or restricted license rows.
4. Allow developer-mode manifest inclusion behind explicit flags.
5. Surface license metadata in manifest metadata and catalog documents.

## Parallelization

- runtime retrieval filtering and manifest export filtering can run in parallel
- both depend on Phase 4 data availability

## Acceptance

- public manifests fail closed on unknown/restricted licenses
- developer mode can opt into unresolved licenses explicitly
- matcher retrieval respects configured license policy
