# Atom License Compliance Phase 3: Provider License Manifests

## Goal

Author provider-owned license manifests and overrides for the sibling atom repos.

## Scope

- repo-level default manifests
- family-level overrides where upstream or vendored licensing differs
- no public enforcement yet

## Standard Manifest Shape

- `scope`
- `scope_key`
- `license_expression`
- `license_status`
- `license_family`
- `source_kind`
- `source_path`
- `upstream_license_expression`
- `notes`

## Worker Split

- worker A: `sciona-atoms` core families
- worker B: `sciona-atoms-signal`
- worker C: `sciona-atoms-bio` + `sciona-atoms-robotics`
- worker D: `sciona-atoms-fintech` + `sciona-atoms-physics`
- worker E: `sciona-atoms-ml`

## Tasks

1. Add repo-level default license manifests.
2. Add family overrides for vendored or differently licensed upstream-derived families.
3. Add validation tests that each manifest is parseable and uses normalized expressions.
4. Mark unresolved or legally ambiguous families as `unknown` or `needs_legal_review`.

## Acceptance

- every provider repo has a committed license manifest
- known family overrides are explicit
- unresolved cases are fail-closed, not silently inferred
