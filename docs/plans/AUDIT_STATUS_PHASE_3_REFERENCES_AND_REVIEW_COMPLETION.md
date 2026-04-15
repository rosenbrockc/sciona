# Audit Status Phase 3: References And Review Completion

## Goal

Close the provider-side reference and review gaps that still prevent otherwise
audited atoms from becoming publishable.

## Why This Phase Matters

References are not the main blocker for the remaining `405` atoms, but they are
still missing for `159` of them. This phase exists so partially completed audit
bundles do not stall on inconsistent reference coverage or review conventions.

## Scope

This phase covers:

- provider `references.json` completion
- cross-provider `registry.json` normalization
- audit-manifest review status normalization
- canonical source attribution for wrapper atoms

This phase does not cover:

- inventing publication fallbacks
- source-derived publication of unaudited atoms

## Primary Repos

- all active sibling provider repos
- `../sciona-atoms` for shared registry tooling

## Worker Ownership

Parallel by provider repo, with one cross-provider integrator for registry and
review conventions.

Recommended split:

- provider workers fill repo-local references and review artifacts
- one integrator reconciles shared registry keys and review-state conventions

## Tasks

1. Fill missing provider references for the family buckets promoted in Phase 2.
   - Prefer canonical upstream sources and family-local evidence.

2. Normalize shared reference registry keys.
   - Ensure provider-local references resolve cleanly into the shared registry.

3. Normalize review status conventions.
   - Ensure audit manifests use consistent approved/rejected/deferred states.

4. Clean wrapper/source attribution where needed.
   - Reference the real upstream or family-owned source, not historical legacy
     locations.

5. Fail closed on unresolved review state.
   - If a family is not actually reviewed, keep it unpublished.

## Validation

- provider references parse and resolve cleanly
- shared registry entries are canonical and non-duplicative
- reviewed atoms have explicit, machine-readable review state
- no provider slice relies on manifest/export fallbacks to become publishable

## Exit Criteria

- the promoted provider slices from Phase 2 have complete references and review
  state
- the remaining non-publishable inventory is blocked by real missing audits, not
  by inconsistent reference/review metadata
