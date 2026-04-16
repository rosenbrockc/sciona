# Atom License Compliance Phase 1: Inventory And Provenance

## Goal

Produce a deterministic inventory of current license provenance for all provider repos and existing catalog atoms.

## Scope

- inspect each sibling provider repo for repo-level license files
- inspect vendored/upstream source references already tracked in audit manifests
- classify each atom/version by likely license provenance source
- produce a machine-readable inventory report that later phases can consume

## Deliverables

- provider license inventory report
- upstream provenance inventory report
- list of atoms requiring manual legal review
- stable manifest schema for provider-owned license metadata

## Inputs

- `../sciona-atoms/src/sciona/atoms/provider_inventory.py`
- `../sciona-atoms/data/audit_manifest.json`
- sibling repo roots from `sources.yml`
- existing `references.json` and vendored source metadata

## Tasks

1. Enumerate provider repo roots and top-level license files.
2. Enumerate family-level or vendored license files where present.
3. Map atoms to provider repo ownership.
4. Map atoms to upstream provenance where available from audit manifests or vendor metadata.
5. Emit a report with these classifications:
   - `repo_root_only`
   - `family_override`
   - `upstream_vendor_only`
   - `conflicting_sources`
   - `missing_license_source`

## Parallelization

- one worker per provider repo may run this inventory in parallel
- one integrator combines the per-repo reports into a canonical inventory

## Acceptance

- every seeded atom is assigned a provenance class
- repo-level license sources are listed for every sibling provider repo
- unresolved atoms are explicitly reported rather than silently defaulted
