# Atom License Compliance Phase 4: Backfill And Catalog Enrichment

## Goal

Populate version-scoped license metadata for existing atoms and artifacts, including a one-off enrichment pass for the already-ingested catalog.

## Scope

- provider-owned seed/backfill in `../sciona-atoms`
- one-off update path for current local Supabase
- compliance reporting for unresolved rows

## Tasks

1. Extend provider seed to attach normalized license metadata to `atom_versions`.
2. Extend artifact seed/sync to attach license metadata to `artifact_versions`.
3. Add a one-off enrichment command for existing rows after reset/reseed.
4. Emit a report of:
   - approved rows
   - unknown rows
   - restricted rows
   - conflicting rows

## Parallelization

- atom version enrichment and artifact version enrichment can run in parallel
- provider-manifest ingestion is shared code and should be integrated once

## Acceptance

- local replay populates version-scoped license fields
- unresolved rows are explicitly reported
- no public filtering yet, but the catalog contains the needed metadata
