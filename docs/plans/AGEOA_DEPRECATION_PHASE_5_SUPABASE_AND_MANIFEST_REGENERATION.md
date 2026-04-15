# AGEOA Deprecation Phase 5: Supabase And Manifest Regeneration

## Goal

Rebuild the local catalog and manifest stack after the cutover so the system is
proven clean end to end.

## Main Steps

1. Run a clean infra-root `supabase db reset --local --yes`.
2. Reseed from canonical `sciona-atoms*` providers only.
3. Rerun the deterministic metadata backfills.
4. Validate that:
   - `public.atoms` contains zero `ageoa.*`
   - benchmark, audit, references, uncertainty, and verification rows all bind
     to canonical `sciona.*` atoms
5. Regenerate the local SQLite manifest from the cleaned Supabase catalog.
6. Validate that the manifest also contains zero `ageoa.*` rows.

## Required Validation Queries

- `select count(*) from public.atoms where fqdn like 'ageoa.%';`
- equivalent checks for any artifact tables once artifact sync is rerun
- manifest-side queries against the generated SQLite file

## Exit Criteria

- clean local Supabase contains zero `ageoa.*` atoms
- regenerated manifest contains zero `ageoa.*` atoms or artifacts
- matcher runtime works against the regenerated canonical catalog

