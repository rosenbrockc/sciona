# AGEOA Deprecation Phase 1: Canonical Namespace And Policy

## Goal

Define one explicit deprecation policy so all later worker changes follow the
same contract.

## Decisions To Freeze

- `ageoa` is deprecated with no runtime fallback.
- canonical published identities must be `sciona.*`.
- `../ageo-atoms` becomes a migration source only, not an active provider.
- legacy names may be rewritten during migration; they do not need permanent
  alias rows in Supabase.
- legacy smashed-together symbols should be normalized when they are touched for
  migration.

## Implementation Work

- Add one canonical deprecation design note in matcher docs.
- Document canonical package ownership for each current family:
  - shared/core -> `sciona-atoms`
  - signal -> `sciona-atoms-signal`
  - bio -> `sciona-atoms-bio`
  - fintech -> `sciona-atoms-fintech`
  - physics -> `sciona-atoms-physics`
  - robotics -> `sciona-atoms-robotics`
  - ml -> `sciona-atoms-ml`
- Define naming policy for legacy generated wrappers:
  - prefer snake_case or explicit descriptive names
  - forbid new collapsed one-word function names like `computekurtosissqi`
- Define success queries for Supabase and manifest validation.

## Files Likely Touched

- matcher docs only

## Exit Criteria

- one written policy exists and is linked from the migration parent plan
- later phases can assume no alias/fallback behavior

