# CDG Expansion and Refinement Gap Closure Plan

## Goal

Make expansion/refinement CDGs the default way to adapt representative base
CDGs to nearby solution variants. A new base CDG should be created only when
the output contract, core topology, or inference paradigm is genuinely absent.

## Phase 1: Standardize the Common Interface

Status: complete.

- Use `ExpansionFamilyAsset` as the provider-facing schema for expansion and
  refinement operations.
- Preserve compatibility aliases for legacy `expansion_id`,
  `expansion_type`, and `prerequisite_expansions` fields.
- Surface operation metadata through `ExpansionDiagnostic` and
  `ExpansionResult.applied_assets`.
- Move obvious existing ML expansion skeletons into provider
  `data/expansions/*.json` assets.

## Phase 2: Provider Inventory and Schema Adoption

Status: in progress.

- Inventory every sibling `sciona-atoms-*` repo for existing expansion,
  refinement, rewrite, skeleton, and heuristic assets.
- Add `data/expansions/*.json` assets for obvious existing operations that
  already have runtime rules or registered atoms.
- Keep provider assets declarative; runtime rewrite behavior remains in
  matcher rule sets until a rule-builder plugin interface exists.
- Add loader tests proving all provider expansion assets are discoverable from
  `sources.yml`.

## Phase 3: Runtime Rule Coverage

Status: complete.

- For each provider expansion asset, ensure there is a matching
  `ExpansionRuleSet` rule name in `default_rule_sets()`.
- Add missing DPO or semantic rewrite rules for obvious operations already
  represented as CDGs or registered atoms.
- Prefer small insert/replace rewrites over broad topology rewrites.
- Record asset-backed provenance in applied expansion results.

## Phase 4: Operation Applicability Contracts

Status: complete.

- Tighten `trigger`, `required_runtime_keys`, boundary requirements,
  primitive requirements, and adjacency requirements.
- Make every expansion safe to skip when the host CDG does not satisfy its
  preconditions.
- Add contraindication fields if needed for operations that are easy to misuse.

## Phase 5: Expansion Retrieval

Status: complete.

- Build a retrieval index over expansion/refinement assets.
- Given a base CDG and missing techniques, retrieve candidate operations by
  family, topology, inputs/outputs, stage names, and technique descriptions.
- Return ranked operation sequences, not just individual operations.

## Phase 6: Delta Planning

Status: complete.

- Add an architect path that plans:
  `base CDG -> operation sequence -> adapted CDG`.
- Score candidates by projected technique coverage divided by intrusion cost.
- Explicitly choose between direct use, refinement, expansion, expansion pack,
  or true novel composition.

## Phase 7: Counterfactual Validation

- Update Kaggle validation to grade adapted CDGs after two or three
  expansion/refinement rounds.
- Report base-only coverage and adapted coverage side by side.
- Track which operations rescue each competition.

## Phase 8: Duplicate CDG Governance

- Before accepting a new base CDG, compare it against existing bases plus
  expansion/refinement operations.
- Reject or flag base CDGs that are mostly isomorphic to an existing base plus
  a small delta.
- Keep true novel CDGs for distinct topology or output-contract gaps.

## Phase 9: Solution-Specific Operation Mining

- Mine unmatched and partial Kaggle validation cases for reusable missing
  deltas.
- Cluster missing techniques into operation families.
- Add only reusable expansion/refinement CDGs in this phase; avoid one-off
  competition patches unless they expose a general pattern.

## Phase 10: Promotion and Manifest Closure

- Register operation helper atoms where needed.
- Reseed provider catalogs and sync skeleton/artifact bindings.
- Ensure Supabase, Memgraph, and SQLite manifest exports all see the same
  expansion/refinement inventory.
- Make operation coverage a release gate for CDG template changes.

## Immediate Acceptance Criteria

- Every sibling repo either has provider `data/expansions/*.json` assets or a
  documented empty inventory.
- `default_rule_sets()` wraps every implemented provider asset with
  `AssetBackedExpansionRuleSet`.
- Validation can distinguish `true_novel` from `base_plus_delta`.
