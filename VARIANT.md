# Variant Families

This repo supports family-specific Principal mutations through a small plugin-style interface in [ageom/principal/variant_mutation.py](/Users/conrad/personal/ageo-matcher/ageom/principal/variant_mutation.py).

## Goal

Use in-place variant swaps only when they generalize to a reusable algorithm family.

Good examples:
- A signal-event-rate scaffold with multiple interchangeable rate estimators.
- A filter-design scaffold with multiple stable coefficient construction variants.
- A probabilistic conjugate-update scaffold with alternative sufficient-statistics implementations.

Bad examples:
- A one-off mutation that exists only to improve a single benchmark case.
- Dataset-specific threshold tuning baked into Principal.
- Variants keyed on a particular goal string like `ECG HR`.

## Interface

A family implements:
- `matches(cdg) -> bool`
- `mutate(cdg, bottleneck_name=...) -> VariantMutationResult`

`VariantMutationResult` carries:
- `cdg`
- `applied`
- `family`
- `variant_name`
- `allow_redecompose`

Principal calls the first registered family whose `matches()` returns true and whose `mutate()` actually applies a change.
If a family returns `applied=False` with `allow_redecompose=False`, Principal stops instead of falling back to a generic re-decomposition.

## How To Add A Family

1. Create or identify a reusable scaffold family.
2. Put the family-specific primitive declarations in a registry module if needed.
3. Add a class in [ageom/principal/variant_mutation.py](/Users/conrad/personal/ageo-matcher/ageom/principal/variant_mutation.py) that:
   - detects the family structurally from the CDG
   - swaps only semantically equivalent node variants
   - returns `applied=False` when no safe variant exists
4. Register the family in `VARIANT_FAMILIES`.
5. Add tests covering:
   - family detection
   - successful mutation
   - no-op behavior on unrelated CDGs
   - compile/eval acceptance in the target workflow

## Rules

- Detect families from structure and primitive assignments, not benchmark names.
- Keep variants semantically compatible with the node contract.
- Do not embed dataset-specific constants in Principal mutation logic.
- Prefer runtime/library variants over ad hoc source rewriting.
- If a change requires changing the graph topology, use time-travel re-decomposition instead of an in-place variant swap.
- If the family owns the scaffold and generic re-decomposition is known to be unsafe or semantically lossy, return `allow_redecompose=False` when variants are exhausted.

## Recommended Workflow

1. Add the new primitive variant to the runtime/library layer.
2. Register the primitive and its alternatives in the family registry.
3. Add the family mutation plugin.
4. Run synthesize/export/profile on a representative task.
5. Only keep the family if it improves at least one real objective without breaking compile/export.

## Current Families

### `signal_event_rate` (curated)

Swaps between interchangeable rate estimators:
- `compute_event_rate`
- `compute_event_rate_smoothed`

Reusable for signal-to-event-to-rate pipelines. Not specific to ECG, even though ECG HR is the current benchmark using it.

### `ledger_bandit` (universal fallback)

Uses a UCB1 multi-armed bandit over an in-memory **Atom Performance Ledger** to select atom variants. Unlike curated families, this family works for any domain and does not require a hardcoded alternatives registry.

How it works:
1. After each trial's `compute_gradients` step, the Principal records every atomic node's `(slot_signature, atom_name, gradient_score)` into the ledger.
2. The **slot signature** groups structurally equivalent CDG positions by `(parent_name, concept_type, input_types, output_types)` — independent of node ID.
3. When a bottleneck node is identified, the ledger ranks all same-category primitives using **UCB1**: atoms with lower historical gradient scores (better performance) rank higher, while untried atoms get an exploration bonus.
4. If the top-ranked atom differs from the current assignment, the family swaps it in-place.

The ledger bandit fires only after curated families have been tried. It always returns `allow_redecompose=True`, so the Principal can still fall through to time-travel re-decomposition if the bandit has no useful suggestion.

Key files:
- `ageom/principal/atom_ledger.py` — `AtomLedger`, `SlotSignature`, `compute_slot_signature`
- `ageom/principal/variant_mutation.py` — `LedgerVariantFamily`

## Future Work

The following enhancements build on the ledger infrastructure but are not yet implemented.

1. **Thompson Sampling** — Replace UCB1 with Beta-posterior sampling for stochastic exploration. Same ledger, swap the ranking formula only.

2. **Contextual Bandit (LinUCB)** — Use slot features (CDG depth, fan-in/fan-out, data dimensionality) as context in a linear bandit model. Requires a small numpy/sklearn dependency for ridge regression.

3. **Atom Affinity Matrix** — Track which atom *pairs* co-occur in successful trials. Score candidate atoms partly on how well they pair with neighboring atoms already in the CDG.

4. **Cross-Goal Transfer** — Share ledger observations across semantically similar goals using goal embeddings. Requires embedding similarity computation.

5. **Persistent Ledger (Postgres)** — Store observations in a dedicated `atom_observations` table for cross-session learning. Uses existing `PostgresTelemetryStore` infrastructure.

6. **Deep Integration into `_suggest_primitive_for_spec`** — Thread the ledger into the Architect's deterministic decomposer to re-rank borderline candidates (token-overlap score 2.0–3.0) during initial decomposition, not just during time-travel updates.
