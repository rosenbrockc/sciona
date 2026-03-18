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

## Current Example

The existing `signal_event_rate` family swaps:
- `compute_event_rate`
- `compute_event_rate_smoothed`

That family is reusable for signal-to-event-to-rate pipelines and is not specific to ECG, even though ECG HR is the current benchmark using it.
