# Expand Built-in Primitives

Add domain-general `AlgorithmicPrimitive` entries to the hard-coded catalog.
Use this when the primitives have no backing implementation — they describe
algorithmic operations the architect should recognize during CDG decomposition.

---

## Where to add

File: `ageom/architect/catalog.py`

Follow the pattern of `_BAYESIAN_PRIMITIVES` and `_SIGNAL_FILTER_PRIMITIVES`.

1. Define a module-level list:

```python
_MY_DOMAIN_PRIMITIVES: list[tuple[AlgorithmicPrimitive, list[str]]] = [
    (
        AlgorithmicPrimitive(
            name="my_primitive_name",          # snake_case, verb-first
            source="ageom-builtins",
            category=ConceptType.SOME_CATEGORY,
            description="One-sentence description, present tense, no subject.",
            inputs=[IOSpec(name="x", type_desc="ndarray", constraints="...")],
            outputs=[IOSpec(name="y", type_desc="ndarray")],
            type_signature="ndarray -> ndarray",
        ),
        ["alias one", "alias two"],  # CDG node name variants
    ),
]
```

2. Register in `seed_builtin_primitives()`:

```python
for prim, aliases in _MY_DOMAIN_PRIMITIVES:
    if catalog.get(prim.name) is None:
        catalog.add(prim)
    for alias in aliases:
        catalog.add_alias(alias, prim.name)
```

## Primitive schema

```python
class AlgorithmicPrimitive(BaseModel):
    name: str                  # unique snake_case, verb-first
    source: str                # "ageom-builtins" for built-ins
    category: ConceptType      # enum from ageom/architect/models.py
    description: str           # one sentence, present tense, no subject
    inputs: list[IOSpec]       # at least one
    outputs: list[IOSpec]      # at least one
    type_signature: str = ""   # "ndarray -> ndarray -> float -> ndarray x ndarray"
```

`IOSpec`: `name`, `type_desc`, `constraints` (optional), `required` (default True),
`default_value_repr` (e.g. `"5"`, `"'median'"` for optional params).

## Categories

See `ConceptType` enum in `ageom/architect/models.py`.  Key values:
`SORTING`, `SEARCHING`, `GRAPH_TRAVERSAL`, `GRAPH_OPTIMIZATION`,
`SIGNAL_FILTER`, `SIGNAL_TRANSFORM`, `SAMPLER`, `MCMC_KERNEL`,
`CONJUGATE_UPDATE`, `MESSAGE_PASSING`, `DYNAMIC_PROGRAMMING`,
`ARITHMETIC`, `DATA_ASSEMBLY`, `CUSTOM`.

Use `CUSTOM` if nothing fits.  Do not add new enum values unless >3
primitives need it.

## Aliases

CDG node names are LLM-generated and vary.  A primitive named
`compute_forward_transform` might need to match "forward transform", "fft",
"forward fft", "dct", "stft analysis".  Include all plausible variants.
Matched case-insensitively with spaces normalized to underscores.

## Keyword inference

If your domain has distinctive keywords, add them to `_KEYWORD_TYPES` in
`ageom/architect/source_catalog.py` so AST-fallback inference works for
source-registry atoms in the same domain.

## Before adding

1. Read all existing primitive lists in `catalog.py` to avoid duplicates.
2. The catalog has embedding-based dedup (cosine > 0.85 + structural gate).
   Two primitives describing the same operation will be merged — write one
   rich entry with many aliases rather than two thin entries.

## Verify

```bash
python -m pytest tests/test_catalog.py tests/test_source_catalog.py -v
```
