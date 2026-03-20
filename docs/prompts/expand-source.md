# Expand Catalog via External Source Registry

Add hand-authored `@register_atom` atoms to the `ageo-atoms` package
(or a new external source).  Use this when you are writing atoms manually
following the full INGESTION.md specification.

---

## Full reference

Read `../ageo-atoms/INGESTION.md` — it is the ground truth for atom quality
and covers signatures, contracts, witnesses, CDG schema, tests, and the
complete checklist.

## Existing packages

Check `../ageo-atoms/ageoa/` for 20+ existing domain packages before
creating a new one.  See the list in the router prompt.

## Creating a new domain package

```
../ageo-atoms/ageoa/mydomain/
    __init__.py        # re-exports atoms, __all__
    atoms.py           # @register_atom decorated functions
    witnesses.py       # ghost witnesses using abstract types
    *cdg*.json         # optional CDG with atomic node metadata
```

Follow `INGESTION.md` for:
- Signature rules (concrete types, no `Any`/`*args`/`**kwargs`)
- Contract ordering (isinstance innermost, `@register_atom` outermost)
- At least one `@require` and `@ensure` per atom
- Ghost witnesses using abstract types only
- Google-style docstrings with `Args:` and `Returns:`
- Tests (5 categories per atom)
- Module exports reaching `ageoa/__init__.py`

## How sources are loaded

`sources.yml` declares:

```yaml
sources:
  - name: ageo-atoms
    path: ../ageo-atoms
    package: ageoa
```

`seed_catalog_from_sources` imports `ageoa`, triggering `@register_atom`
decorators, scans `**/*cdg*.json` for metadata, and derives
`AlgorithmicPrimitive` entries with de-duplication.

## Adding a non-ageo-atoms source

Add to `sources.yml`:

```yaml
  - name: my-source
    path: ../my-source-repo
    package: mysourcepkg
```

The package needs `ghost/registry.py` with a `REGISTRY` dict and
`register_atom` decorator (same pattern as `ageoa`).

## Verify

```bash
python -m pytest tests/test_source_catalog.py -v
python -m sciona catalog-gaps --cdg path/to/some_cdg.json
```
