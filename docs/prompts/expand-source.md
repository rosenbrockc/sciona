# Expand Catalog via External Source Registry

Add hand-authored `@register_atom` atoms to the `sciona.atoms` package
(or a new external source).  Use this when you are writing atoms manually
following the full INGESTION.md specification.

---

## Full reference

Read `../sciona-atoms/INGESTION.md` — it is the ground truth for atom quality
and covers signatures, contracts, witnesses, CDG schema, tests, and the
complete checklist.

## Existing packages

Check `../sciona-atoms/sciona/atoms/` for existing domain packages before
creating a new one.  See the list in the router prompt.

## Creating a new domain package

```
../sciona-atoms/sciona/atoms/mydomain/
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
- Module exports reaching `sciona/atoms/__init__.py`

## How sources are loaded

`sources.yml` declares:

```yaml
sources:
  - name: sciona-atoms
    path: ../sciona-atoms
    package: sciona.atoms
```

`seed_catalog_from_sources` imports `sciona.atoms`, triggering `@register_atom`
decorators, scans `**/*cdg*.json` for metadata, and derives
`AlgorithmicPrimitive` entries with de-duplication.

## Adding a non-sciona-atoms source

Add to `sources.yml`:

```yaml
  - name: my-source
    path: ../my-source-repo
    package: mysourcepkg
```

The package needs `ghost/registry.py` with a `REGISTRY` dict and
`register_atom` decorator (same pattern as `sciona.atoms`).

## Verify

```bash
python -m pytest tests/test_source_catalog.py -v
python -m sciona catalog-gaps --cdg path/to/some_cdg.json
```
