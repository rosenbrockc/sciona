# Expand Catalog via Ingestion

Use `ageom ingest` to automatically decompose existing source code into
atoms and register them in the catalog.  This is the preferred method when
you have a concrete implementation to work from.

---

## Full reference

Read these files in `../ageo-atoms/` before proceeding:

| File | What it covers |
|------|----------------|
| `INGEST_PROMPT.md` | Complete `ageom ingest` command reference, all languages, recursive decomposition, output validation, verification checklist |
| `INGESTION.md` | Atom authoring spec: signatures, contracts, witnesses, CDG schema, tests |
| `INTEREST.md` | Curated interesting algorithms to ingest, organized by source repo |
| `PENDING.md` | Algorithms already identified for future ingestion |

## Workflow

### 1. Pick a target

Check `../ageo-atoms/INTEREST.md` and `../ageo-atoms/PENDING.md` for
pre-identified targets.  Check existing packages in `../ageo-atoms/ageoa/`
to avoid duplicating work.

### 2. Run the ingester

```bash
# LLM-assisted (default)
ageom ingest path/to/source.py --class ClassName \
    --output ../ageo-atoms/ageoa/mydomain

# Deterministic (no LLM)
ageom ingest path/to/source.py --class ClassName \
    --procedural --output ../ageo-atoms/ageoa/mydomain

# With monitoring for large classes
ageom ingest path/to/source.py --class ClassName \
    --output ../ageo-atoms/ageoa/mydomain --monitor --trace
```

Supported languages: `.py`, `.rs`, `.jl`, `.cpp`, `.h`, `.hpp` (auto-detected).

Output directly into `../ageo-atoms/ageoa/` so atoms are available via the
existing `sources.yml` entry without additional configuration.

### 3. Validate

Both `mypy passed` and `Ghost sim passed` must be `True`.  If either fails,
follow Task 2 in `../ageo-atoms/INGEST_PROMPT.md`.

### 4. Write tests

5 categories per atom (see `../ageo-atoms/INGESTION.md` section 12):
positive path, precondition violations, postcondition verification, edge
cases, upstream parity.

### 5. Export

Ensure all atoms are imported in `__init__.py` and reachable from
`ageoa/__init__.py`.

## Recursive decomposition

For complex classes:

```bash
export AGEOM_INGESTER_MAX_DEPTH=3
ageom ingest path/to/source.py --class LargeClass \
    --output ../ageo-atoms/ageoa/mydomain
```

## Batch ingestion

```bash
for cls in ClassA ClassB ClassC; do
    ageom ingest path/to/source.py --class "$cls" \
        --output "../ageo-atoms/ageoa/${cls,,}"
done
```

## Bulk from curated repos

```bash
ageom skill ingest --source clrs --path /tmp/clrs
ageom skill ingest --source coq100 --path /tmp/coq100
```

Writes `catalog_*.json` to the skill index directory.

## How ingested atoms reach the catalog

`sources.yml` declares `ageo-atoms` pointing at `../ageo-atoms` with
package `ageoa`.  `seed_catalog_from_sources` imports the package
(triggering `@register_atom`), scans `**/*cdg*.json`, and derives
`AlgorithmicPrimitive` entries with full de-duplication.
