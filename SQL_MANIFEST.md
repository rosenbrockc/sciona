# manifest.sqlite Implementation Plan

Goal: The hunter/matcher queries a local manifest.sqlite to discover atoms
across all provider repos (including repos not locally installed). The sqlite
is compiled via CI from Supabase and downloaded based on the user's
contribution/role level.

## Current State

| Component | Status |
|---|---|
| `fetch_manifest_data()` in `api/snapshot.py` | Works — queries Supabase for approved atoms, hyperparams, benchmarks, audit rollups, descriptions |
| `generate_manifest_sqlite()` in `api/snapshot.py` | Works — creates sqlite with 5 tables (atoms, hyperparams, benchmarks, audit_rollups, descriptions) |
| `sciona catalog sync` CLI in `commands/catalog_cmds.py` | Works — downloads from S3 to `~/.sciona/manifest.sqlite` |
| Hyperparams from manifest | Works — `load_hyperparams_manifest_sqlite()` in `architect/hyperparams.py` |
| Benchmarks from manifest | Works — `load_benchmarks_sqlite()` in `ecosystem/benchmarks.py` |
| Hunter atom discovery | Uses FAISS `SemanticIndex` from local sources only |
| `PrimitiveCatalog` population | From builtins + `sources.yml` live imports only |
| Role-based tiering | Not implemented |
| CI export pipeline | Not implemented |
| I/O specs in manifest | Not in schema |
| Manifest versioning | Not implemented |

---

## Step 1: Extend manifest.sqlite Schema

### 1a. Add `io_specs` table to schema

File: `sciona/api/snapshot.py`

Add after the `descriptions` CREATE TABLE (around line 404):

```sql
CREATE TABLE IF NOT EXISTS io_specs (
    atom_id    TEXT NOT NULL,
    port_name  TEXT NOT NULL,
    direction  TEXT NOT NULL,   -- 'input' | 'output'
    type_desc  TEXT NOT NULL DEFAULT '',
    constraints TEXT NOT NULL DEFAULT '',
    data_kind  TEXT NOT NULL DEFAULT '',
    required   INTEGER NOT NULL DEFAULT 1,
    default_value_repr TEXT NOT NULL DEFAULT '',
    ordinal    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (atom_id, direction, port_name)
);
```

Add a corresponding `_insert_io_spec()` helper following the pattern of
`_insert_atom()` et al.

### 1b. Fetch `io_specs` from Supabase

In `fetch_manifest_data()`, add a query for the `atom_io_specs` table
(or whatever the Supabase table is named), filtered to atoms already fetched.
Add the results to the returned dict under the key `"io_specs"`.

### 1c. Add `manifest_metadata` table

```sql
CREATE TABLE IF NOT EXISTS manifest_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Populate with rows:
- `generated_at` — ISO 8601 UTC timestamp
- `generator_version` — version of the snapshot module
- `visibility_tier` — which tier this manifest was built for (or `"all"`)
- `content_hash` — SHA-256 of the concatenated atom FQDNs for integrity check

Insert this metadata at the end of `generate_manifest_sqlite()`.

### 1d. Write `io_specs` rows in `generate_manifest_sqlite()`

Extend the function to accept and insert io_specs data. The existing
signature already accepts `atoms_or_data` as a `Mapping[str, list[dict]]`,
so io_specs can be passed as `data["io_specs"]`.

---

## Step 2: Create `seed_catalog_from_manifest_sqlite()`

### 2a. New function in `sciona/architect/source_catalog.py`

```python
def seed_catalog_from_manifest_sqlite(
    catalog: PrimitiveCatalog,
    manifest_path: Path,
    *,
    skip_locally_installed: bool = True,
    dedup_threshold: float = 0.85,
    skill_index: SkillIndex | None = None,
    report: CatalogReport | None = None,
) -> int:
```

Logic:
1. Open `manifest_path` as read-only sqlite connection.
2. Query `atoms` joined with `io_specs` and `descriptions`:
   ```sql
   SELECT a.fqdn, a.description, a.domain_tags, a.source_kind,
          a.namespace_root, a.source_repo_id, a.visibility_tier
   FROM atoms a
   WHERE a.status = 'approved'
   ```
3. For each atom, query its io_specs:
   ```sql
   SELECT port_name, direction, type_desc, constraints, data_kind,
          required, default_value_repr
   FROM io_specs
   WHERE atom_id = ? ORDER BY direction DESC, ordinal
   ```
4. Build `IOSpec` objects for inputs (direction='input') and outputs
   (direction='output').
5. Infer `ConceptType` from `domain_tags` using an existing or new
   `_infer_concept_type(domain_tags: str) -> ConceptType` helper.
6. Construct `AlgorithmicPrimitive`:
   ```python
   AlgorithmicPrimitive(
       name=fqdn,
       source=f"manifest:{source_repo_id}",
       category=concept_type,
       description=description,
       inputs=input_specs,
       outputs=output_specs,
       type_signature=_build_type_sig(input_specs, output_specs),
   )
   ```
7. If `skip_locally_installed`, check whether the atom's FQDN already
   exists in the catalog before adding.
8. Add via `catalog.add_with_dedup()` if skill_index is available,
   otherwise `catalog.add()`.
9. Return count of primitives added.

### 2b. Wire into `_load_architect_catalog()`

File: `sciona/commands/runtime_helpers.py`

After the `seed_catalog_from_sources()` call (around line 140), add:

```python
manifest_sqlite = Path.home() / ".sciona" / "manifest.sqlite"
if manifest_sqlite.is_file():
    n_manifest = seed_catalog_from_manifest_sqlite(
        catalog,
        manifest_sqlite,
        skip_locally_installed=True,
        skill_index=skill_index,
        report=report,
    )
```

This ensures locally-installed atoms (from sources.yml) take precedence,
and manifest atoms fill in the gaps for non-installed repos.

---

## Step 3: Build SemanticIndex from Manifest

### 3a. New function in `sciona/indexer/builder.py`

```python
def build_index_from_manifest_sqlite(
    manifest_path: Path,
    embedder: Embedder | None = None,
    existing_store: FAISSStore | None = None,
) -> FAISSStore:
```

Logic:
1. Open manifest.sqlite read-only.
2. Query atoms + descriptions:
   ```sql
   SELECT a.fqdn, a.description, d.content AS dejargonized_desc,
          a.domain_tags
   FROM atoms a
   LEFT JOIN descriptions d ON a.atom_id = d.atom_id
   WHERE a.status = 'approved'
   ```
3. For each atom, create a `Declaration`:
   ```python
   Declaration(
       name=fqdn,
       type_signature=type_sig,  # from io_specs
       docstring=dejargonized_desc or description,
       source_lib=f"manifest:{source_repo_id}",
       prover=Prover.PYTHON,
   )
   ```
4. Call `IndexBuilder.build_from_declarations(declarations)` or append
   to `existing_store` if merging with a local index.
5. Return the FAISSStore.

### 3b. Create a `UnionSemanticIndex` (if not already sufficient)

File: `sciona/indexer/unified.py` (check if `UnifiedSemanticIndex` already
handles multiple backends)

If the existing `UnifiedSemanticIndex` can wrap a local FAISS index and a
manifest FAISS index, use it. Otherwise create a thin wrapper that searches
both and merges results by score.

### 3c. Wire into runtime index loading

File: `sciona/commands/runtime_helpers.py` (or wherever the SemanticIndex
is constructed for the hunter)

After loading the local FAISS index, check for `~/.sciona/manifest.sqlite`
and merge manifest-derived declarations into the index:

```python
manifest_sqlite = Path.home() / ".sciona" / "manifest.sqlite"
if manifest_sqlite.is_file():
    manifest_store = build_index_from_manifest_sqlite(
        manifest_sqlite,
        embedder=embedder,
    )
    index = UnifiedSemanticIndex([local_index, manifest_index])
```

---

## Step 4: Add Role-Based Tiering to Export

### 4a. Parameterize `fetch_manifest_data()`

Add a `visibility_tiers` parameter:

```python
async def fetch_manifest_data(
    base_url: str,
    access_token: str,
    *,
    visibility_tiers: list[str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, list[dict[str, Any]]]:
```

When `visibility_tiers` is set, add filter:
```python
filters["visibility_tier"] = f"in.({','.join(visibility_tiers)})"
```

### 4b. Define tier levels

Create a constant in `api/snapshot.py`:

```python
MANIFEST_TIERS: dict[str, list[str]] = {
    "public": ["general"],
    "contributor": ["general", "contributor"],
    "researcher": ["general", "contributor", "researcher"],
    "enterprise": ["general", "contributor", "researcher", "enterprise"],
}
```

### 4c. Multi-manifest export function

```python
async def export_tiered_manifests(
    base_url: str,
    access_token: str,
    output_dir: Path,
) -> dict[str, Path]:
```

For each tier in `MANIFEST_TIERS`:
1. Call `fetch_manifest_data(..., visibility_tiers=tiers)`
2. Call `generate_manifest_sqlite(data, output_path=output_dir / f"manifest-{tier_name}.sqlite")`
3. Write `manifest_metadata` with `visibility_tier=tier_name`

Return mapping of tier_name -> output_path.

### 4d. Update `sciona catalog sync` to accept tier

Add `--tier` argument to the CLI command. Default: `"public"`.

Resolve download URL as:
```
manifests/manifest-{tier}.sqlite
```

Store locally as `~/.sciona/manifest.sqlite` (overwrite regardless of tier —
the metadata table records which tier it is).

---

## Step 5: CI Export Workflow (Coded but Not Enabled)

### 5a. Export script

File: `scripts/export_manifest.py`

```python
"""Export manifest.sqlite from Supabase and optionally upload to S3.

Usage:
    python scripts/export_manifest.py --output-dir /tmp/manifests
    python scripts/export_manifest.py --output-dir /tmp/manifests --upload
"""
```

Steps:
1. Read `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` from env.
2. Call `export_tiered_manifests()` for all tiers.
3. If `--upload`, upload each file to S3:
   `s3://{SCIONA_CATALOG_BUCKET}/manifests/manifest-{tier}.sqlite`
4. Also upload a `manifests/latest.json` with generation timestamp and
   content hashes per tier.

### 5b. GitHub Actions workflow (disabled)

File: `.github/workflows/export-manifest.yml`

```yaml
name: Export Manifest SQLite
# Disabled until AWS account is provisioned and Supabase keys are configured.
# To enable: uncomment the 'on:' triggers below.
#
# on:
#   schedule:
#     - cron: '0 4 * * *'   # daily at 04:00 UTC
#   workflow_dispatch:

on: workflow_dispatch   # manual-only for now

jobs:
  export:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install -e ".[manifest]"
      - run: python scripts/export_manifest.py --output-dir /tmp/manifests --upload
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
          SCIONA_CATALOG_BUCKET: ${{ secrets.SCIONA_CATALOG_BUCKET }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: us-east-1
```

The workflow is `workflow_dispatch` only (manual trigger). The scheduled
cron and push triggers are commented out. Enable them after:
1. AWS S3 bucket is provisioned
2. Supabase service key is stored in GitHub Secrets
3. The export script has been verified locally
4. Role tiers are populated in the Supabase `atoms.visibility_tier` column

---

## Step 6: Manifest Integrity and Freshness

### 6a. Validate on load

Add to `load_hyperparams_manifest_sqlite()` and `load_benchmarks_sqlite()`:

```python
def _check_manifest_freshness(conn: sqlite3.Connection, max_age_days: int = 30) -> None:
    try:
        row = conn.execute(
            "SELECT value FROM manifest_metadata WHERE key = 'generated_at'"
        ).fetchone()
    except sqlite3.OperationalError:
        return  # legacy manifest without metadata table
    if row is None:
        return
    generated = datetime.fromisoformat(row[0])
    age = datetime.now(timezone.utc) - generated
    if age.days > max_age_days:
        warnings.warn(
            f"manifest.sqlite is {age.days} days old. "
            f"Run 'sciona catalog sync' to update.",
            stacklevel=2,
        )
```

Call this at the top of every manifest reader.

### 6b. Content hash validation

In `generate_manifest_sqlite()`, after inserting all rows, compute:
```python
fqdns = [row[0] for row in conn.execute("SELECT fqdn FROM atoms ORDER BY fqdn")]
content_hash = hashlib.sha256("\n".join(fqdns).encode()).hexdigest()
conn.execute(
    "INSERT INTO manifest_metadata (key, value) VALUES ('content_hash', ?)",
    (content_hash,),
)
```

On load, optionally verify the hash matches.

---

## Implementation Order

| Phase | Steps | Depends On | Deliverables |
|---|---|---|---|
| **A** | 1a, 1b, 1c, 1d | Nothing | Extended schema with io_specs + metadata |
| **B** | 2a, 2b | Phase A | `seed_catalog_from_manifest_sqlite()`, wired into runtime |
| **C** | 3a, 3b, 3c | Phase A | Manifest-backed SemanticIndex for hunter |
| **D** | 4a, 4b, 4c, 4d | Phase A | Role-based tiering in fetch + export + sync CLI |
| **E** | 5a, 5b | Phase D | Export script + disabled GitHub Actions workflow |
| **F** | 6a, 6b | Phase A | Freshness warnings + integrity checks |

Phases A-C are the critical path for "hunter queries manifest.sqlite".
Phase D-E are the distribution pipeline.
Phase F is hardening.

## Detailed Phase Docs

The phase summary above now has worker-ready implementation docs under
`docs/plans/`:

- [SQL_MANIFEST_IMPLEMENTATION_PLAN.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_IMPLEMENTATION_PLAN.md)
- [SQL_MANIFEST_PHASE_A_SCHEMA_AND_METADATA.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_A_SCHEMA_AND_METADATA.md)
- [SQL_MANIFEST_PHASE_B_MANIFEST_CATALOG_SEEDING.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_B_MANIFEST_CATALOG_SEEDING.md)
- [SQL_MANIFEST_PHASE_C_MANIFEST_SEMANTIC_INDEX.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_C_MANIFEST_SEMANTIC_INDEX.md)
- [SQL_MANIFEST_PHASE_D_TIERED_EXPORT_AND_SYNC.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_D_TIERED_EXPORT_AND_SYNC.md)
- [SQL_MANIFEST_PHASE_E_CI_EXPORT_WORKFLOW.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_E_CI_EXPORT_WORKFLOW.md)
- [SQL_MANIFEST_PHASE_F_INTEGRITY_AND_FRESHNESS.md](/Users/conrad/personal/sciona-matcher/docs/plans/SQL_MANIFEST_PHASE_F_INTEGRITY_AND_FRESHNESS.md)

---

## Files Modified or Created

| File | Action |
|---|---|
| `sciona/api/snapshot.py` | Modify — add io_specs table, metadata table, tiering param |
| `sciona/architect/source_catalog.py` | Modify — add `seed_catalog_from_manifest_sqlite()` |
| `sciona/commands/runtime_helpers.py` | Modify — wire manifest catalog + index into runtime |
| `sciona/indexer/builder.py` | Modify — add `build_index_from_manifest_sqlite()` |
| `sciona/commands/catalog_cmds.py` | Modify — add `--tier` argument to sync |
| `sciona/architect/hyperparams.py` | Modify — add freshness check |
| `sciona/ecosystem/benchmarks.py` | Modify — add freshness check |
| `scripts/export_manifest.py` | Create — tiered export + optional S3 upload |
| `.github/workflows/export-manifest.yml` | Create — disabled CI workflow |
