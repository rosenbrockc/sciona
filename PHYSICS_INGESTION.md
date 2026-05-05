# Physics Ingestion Runbook

This file captures the current operational path for ingesting external physics
equation sources into the symbolic ingestion tables.

## Local Supabase

Run ingestion against the Sciona Supabase stack from `../sciona-infra`.

```bash
cd ../sciona-infra
SUPABASE_GITHUB_CLIENT_ID=dummy SUPABASE_GITHUB_CLIENT_SECRET=dummy supabase start
supabase db reset --local --yes
```

Set the local PostgREST credentials in the matcher shell. Use the service role
key printed by `supabase start` or `supabase status`.

```bash
export SUPABASE_URL=http://127.0.0.1:54321
export SUPABASE_SERVICE_ROLE_KEY=<local-service-role-key>
```

If another local Supabase stack is already bound to the ports, stop it before
starting `../sciona-infra`.

## Wikidata Equation Query

The current Wikidata candidate source is physics scoped. It starts from
Wikidata items that carry equation-like math and are classified under either
`Physics equations` or `physical law`, instead of importing every item with a
formula property. This keeps the first backfill focused while still preserving
raw source evidence for later expansion.

The query builder and response parser live in
`sciona.physics_ingest.sources.wikidata`. To refresh a candidate list, use the
retrieval adapter or call the builder directly:

```bash
python - <<'PY'
from pathlib import Path
import json

from sciona.physics_ingest.sources.wikidata import (
    build_physics_ingestion_candidate_query,
    build_wave0_candidate_records,
    execute_sparql_query,
)

out = Path("output/physics_wikidata_physics_candidate_list_manual")
out.mkdir(parents=True, exist_ok=True)
query = build_physics_ingestion_candidate_query(limit=50)
response = execute_sparql_query(query)
records = build_wave0_candidate_records(response)

(out / "query.sparql").write_text(query, encoding="utf-8")
(out / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
(out / "wave0_candidate_records.json").write_text(
    json.dumps(records, indent=2),
    encoding="utf-8",
)
print(f"wrote {len(records)} candidates to {out}")
PY
```

The last reviewed candidate list is:

```text
output/physics_wikidata_physics_candidate_list_20260505T001957Z/
```

It contains 23 distinct physics equation candidates from 50 live source
bindings.

## Wikidata Upsert

Build and optionally apply the write plan with:

```bash
python scripts/physics_ingest_wikidata_candidates.py \
  --candidate-dir output/physics_wikidata_physics_candidate_list_20260505T001957Z \
  --out-dir output/physics_wikidata_physics_ingestion_local_apply_20260505T001957Z \
  --apply
```

The runner creates:

- `physics_ingest_snapshots`: one deterministic source snapshot row.
- `physics_equation_candidates`: one deterministic candidate row per Wikidata
  equation candidate.

Both tables are written with upsert mode using the conflict metadata in
`sciona.physics_ingest.write_plan.CONFLICT_KEYS_BY_TABLE`.

Generated artifacts include `source_bundle.json`, `planned_source_bundle.json`,
`write_plan.json`, `insert_rows_by_table.json`, `backfill_report.json`,
`apply_report.json`, `summary.json`, and `ingestion_template_learnings.md`.

After applying, verify the local database with:

```bash
python - <<'PY'
import psycopg

run_snapshot = "d66a5026-a727-5223-954d-561c0b90baa8"
conn = psycopg.connect("postgresql://postgres:postgres@127.0.0.1:54322/postgres")
with conn, conn.cursor() as cur:
    cur.execute(
        "select count(*) from physics_ingest_snapshots where snapshot_id = %s",
        (run_snapshot,),
    )
    print("snapshot_rows_for_run", cur.fetchone()[0])
    cur.execute(
        "select count(*) from physics_equation_candidates where snapshot_id = %s",
        (run_snapshot,),
    )
    print("candidate_rows_for_run", cur.fetchone()[0])
PY
```

The first local apply on May 5, 2026 inserted/upserted one snapshot and 23
candidate rows.

## PDG First Wave

The PDG path starts from already retrieved Physics Derivation Graph payloads.
The adapter intentionally does not fetch PDG content; it parses source payloads
into equation nodes and premise-to-conclusion inference edges.

Run the first local wave through source candidates, symbolic expression rows,
PDG relationship rows, and CDG rows with:

```bash
python scripts/physics_ingest_pdg_first_wave.py --apply
```

By default the runner uses:

```text
tests/physics_ingest/fixtures/pdg_payloads/solve_substitute_chain.pdg.json
```

The write includes:

- `physics_ingest_snapshots` and `physics_equation_candidates` for the PDG
  payload and its equation nodes.
- minimal `artifacts` and `artifact_versions` for each equation expression.
- `artifact_symbolic_expressions` for the equation nodes. These are preserved as
  symbolic equation rows even when full SymPy normalization is still pending.
- `artifact_relationships` for PDG derivation edges.
- `artifacts` and `artifact_versions` for the generated PDG-derived CDG.
- `artifact_cdg_nodes`, `artifact_cdg_edges`, and `artifact_cdg_bindings` for the
  derivation graph.

Catalog projection rows are intentionally not applied in this wave because the
current local schema exposes `catalog_symbolic_artifacts` as a read-only view.
The CDG itself is still materialized in the unified artifact and CDG tables.

## PDG Remote Inventory

Before ingesting the remote PDG corpus, generate an immutable inventory of the
upstream repository tree and classify files/derivations into staged ingestion
waves. The inventory runner is conservative by default: it uses the GitHub API,
sends an explicit `User-Agent`, spaces requests, and stops optional fetches when
the remaining quota reaches the configured reserve.

```bash
python scripts/physics_inventory_pdg_remote.py \
  --repo allofphysicsgraph/ui_v8_website_flask_neo4j \
  --ref gh-pages \
  --license-expression CC-BY-4.0 \
  --fetch-core-files \
  --min-delay-seconds 2 \
  --max-requests 8 \
  --rate-limit-reserve 10
```

The runner writes:

- `pdg_remote_inventory_manifest.json`: complete inventory envelope.
- `tree_inventory.json`: repository file roles and raw-ingestion candidates.
- `core_file_analysis.json`: parsed Cypher node/relationship counts and rule
  classifications when core files are fetched.
- `pdg_derivations_inventory.json`: derivation-level rows with step counts,
  input/output expression counts, operation-kind counts, readiness scores, and
  recommended waves.
- `ingestion_waves.json` and `PDG_REMOTE_INGESTION_WAVES.md`: actionable wave
  plan for raw preservation and later symbolic/CDG promotion.
- `rate_limit_report.json`: request URLs, statuses, durations, and GitHub
  rate-limit headers.

Selection policy:

- preserve every license-allowed PDG resource as raw source evidence,
- promote first derivations with graph-complete steps, input/output
  expressions, and recognized operation kinds,
- keep derivations with weaker structure as relationship candidates or
  metadata-only candidates rather than dropping them,
- defer notebooks, web app support, Docker files, and test artifacts unless
  they carry provenance needed by a source snapshot.

## PDG Remote Promotion Wave

After inventory, promote a small representative set of remote derivations
through source candidates, symbolic expression rows, relationship rows, and
one CDG artifact per derivation.

```bash
python scripts/physics_ingest_pdg_remote_wave.py \
  --repo allofphysicsgraph/ui_v8_website_flask_neo4j \
  --ref gh-pages \
  --derivation-id 000008 \
  --derivation-id 000009 \
  --derivation-id 000018 \
  --derivation-id 129143 \
  --derivation-id 000004 \
  --min-delay-seconds 2 \
  --max-requests 7 \
  --rate-limit-reserve 10 \
  --apply
```

The runner fetches only repository metadata and the six core Cypher graph files:
`deriv.cypher`, `infrules.cypher`, `operators.cypher`, `steps.cypher`,
`expr_and_feed.cypher`, and `symbols.cypher`. It converts selected derivation
steps into premise-to-conclusion PDG edges and preserves feed values in edge
binding metadata. Expression rows remain `needs_human`/`unknown` until symbolic
normalization and dimensional validation are explicitly completed.
