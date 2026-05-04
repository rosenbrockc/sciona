# Physics Ingest Publication Pipeline

This note documents the landed publication path for physics ingestion rows. It
is intentionally narrower than the full implementation plan: it covers the
current code that stages source snapshots, equation candidates, symbolic
publication rows, write plans, and caller-owned database writes.

## Current Architecture

The current pipeline is split at the storage boundary:

- `sciona.physics_ingest.ids` assigns deterministic UUIDv5 IDs to source
  snapshots and candidates without database calls.
- `sciona.physics_ingest.staging` validates raw snapshot and candidate rows
  against the Wave 0 table contract.
- `sciona.physics_ingest.publication` validates symbolic expression, variable,
  and validity-bound manifest rows after resolving artifact/version bindings.
- `sciona.physics_ingest.orchestration` combines source bundles and publication
  manifests into validated `insert_rows_by_table`.
- `sciona.physics_ingest.write_plan` orders rows by dependency and records per
  table insert/upsert modes.
- `sciona.physics_ingest.writer` applies a write plan through an injected
  `PublicationTableClient`; it does not import Supabase.
- `sciona.physics_ingest.supabase_adapter` wraps injected PostgREST-style
  clients and can preflight planned writes without importing Supabase or writing
  rows. It also exposes a high-level apply helper for injected clients so
  deployment code can share the same dry-run/write accounting path.
- `sciona.physics_ingest.pipeline` composes all steps and can either dry-run,
  stop at a side-effect-free plan, or execute through an injected client.
- `sciona.physics_ingest.sources.retrieval_plan` emits deterministic
  executor-facing request envelopes for retrieval jobs without performing
  network IO.
- `sciona.physics_ingest.sources.executor` can execute those retrieval envelopes
  through injected HTTP clients and snapshot sinks; dry runs and manual sources
  remain side-effect free.
- `sciona.physics_ingest.sources.runtime_adapters` wraps injected HTTP/session
  objects and snapshot sinks into executor-ready adapters with JSON-safe
  capability reports, preflight metadata, and normalized snapshot receipts.
- `sciona.physics_ingest.normalization` includes opt-in QUDT-assisted dimension
  resolution before symbolic normalization; unresolved or ambiguous dimensions
  stay reviewable.
- `sciona.physics_ingest.cli` provides JSON-serializable dry-run report helpers
  for decoded payloads.
- `sciona.physics_ingest.validation` provides the offline validation report used
  to check symbolic fixtures, PDG-derived CDG rows, source execution readiness,
  adapter coverage, and data-artifact seed shape without Supabase.
- `sciona.physics_ingest.backfill`, `sciona.physics_ingest.pdg_cdg`, and
  `sciona.physics_ingest.review` expose JSON-safe rollups for bulk dashboards,
  PDG/CDG publication audit, and Phase 5 trust review triage. Backfill reports
  can opt into source request-envelope, publication write preflight, and
  persistable audit/dashboard artifact manifest sections. PDG/CDG helpers also
  project derived CDGs into deterministic catalog/search rows for review before
  production catalog storage is wired.
- `sciona.physics_ingest.retrieval` provides side-effect-free symbolic
  retrieval and synthesis ranking over already-fetched catalog/document rows.
- `sciona.physics_ingest.retrieval_io` plans and executes catalog/RPC fetches
  through injected clients before handing rows to the side-effect-free rankers,
  and can build/execute planner request envelopes that preserve replay hashes,
  compiler expectations, and trust-policy blockers.

Publication table order is:

1. `physics_ingest_snapshots`
2. `physics_equation_candidates`
3. `artifact_symbolic_expressions`
4. `artifact_symbolic_variables`
5. `artifact_validity_bounds`
6. `artifact_relationships`

Unknown tables are still accepted by the write planner, but they are sorted
after the known publication tables. This keeps the core path deterministic while
leaving room for downstream extension rows.

## Source Bundle Shape

Source adapters should emit ordinary Python mappings or objects with
`snapshot_row` and `candidate_rows`. The source rows should be JSON-compatible
and should not perform database writes themselves.

```python
source_bundle = {
    "bundle_key": "fixture-bundle",
    "snapshot_row": {
        "source_system": "manual",
        "source_version": "fixture-v1",
        "adapter_name": "fixture.adapter",
        "payload_sha256": "a" * 64,
        "payload": {"record_count": 1},
    },
    "candidate_rows": [
        {
            "source_candidate_id": "fixture:eq:force",
            "source_label": "Newton second law",
            "raw_formula": "F = m a",
            "raw_formula_format": "plain_text",
            "candidate_status": "raw_imported",
            "parse_confidence": 0.5,
            "source_payload": {"fixture": True},
        }
    ],
}
```

`plan_source_bundle_ids()` copies these rows, adds a deterministic
`snapshot_id` to the snapshot row, and adds deterministic `candidate_id` values
to candidate rows. It never mutates the caller's original bundle.

Snapshot binding keys are derived from `bundle_key`, `key`, `name`,
`source_system`, `adapter_name`, and `source_uri`. Candidate IDs are scoped to
the deterministic snapshot ID plus the source candidate ID.

## Publication Manifest Shape

Publication manifests describe rows for already-resolved Sciona artifacts. The
loader resolves `artifact_key`, `local_artifact_key`, `atom_name`, or
`registry_name` through explicit artifact bindings.

```python
publication_manifest = {
    "provider": "fixture",
    "artifact_symbolic_expressions": [
        {
            "artifact_key": "local:fixture.force",
            "local_artifact_key": "local:fixture.force",
            "atom_name": "force_atom",
            "registry_name": "force_atom",
            "expression_srepr": (
                "Equality(Symbol('F'), Mul(Symbol('m'), Symbol('a')))"
            ),
            "expression_text": "Eq(F, a*m)",
        }
    ],
    "artifact_symbolic_variables": [
        {
            "artifact_key": "local:fixture.force",
            "atom_name": "force_atom",
            "symbol": "F",
            "role": "output",
        }
    ],
    "artifact_validity_bounds": [],
}

artifact_bindings = {
    "local:fixture.force": {
        "artifact_id": "20000000-0000-0000-0000-000000000001",
        "version_id": "30000000-0000-0000-0000-000000000001",
    }
}
```

Manifest loading is non-fatal for row-level issues: invalid rows produce
diagnostics, while valid rows continue into the write plan.

## Dry-Run Helpers

For CLI-style payloads, use
`build_publication_dry_run_report_from_payload(payload, include_rows=False)`.
The payload keys are:

- `source_bundles`
- `publication_manifests`
- `artifact_bindings`
- `snapshot_id_bindings`
- `table_modes`
- `plan_ids`

Example:

```python
from sciona.physics_ingest.cli import build_publication_dry_run_report_from_payload

report = build_publication_dry_run_report_from_payload(
    {
        "source_bundles": [source_bundle],
        "publication_manifests": [publication_manifest],
        "artifact_bindings": artifact_bindings,
        "table_modes": {"artifact_symbolic_expressions": "upsert"},
        "plan_ids": True,
    },
    include_rows=True,
)

assert report["report_kind"] == "physics_ingest_publication_dry_run"
assert report["dry_run"] is True
assert report["write_plan"]["batches"][0]["table"] == "physics_ingest_snapshots"
```

Use `include_rows=True` only for debugging or local audit artifacts. The default
report is compact and JSON-serializable.

For in-process use, `run_physics_publication_pipeline(..., dry_run=True)`
returns all intermediate artifacts:

```python
from sciona.physics_ingest.pipeline import run_physics_publication_pipeline

result = run_physics_publication_pipeline(
    source_bundles=[source_bundle],
    publication_manifests=[publication_manifest],
    artifact_bindings=artifact_bindings,
    table_modes={"artifact_symbolic_expressions": "upsert"},
    dry_run=True,
)

assert result.summary.has_errors is False
assert result.write_result is not None
assert result.write_result.affected_count == 0
```

## Write Plan Contract

`build_publication_write_plan(rows_by_table, table_modes=None)` returns an inert
`PublicationWritePlan`.

The contract is:

- input is a mapping of table name to iterable row mappings;
- rows are shallow-copied into ordered `PublicationWriteBatch` objects;
- known publication tables use the dependency order listed above;
- unknown tables are sorted after known tables;
- empty batches are omitted;
- `table_modes` may set a table to `insert` or `upsert`;
- conflict keys are metadata only and are not applied by the planner;
- no schema validation or database IO happens in the planner.

`PublicationWritePlan` exposes:

- `ordered_tables()` for execution order;
- `batches_by_table()` for table lookup;
- `mode_for(table)` for insert/upsert resolution;
- `to_insert_rows()` for converting back to plain row mappings;
- `audit_summary` for planned row counts and table order.

## Writer Adapter Contract

Storage adapters implement `PublicationTableClient`:

```python
from collections.abc import Mapping, Sequence
from typing import Any

class SupabasePublicationClient:
    def insert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> Any:
        ...

    def upsert(self, table: str, rows: Sequence[Mapping[str, Any]]) -> Any:
        ...
```

The writer owns only dependency-ordered dispatch and accounting:

- `PublicationWriter(client).write(plan, dry_run=True)` never calls the client.
- With `dry_run=False`, each non-empty batch is written in plan order.
- `insert` is the default mode; per-table `upsert` is supported through
  `PublicationWritePlan.from_rows(..., table_modes=...)` or
  `write_publication_rows(..., table_modes=...)`.
- On a write exception, the writer records an error diagnostic for that table
  and stops before later tables.
- Affected-row counts are inferred from integer responses, response mappings
  with `count` or `data`, response objects with `count` or `data`, or else the
  planned row count.

Example fake adapter:

```python
class FakePublicationClient:
    def __init__(self) -> None:
        self.calls = []

    def insert(self, table, rows):
        self.calls.append(("insert", table, tuple(rows)))
        return {"count": len(rows)}

    def upsert(self, table, rows):
        self.calls.append(("upsert", table, tuple(rows)))
        return {"count": len(rows)}
```

Execution example:

```python
client = FakePublicationClient()

result = run_physics_publication_pipeline(
    source_bundles=[source_bundle],
    publication_manifests=[publication_manifest],
    artifact_bindings=artifact_bindings,
    client=client,
    table_modes={"artifact_symbolic_expressions": "upsert"},
    dry_run=False,
)

assert result.summary.affected_row_count == 4
```

If `dry_run=False` and no client is provided, the pipeline stops after building
the side-effect-free write plan. This is useful for tests and review tooling
that should inspect rows without requiring credentials.

## Remaining Work

The current publication pipeline does not yet complete the full physics
ingestion roadmap. Remaining work includes:

- wire the source runtime adapter bundle into deployment code for the full
  external source set;
- wire injected production PostgREST/Supabase clients through deployment code
  using the shared apply/preflight helper;
- connect PDG-derived CDG publication and catalog projection rows to production
  storage and catalog views;
- broaden symbolic normalization coverage across the long-tail equation corpus
  and keep expanding QUDT/unit alias coverage;
- wire review queue task rows for `needs_human`, `human_reviewed`, and
  `blocked` into production queues and reviewer UX;
- add production bulk backfill orchestration and storage adapters for persisted
  coverage dashboard and replay/audit artifact manifests;
- connect the planner request-envelope boundary to the production runtime
  planner service.
