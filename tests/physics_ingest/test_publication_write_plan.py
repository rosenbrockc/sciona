from __future__ import annotations

import pytest

from sciona.physics_ingest.write_plan import (
    build_publication_write_plan,
    merge_publication_insert_rows,
)


def test_write_plan_orders_publication_batches_with_conflict_metadata() -> None:
    rows = {
        "artifact_symbolic_variables": [{"variable_id": "var-1"}],
        "artifact_relationships": [{"relationship_id": "rel-1"}],
        "physics_equation_candidates": [{"candidate_id": "cand-1"}],
        "artifact_validity_bounds": [{"bound_id": "bound-1"}],
        "artifact_symbolic_expressions": [{"expression_id": "expr-1"}],
        "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
        "artifact_cdg_edges": [{"version_id": "ver-1", "source_id": "a"}],
        "artifact_cdg_bindings": [{"version_id": "ver-1", "node_id": "a"}],
        "artifact_cdg_nodes": [{"version_id": "ver-1", "node_id": "a"}],
    }

    plan = build_publication_write_plan(rows)

    assert [batch.table for batch in plan.batches] == [
        "physics_ingest_snapshots",
        "physics_equation_candidates",
        "artifact_symbolic_expressions",
        "artifact_symbolic_variables",
        "artifact_validity_bounds",
        "artifact_relationships",
        "artifact_cdg_nodes",
        "artifact_cdg_edges",
        "artifact_cdg_bindings",
    ]
    assert {
        batch.table: batch.conflict_keys for batch in plan.batches
    } == {
        "physics_ingest_snapshots": ("snapshot_id",),
        "physics_equation_candidates": ("candidate_id",),
        "artifact_symbolic_expressions": ("expression_id",),
        "artifact_symbolic_variables": ("variable_id",),
        "artifact_validity_bounds": ("bound_id",),
        "artifact_relationships": ("relationship_id",),
        "artifact_cdg_nodes": ("version_id", "node_id"),
        "artifact_cdg_edges": (
            "version_id",
            "source_id",
            "target_id",
            "output_name",
            "input_name",
        ),
        "artifact_cdg_bindings": ("version_id", "node_id", "bound_artifact_fqdn"),
    }


def test_write_plan_copies_rows_and_summarizes_counts() -> None:
    original_row = {"snapshot_id": "snap-1", "payload": {"source": "fixture"}}
    plan = build_publication_write_plan(
        {
            "physics_ingest_snapshots": [original_row],
            "artifact_symbolic_expressions": [],
        }
    )

    original_row["snapshot_id"] = "changed"

    assert plan.to_insert_rows() == {
        "physics_ingest_snapshots": [
            {"snapshot_id": "snap-1", "payload": {"source": "fixture"}}
        ]
    }
    assert plan.audit_summary.to_dict() == {
        "input_row_counts": {
            "artifact_symbolic_expressions": 0,
            "physics_ingest_snapshots": 1,
        },
        "planned_row_counts": {"physics_ingest_snapshots": 1},
        "table_order": ["physics_ingest_snapshots"],
        "batch_count": 1,
        "total_row_count": 1,
    }


def test_write_plan_places_unknown_tables_after_known_tables() -> None:
    plan = build_publication_write_plan(
        {
            "z_extra_table": [{"id": "z"}],
            "physics_ingest_snapshots": [{"snapshot_id": "snap-1"}],
            "a_extra_table": [{"id": "a"}],
        }
    )

    assert [batch.table for batch in plan.batches] == [
        "physics_ingest_snapshots",
        "a_extra_table",
        "z_extra_table",
    ]
    assert plan.batches_by_table()["a_extra_table"].conflict_keys == ()


def test_merge_publication_insert_rows_concatenates_and_copies_rows() -> None:
    source_row = {"relationship_id": "rel-1"}
    merged = merge_publication_insert_rows(
        {"artifact_relationships": [source_row]},
        {
            "artifact_relationships": [{"relationship_id": "rel-2"}],
            "artifact_cdg_nodes": [{"version_id": "ver-1", "node_id": "step-1"}],
        },
    )

    source_row["relationship_id"] = "changed"

    assert merged == {
        "artifact_relationships": [
            {"relationship_id": "rel-1"},
            {"relationship_id": "rel-2"},
        ],
        "artifact_cdg_nodes": [{"version_id": "ver-1", "node_id": "step-1"}],
    }


@pytest.mark.parametrize(
    ("insert_rows_by_table", "message"),
    [
        ([], "insert_rows_by_table must be a mapping"),
        ({"physics_ingest_snapshots": {"snapshot_id": "snap-1"}}, "rows must be"),
        ({"physics_ingest_snapshots": [object()]}, "row 0 must be a mapping"),
        ({None: []}, "table names must be non-empty strings"),
    ],
)
def test_write_plan_validates_mapping_shapes(
    insert_rows_by_table: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_publication_write_plan(insert_rows_by_table)  # type: ignore[arg-type]
