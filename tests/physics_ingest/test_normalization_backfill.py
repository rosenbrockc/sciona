from copy import deepcopy
import os
from uuid import uuid4

import pytest

from sciona.physics_ingest.normalization_backfill import prepare_normalization, apply_normalization


def raw_row():
    return {
        "expression_id": "00000000-0000-0000-0000-000000000001",
        "artifact_id": "00000000-0000-0000-0000-000000000002",
        "version_id": "00000000-0000-0000-0000-000000000003",
        "raw_formula": "F = m*a", "raw_formula_format": "plain_text",
        "parse_status": "raw_imported", "review_status": "needs_human",
        "validation_status": "unknown", "evidence_json": {"source_review": {"pending": True}},
    }


def test_backfill_adds_parse_evidence_without_fabricating_dimensions_or_review():
    row = raw_row()
    before = deepcopy(row)
    result = prepare_normalization(row)
    assert row == before
    assert result["parse_status"] == "normalized"
    assert result["evidence_json"]["parse_roundtrip"]["status"] == "passed"
    assert result["evidence_json"]["source_review"] == {"pending": True}
    assert "missing_required_dimension" in result["diagnostic_codes"]
    assert "review_status" not in result
    assert "validation_status" not in result
    assert "dimensional_hash" not in result


@pytest.mark.parametrize("field,value", [("review_status", "human_reviewed"), ("parse_status", "normalized")])
def test_backfill_refuses_existing_review_or_processing(field, value):
    row = raw_row()
    row[field] = value
    with pytest.raises(ValueError, match="unprocessed"):
        prepare_normalization(row)


def test_backfill_refuses_evidence_overwrite():
    row = raw_row()
    row["evidence_json"]["parse_roundtrip"] = {"status": "failed", "reviewed": True}
    with pytest.raises(ValueError, match="overwrite"):
        prepare_normalization(row)


def test_incomplete_latex_is_not_accepted_as_a_valid_prefix():
    row = raw_row()
    row.update(raw_formula="x -", raw_formula_format="latex")
    result = prepare_normalization(row)
    assert result["parse_status"] == "parse_failed"
    assert result["evidence_json"]["parse_roundtrip"]["status"] != "passed"


def test_reprocessing_preserves_prior_evidence_without_review_upgrade():
    row = raw_row()
    first = prepare_normalization(row)
    processed = {**row, "parse_status": first["parse_status"], "evidence_json": first["evidence_json"]}
    second = prepare_normalization(processed, reprocess_owned=True)
    assert second["original_parse_status"] == "normalized"
    assert second["evidence_json"]["normalization_history"][0]["evidence"] == first["evidence_json"]
    assert second["evidence_json"]["source_review"] == {"pending": True}
    assert "review_status" not in second
    with pytest.raises(ValueError, match="unprocessed"):
        prepare_normalization({**processed, "review_status": "human_reviewed"}, reprocess_owned=True)


def test_database_normalization_preserves_review_and_rolls_back_on_drift():
    url = os.environ.get("SCIONA_PROMOTION_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires explicit promotion integration-test database")
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    db = psycopg.connect(url, row_factory=dict_row)
    rows = []
    try:
        for _ in range(2):
            row = raw_row()
            for field in ["expression_id", "artifact_id", "version_id"]:
                row[field] = str(uuid4())
            db.execute("INSERT INTO artifacts(artifact_id,artifact_kind,fqdn) VALUES(%s,'atom',%s)", (row["artifact_id"], "synthetic.normalization." + uuid4().hex))
            db.execute("INSERT INTO artifact_versions(version_id,artifact_id,content_hash,semver) VALUES(%s,%s,%s,'0')", (row["version_id"], row["artifact_id"], "synthetic-" + uuid4().hex))
            db.execute(
                "INSERT INTO artifact_symbolic_expressions(expression_id,artifact_id,version_id,raw_formula,raw_formula_format,parse_status,review_status,evidence_json,expression_kind) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'equation')",
                tuple(row[k] for k in ["expression_id", "artifact_id", "version_id", "raw_formula", "raw_formula_format", "parse_status", "review_status"]) + (Jsonb(row["evidence_json"]),),
            )
            rows.append(row)
        updates = [prepare_normalization(row) for row in rows]
        db.execute("UPDATE artifact_symbolic_expressions SET evidence_json='{}' WHERE expression_id=%s", (rows[1]["expression_id"],))
        with pytest.raises(ValueError, match="batch rolled back"):
            apply_normalization(db, updates)
        assert db.execute("SELECT parse_status FROM artifact_symbolic_expressions WHERE expression_id=%s", (rows[0]["expression_id"],)).fetchone()["parse_status"] == "raw_imported"
        assert apply_normalization(db, updates[:1]) == {"normalized": 1}
        state = db.execute("SELECT review_status,validation_status,raw_formula FROM artifact_symbolic_expressions WHERE expression_id=%s", (rows[0]["expression_id"],)).fetchone()
        assert state == {"review_status": "needs_human", "validation_status": "unknown", "raw_formula": rows[0]["raw_formula"]}
    finally:
        db.rollback()
        db.close()
