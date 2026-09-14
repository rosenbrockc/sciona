import os
import pytest
from tests.physics_ingest.test_parse_corroboration import ready_row
from tests.physics_ingest.test_pdg_evidence import symbol_bytes
from sciona.physics_ingest.pdg_dimensions import prepare_pdg_dimensions, materialize_pdg_dimensions


def test_dimensions_follow_existing_evaluation_input_policy():
    source=ready_row()
    evidence,variables=prepare_pdg_dimensions(source,symbol_bytes())
    assert evidence['status']=='passed'
    assert len(variables)==1
    assert variables[0]['variable_role']=='input'
    assert variables[0]['dimension_source']=='source'
    assert variables[0]['symbol_name']=='x'
    assert 'review_status' not in evidence


def test_stale_evidence_is_rejected():
    source=ready_row();source['sympy_srepr']="Symbol('z')"
    with pytest.raises(ValueError,match='stale'):
        prepare_pdg_dimensions(source,symbol_bytes())


def test_database_dimensions_rollback_on_conflict_and_preserve_review():
    url=os.environ.get('SCIONA_PROMOTION_TEST_DATABASE_URL')
    if not url: pytest.skip('requires explicit integration database')
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    db=psycopg.connect(url,row_factory=dict_row)
    try:
        db.execute('CREATE TEMP TABLE artifact_symbolic_expressions(expression_id uuid,version_id uuid,candidate_id uuid,sympy_srepr text,evidence_json jsonb,review_status text,validation_status text,parse_status text,dimensional_hash text)')
        db.execute('CREATE TEMP TABLE physics_equation_candidates(candidate_id uuid,snapshot_id uuid,source_payload jsonb)')
        db.execute('CREATE TEMP TABLE physics_ingest_snapshots(snapshot_id uuid,payload jsonb)')
        db.execute('CREATE TEMP TABLE artifact_symbolic_variables(variable_id uuid,expression_id uuid,symbol_name text,source_symbol text,variable_role text,dim_signature text,dimension_source text,evidence_json jsonb,ordinal integer)')
        rows=sorted([ready_row(),ready_row()],key=lambda r:r['expression_id'])
        for r in rows:
            db.execute("INSERT INTO artifact_symbolic_expressions VALUES(%s,%s,%s,%s,%s,'needs_human','unknown','normalized','')",(r['expression_id'],r['version_id'],r['candidate_id'],r['sympy_srepr'],Jsonb(r['evidence_json'])))
            db.execute('INSERT INTO physics_equation_candidates VALUES(%s,%s,%s)',(r['candidate_id'],r['snapshot_id'],Jsonb(r['source_payload'])))
            db.execute('INSERT INTO physics_ingest_snapshots VALUES(%s,%s)',(r['snapshot_id'],Jsonb(r['snapshot_payload'])))
        db.execute("UPDATE artifact_symbolic_expressions SET dimensional_hash='reviewer-evidence' WHERE expression_id=%s",(rows[1]['expression_id'],))
        with pytest.raises(ValueError,match='conflict'): materialize_pdg_dimensions(db,symbol_bytes())
        assert db.execute('SELECT count(*) AS n FROM artifact_symbolic_variables').fetchone()['n']==0
        db.execute("UPDATE artifact_symbolic_expressions SET dimensional_hash='' WHERE expression_id=%s",(rows[1]['expression_id'],))
        assert materialize_pdg_dimensions(db,symbol_bytes())=={'expressions':2,'variables':2}
        assert materialize_pdg_dimensions(db,symbol_bytes())=={'unchanged':2}
        states=db.execute('SELECT review_status,validation_status FROM artifact_symbolic_expressions').fetchall()
        assert all(r=={'review_status':'needs_human','validation_status':'unknown'} for r in states)
    finally:
        db.rollback();db.close()
