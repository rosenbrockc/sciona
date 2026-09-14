from copy import deepcopy
import os
import pytest
from tests.physics_ingest.test_pdg_evidence import row, symbol_bytes
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence
from sciona.physics_ingest.parse_corroboration import prepare_parse_corroboration, corroborate_pending_parses


def ready_row():
    r = row()
    r.update(review_status='needs_human', parse_status='normalized', parse_confidence=0.4)
    r['evidence_json']['parse_roundtrip'] = {'status':'passed'}
    r['evidence_json']['pdg_source_comparison'] = prepare_pdg_evidence(r,symbol_bytes())
    return r


def test_corroboration_retains_prior_score_and_does_not_approve():
    r = ready_row(); original = deepcopy(r)
    update = prepare_parse_corroboration(r,symbol_bytes())
    assert r == original
    assert update['parse_confidence'] == 0.95
    assert update['evidence_json']['parse_corroboration']['prior_parse_confidence'] == 0.4
    assert set(update) == {'parse_confidence','evidence_json'}
    assert prepare_parse_corroboration({**r, **update},symbol_bytes()) is None


@pytest.mark.parametrize('change', ['reviewed','failed_roundtrip','changed_source','changed_ast','changed_evidence','changed_pin'])
def test_corroboration_rejects_invalid_or_stale_evidence(change):
    r = ready_row(); content = symbol_bytes()
    if change == 'reviewed': r['review_status']='human_reviewed'
    elif change == 'failed_roundtrip': r['evidence_json']['parse_roundtrip']['status']='failed'
    elif change == 'changed_source': r['source_payload']={}
    elif change == 'changed_ast': r['sympy_srepr']="Symbol('y')"
    elif change == 'changed_evidence': r['evidence_json']['pdg_source_comparison']['scope']='edited'
    elif change == 'changed_pin': content += b'\n'
    with pytest.raises(ValueError):
        prepare_parse_corroboration(r,content)


def test_database_batch_is_atomic_and_keeps_review_states():
    url = os.environ.get('SCIONA_PROMOTION_TEST_DATABASE_URL')
    if not url: pytest.skip('requires explicit integration database')
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    db=psycopg.connect(url,row_factory=dict_row)
    try:
        db.execute('CREATE TEMP TABLE artifact_symbolic_expressions(expression_id uuid,version_id uuid,candidate_id uuid,sympy_srepr text,evidence_json jsonb,review_status text,validation_status text,parse_status text,parse_confidence double precision)')
        db.execute('CREATE TEMP TABLE physics_equation_candidates(candidate_id uuid,snapshot_id uuid,source_payload jsonb)')
        db.execute('CREATE TEMP TABLE physics_ingest_snapshots(snapshot_id uuid,payload jsonb)')
        rows=sorted([ready_row(),ready_row()],key=lambda r:r['expression_id'])
        for r in rows:
            db.execute("INSERT INTO artifact_symbolic_expressions VALUES(%s,%s,%s,%s,%s,'needs_human','unknown','normalized',0.4)",(r['expression_id'],r['version_id'],r['candidate_id'],r['sympy_srepr'],Jsonb(r['evidence_json'])))
            db.execute('INSERT INTO physics_equation_candidates VALUES(%s,%s,%s)',(r['candidate_id'],r['snapshot_id'],Jsonb(r['source_payload'])))
            db.execute('INSERT INTO physics_ingest_snapshots VALUES(%s,%s)',(r['snapshot_id'],Jsonb(r['snapshot_payload'])))
        db.execute("UPDATE physics_equation_candidates SET source_payload='{}' WHERE candidate_id=%s",(rows[1]['candidate_id'],))
        with pytest.raises(ValueError,match='stale'): corroborate_pending_parses(db,symbol_bytes())
        assert db.execute('SELECT min(parse_confidence) AS lo,max(parse_confidence) AS hi FROM artifact_symbolic_expressions').fetchone()=={'lo':0.4,'hi':0.4}
        db.execute('UPDATE physics_equation_candidates SET source_payload=%s WHERE candidate_id=%s',(Jsonb(rows[1]['source_payload']),rows[1]['candidate_id']))
        assert corroborate_pending_parses(db,symbol_bytes())=={'corroborated':2}
        assert corroborate_pending_parses(db,symbol_bytes())=={'unchanged':2}
        states=db.execute('SELECT review_status,validation_status FROM artifact_symbolic_expressions').fetchall()
        assert all(r=={'review_status':'needs_human','validation_status':'unknown'} for r in states)
    finally:
        db.rollback();db.close()
