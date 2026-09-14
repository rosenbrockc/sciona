"""Synthetic evidence and isolated temporary-table transaction tests."""
import hashlib
import os
from uuid import uuid4
import pytest
import sympy as sp
from sciona.ghost.symbolic import serialize_expr
from sciona.physics_ingest.pdg_evidence import prepare_pdg_evidence, attach_pdg_source_evidence


def symbol_bytes():
    fields = ['time','electric_charge','luminous_intensity','length','amount_of_substance','mass','temperature']
    return ('UNWIND [{id:"42", properties:{latex:"x",' + ','.join('dimension_'+k+':0' for k in fields)
            + '}}] AS row\nCREATE (n:scalar{id: row.id}) SET n += row.properties SET n:symbol;\n').encode()


def row():
    return {
        'expression_id': str(uuid4()), 'version_id': str(uuid4()),
        'candidate_id': str(uuid4()), 'snapshot_id': str(uuid4()),
        'sympy_srepr': serialize_expr(sp.Eq(sp.Symbol('x'), sp.Symbol('x'), evaluate=False)),
        'source_payload': {'raw_payload': {'latex_relation':'=', 'sympy_lhs':"Symbol('pdg42')", 'sympy_rhs':"Symbol('pdg42')"}},
        'snapshot_payload': {'core_file_sha256': {'conversion_of_data_formats/symbols.cypher': hashlib.sha256(symbol_bytes()).hexdigest()}},
        'evidence_json': {'human_review': {'pending': True}},
    }


def test_evidence_preserves_version_identity_without_inventing_roles():
    source = row()
    evidence = prepare_pdg_evidence(source, symbol_bytes())
    assert evidence['expression_version_id'] == source['version_id']
    assert evidence['source_comparison']['correspondence'] == 'exact_ast_match'
    assert evidence['source_comparison']['dimension_status'] == 'checker_passed'
    assert evidence['source_variable_dimensions'][0]['symbol_name'] == 'x'
    assert 'variable_role' not in evidence['source_variable_dimensions'][0]
    assert 'validation_status' not in evidence
    assert 'review_status' not in evidence
    assert source['evidence_json'] == {'human_review': {'pending': True}}


def test_different_ast_does_not_supply_stored_variable_dimensions():
    source = row()
    source['sympy_srepr'] = serialize_expr(sp.Symbol('y'))
    assert prepare_pdg_evidence(source, symbol_bytes())['source_variable_dimensions'] == []


def test_database_batch_rollback_idempotence_and_review_preservation():
    url = os.environ.get('SCIONA_PROMOTION_TEST_DATABASE_URL')
    if not url:
        pytest.skip('requires explicit integration database')
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    db = psycopg.connect(url, row_factory=dict_row)
    try:
        db.execute('CREATE TEMP TABLE artifact_symbolic_expressions(expression_id uuid,version_id uuid,candidate_id uuid,sympy_srepr text,evidence_json jsonb,review_status text,validation_status text)')
        db.execute('CREATE TEMP TABLE physics_equation_candidates(candidate_id uuid,snapshot_id uuid,source_payload jsonb)')
        db.execute('CREATE TEMP TABLE physics_ingest_snapshots(snapshot_id uuid,payload jsonb)')
        rows = sorted([row(),row()], key=lambda r:r['expression_id'])
        for r in rows:
            db.execute("INSERT INTO artifact_symbolic_expressions VALUES(%s,%s,%s,%s,%s,'needs_human','unknown')", (r['expression_id'],r['version_id'],r['candidate_id'],r['sympy_srepr'],Jsonb(r['evidence_json'])))
            db.execute('INSERT INTO physics_equation_candidates VALUES(%s,%s,%s)', (r['candidate_id'],r['snapshot_id'],Jsonb(r['source_payload'])))
            db.execute('INSERT INTO physics_ingest_snapshots VALUES(%s,%s)', (r['snapshot_id'],Jsonb(r['snapshot_payload'])))
        db.execute("UPDATE physics_ingest_snapshots SET payload=%s WHERE snapshot_id=%s", (Jsonb({'core_file_sha256':{}}),rows[1]['snapshot_id']))
        with pytest.raises(ValueError, match='pin'):
            attach_pdg_source_evidence(db,symbol_bytes())
        assert db.execute("SELECT count(*) AS n FROM artifact_symbolic_expressions WHERE evidence_json ? 'pdg_source_comparison'").fetchone()['n'] == 0
        db.execute('UPDATE physics_ingest_snapshots SET payload=%s WHERE snapshot_id=%s', (Jsonb(rows[1]['snapshot_payload']),rows[1]['snapshot_id']))
        assert attach_pdg_source_evidence(db,symbol_bytes())['attached'] == 2
        assert attach_pdg_source_evidence(db,symbol_bytes()) == {'unchanged':2}
        states = db.execute('SELECT review_status,validation_status,evidence_json FROM artifact_symbolic_expressions').fetchall()
        assert all(r['review_status']=='needs_human' and r['validation_status']=='unknown' for r in states)
        assert all(r['evidence_json']['human_review']=={'pending':True} for r in states)
        # A changed source cannot silently overwrite the existing comparison.
        db.execute("UPDATE physics_equation_candidates SET source_payload='{}'")
        with pytest.raises(ValueError, match='existing source evidence'):
            attach_pdg_source_evidence(db,symbol_bytes())
    finally:
        db.rollback()
        db.close()
