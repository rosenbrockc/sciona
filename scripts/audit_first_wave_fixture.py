"""Read-only source classification of the independent first-wave physics graph."""
import hashlib
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

SOURCE_VERSION = '693697cf-a97c-53bd-8d6b-19dc3c7697c1'
SOURCE_HASH = '97dbc4000139b6514d678829e16c39e045f02f47ecbe91356863028decb932b3'
FIXTURE = 'tests/physics_ingest/fixtures/pdg_payloads/solve_substitute_chain.pdg.json'
FIXTURE_HASH = '6c292ce672c09efe66af87ab154646713eee72a1d8be81c89e4b2eb44b4ae506'


def classify(snapshot, fixture):
    """Require full payload equality, not merely fixture-shaped identifiers."""
    if snapshot != fixture:
        raise ValueError('Stored source differs from the reviewed synthetic fixture')
    if fixture.get('validation_subject') != 'pdg_fixture:solve_substitute_chain':
        raise ValueError('Unexpected fixture provenance')
    return 'synthetic_parser_fixture'


def audit(root):
    raw = (root / FIXTURE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_HASH:
        raise ValueError('Reviewed fixture changed')
    fixture = json.loads(raw)
    with psycopg.connect(dotenv_values(root / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
                         row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if graph != dict(status='draft', is_publishable=False, content_hash=SOURCE_HASH):
            raise ValueError('Original graph state changed')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s', (SOURCE_VERSION,)).fetchall()
        if len(bindings) != 4 or any(b['status'] != 'active' for b in bindings):
            raise ValueError('Source binding inventory changed')
        equations = {}
        for binding in bindings:
            rows = db.execute('''SELECT q.source_payload,s.payload FROM artifacts a
                JOIN artifact_versions v USING(artifact_id)
                JOIN artifact_symbolic_expressions e ON e.version_id=v.version_id
                JOIN physics_equation_candidates q USING(candidate_id)
                JOIN physics_ingest_snapshots s USING(snapshot_id)
                WHERE a.fqdn=%s AND v.content_hash=%s''',
                (binding['bound_artifact_fqdn'], binding['bound_version_content_hash'])).fetchall()
            if len(rows) != 1:
                raise ValueError('Source binding must resolve uniquely')
            row = rows[0]
            classify(row['payload'], fixture)
            equations[row['source_payload']['id']] = row['source_payload']
        if equations != {e['id']: e for e in fixture['equations']}:
            raise ValueError('Stored candidate coverage differs from fixture')
        count = db.execute("SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_kind='cdg'").fetchone()['n']
    return dict(read_only=True, approval_applied=False, source_version=SOURCE_VERSION,
                source_hash=SOURCE_HASH, classification='synthetic_parser_fixture',
                fixture=FIXTURE, fixture_sha256=FIXTURE_HASH, exact_snapshot_match=True,
                exact_candidate_coverage=True, bindings=4, equations=3, served_cdgs=count,
                disposition='Retain as draft test provenance; exclude from externally sourced physics backlog.',
                missing_premises=['Nonzero mass for division (positive mass for the physical domain).',
                                  'Acceleration equals the second time derivative of position.',
                                  'Consistent time-dependent force and acceleration; twice differentiable position.',
                                  'Inertial frame and net force for the Newtonian interpretation.'],
                limitations=['Classification is not execution approval.',
                             'A future executable derivation must explicitly add premises and undergo independent validation.',
                             'No status, publication, binding, or pipeline selection behavior changed.'],
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    (root / 'docs/reviews/first_wave_fixture_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
