from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json
import numpy as np
import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.infall_speed import infall_speed
from sciona.physics_ingest.escape_speed_execution import build_escape_speed_execution


def verify_reused_provider(root):
    retained=json.loads((root/'docs/reviews/infall_speed_execution.json').read_text())
    path=root.parent/'sciona-atoms-physics/src/sciona/atoms/physics/infall_speed.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=retained['provider_sha256']:
        raise ValueError('Approved provider file changed')
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        row=db.execute('SELECT a.status,a.is_publishable,v.trust_tier,v.is_latest,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN catalog_artifacts_served s USING(artifact_id) WHERE v.version_id=%s',('23b6922d-d058-5dbc-aee2-2276694c7cdd',)).fetchone()
    expected=dict(status='approved',is_publishable=True,trust_tier=3,is_latest=True,
                  content_hash='485f1797fab10da7d52f2ef79af046281f01db9abadff2070b520363cb5bcd04')
    if row!=expected:raise ValueError('Exact approved provider not served')
    return dict(version_id='23b6922d-d058-5dbc-aee2-2276694c7cdd',**row)


def reference(G,m,M,r):
    with localcontext() as ctx:
        ctx.prec=2500
        G,m,M,r=(Decimal.from_float(float(v)) for v in (G,m,M,r))
        potential=-G*M*m/r
        required=-potential
        speed=(-2*potential/m).sqrt()
        return tuple(float(v) for v in [speed,-speed,required,potential])


def test_escape_contract_preserves_counterpart_direction():
    graph=build_escape_speed_execution()
    assert graph.nodes[0].matched_primitive=='sciona.atoms.physics.infall_speed.infall_speed'
    assert [p.name for p in graph.nodes[0].outputs]==['escape_speed','inward_counterpart_velocity','required_launch_energy','potential_energy']
    assert 'NOT outward escape velocity' in graph.nodes[0].outputs[1].constraints


def test_launch_energy_and_independent_reference():
    args=(np.array([1.,1e308,1e-308]),np.array([1e-6]*3),np.ones(3),np.array([2.,1e308,1.]))
    result=infall_speed(*args)
    expected=np.array([reference(*row) for row in zip(*args)]).T
    for a,b in zip(result,expected):np.testing.assert_array_equal(a,b)
    assert np.all(result[0]>0) and np.all(result[1]<0)
    np.testing.assert_array_equal(result[2],-result[3])


def test_test_mass_changes_energy_not_threshold():
    values=infall_speed(np.ones(2),np.array([1e-6,2e-6]),np.ones(2),np.ones(2))
    assert values[0][0]==values[0][1]
    assert values[2][1]==2*values[2][0]
