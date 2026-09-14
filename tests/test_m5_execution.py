import copy
import json
import numpy as np
import pytest
from sciona.m5_execution import prepare,execute,Prepared
from tests.test_m5_preprocessing import fixture


def payload():
    c=json.loads(json.dumps(fixture(),default=lambda value:value.tolist() if isinstance(value,np.ndarray) else value))
    return dict(version=1,identities=['synthetic-a','synthetic-b'],configuration=c,
                controls=dict(recursive_first_day=0,nonrecursive_first_day=0))


def test_full_private_boundary_and_detached_configuration():
    p=payload();prepared=prepare(p)
    p['configuration']['history'][0][0]=999
    out=execute(prepared)
    assert set(out)=={'forecast','models','horizon','model_families','scope'}
    assert out['models']==10 and np.asarray(out['forecast']).shape==(2,28)
    assert 'synthetic-a' not in json.dumps(out,allow_nan=False)
    assert 'configuration' not in repr(prepared)


@pytest.mark.parametrize('change',[
    lambda p:p.update(version=True),
    lambda p:p['configuration']['history'][0].__setitem__(0,True),
    lambda p:p['configuration']['history'][0].__setitem__(0,np.nan),
    lambda p:p['identities'].__setitem__(1,p['identities'][0]),
    lambda p:p['configuration']['roles'][0].__setitem__(4,9),
    lambda p:p['configuration'].update(future_targets=[[1.]]),
    lambda p:p['controls'].update(rounds=1),
])
def test_incompatible_boundary_rejects(change):
    p=payload();change(p)
    with pytest.raises(ValueError):prepare(p)


def test_tampered_prepared_rejects():
    prepared=prepare(payload())
    with pytest.raises(ValueError):execute(Prepared(prepared.configuration+' ',prepared.fingerprint))
