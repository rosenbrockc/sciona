from datetime import date,timedelta
import hashlib
import json
from unittest.mock import patch
import pytest
from sciona.m5u_execution import prepare,execute,Prepared


def payload():
    n=1338
    return dict(version=1,units=[[1.,2.,3.] for _ in range(n)],prices=[[2.,2.,2.] for _ in range(n)],
                roles=[[0,0,i,i,i] for i in range(3)],partitions=[[0,13],[1,14],[2,15]],
                calendar=dict(dates=[(date(2000,2,1)+timedelta(days=i)).isoformat() for i in range(n+28)],
                              holidays=[[0.] for _ in range(n+28)],state_codes=[0],events=[[0] for _ in range(n+28)]),
                controls=dict(minimum_day=299,seed=514))


def test_canonical_private_input_is_isolated_from_mutation_and_repr():
    p=payload();prepared=prepare(p);original=prepared.configuration
    p['units'][0][0]=99
    assert prepared.configuration==original
    assert prepare(json.loads(original))==prepared
    assert original not in repr(prepared) and prepared.fingerprint not in repr(prepared)


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(extra=0),lambda p:p.update(version=True),
    lambda p:p['units'][0].__setitem__(0,True),lambda p:p['units'][0].__setitem__(0,float('nan')),
    lambda p:p['calendar']['dates'].__setitem__(1,p['calendar']['dates'][0]),
    lambda p:p['calendar']['events'][0].__setitem__(0,2),
    lambda p:p['partitions'].__setitem__(1,[0,14]),
    lambda p:p['roles'][1].__setitem__(4,0),
    lambda p:p['calendar']['state_codes'].__setitem__(0,1),
    lambda p:p['prices'].pop(),lambda p:p['controls'].__setitem__('seed',0),
])
def test_invalid_contract_rejects_before_training(mutation):
    p=payload();mutation(p)
    with patch('sciona.m5u_pipeline.run') as runner:
        with pytest.raises(ValueError):prepare(p)
        runner.assert_not_called()


def test_fingerprint_and_noncanonical_forgery_reject_before_execution():
    p=prepare(payload())
    altered=json.dumps(json.loads(p.configuration),indent=2)
    invalid=[Prepared(p.configuration,'wrong'),Prepared(altered,hashlib.sha256(altered.encode()).hexdigest())]
    with patch('sciona.m5u_pipeline.run') as runner:
        for value in invalid:
            with pytest.raises(ValueError):execute(value)
        runner.assert_not_called()
