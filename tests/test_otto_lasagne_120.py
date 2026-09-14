import numpy as np
import pytest
from sciona import otto_lasagne_120 as module


def controls(variant):
    return dict(hidden=[16]*(2 if variant=='two_hidden' else 3),epochs=[20,21,22]*40,batch_size=64,learning_rate=.1,momentum=.9,ddof=1,representation='log1p')


@pytest.mark.parametrize('variant',['two_hidden','three_hidden'])
def test_full_120_member_native_bag_learning_and_averaging(variant,monkeypatch):
    y=np.tile(np.arange(9),20);x=np.eye(9)[y]*20;q=np.eye(9)*20
    original=module.np.load;seen=[]
    def capture(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if str(path).endswith('scores.npy'):seen.append(value.copy())
        return value
    monkeypatch.setattr(module.np,'load',capture)
    scores=module.fit_predict(x,y,q,variant=variant,seed=12,controls=controls(variant))
    assert len(seen)==1 and seen[0].shape==(120,9,9)
    np.testing.assert_allclose(scores,sum(seen[0][i].astype('float64') for i in range(120))/120,rtol=1e-5,atol=1e-7)
    np.testing.assert_array_equal(scores.argmax(axis=1),np.arange(9))
    np.testing.assert_allclose(scores.sum(axis=1),1.,atol=1e-6)


@pytest.mark.parametrize('problem',['short','constant','depth'])
def test_invalid_source_bag_contract_rejected_before_worker(problem,monkeypatch):
    y=np.tile(np.arange(9),20);x=np.eye(9)[y]*20;q=np.eye(9)*20
    options=controls('two_hidden')
    if problem=='short':options['epochs']=options['epochs'][:119]
    elif problem=='constant':options['epochs']=[20]*120
    else:options['hidden']=[16]
    def forbidden(*args,**kwargs):raise AssertionError('Worker started')
    monkeypatch.setattr(module.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError):module.fit_predict(x,y,q,variant='two_hidden',seed=12,controls=options)
