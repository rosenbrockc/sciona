import numpy as np
import pytest
from sciona import otto_lasagne as module


def inputs():
    y=np.tile(np.arange(9),30)
    return np.eye(9)[y]*20,y,np.eye(9)*20


def controls(variant):
    return dict(hidden=[16],epochs=[30]*(2 if variant=='pair' else 6),batch_size=64,learning_rate=.1,momentum=.9,ddof=1)


@pytest.mark.parametrize('variant',['pair','six_log'])
def test_native_source_run_counts_learn_and_repeat(variant):
    x,y,q=inputs();kwargs=dict(variant=variant,seed=12,controls=controls(variant))
    a=module.fit_predict(x,y,q,**kwargs);b=module.fit_predict(x,y,q[::-1],**kwargs)
    np.testing.assert_array_equal(a.argmax(axis=1),np.arange(9))
    np.testing.assert_allclose(a,b[::-1],rtol=0,atol=1e-7)
    np.testing.assert_allclose(a.sum(axis=1),1.,atol=1e-6)


def test_wrong_run_schedule_rejected_before_worker(monkeypatch):
    x,y,q=inputs();options=controls('pair');options['epochs']=[30]
    def forbidden(*args,**kwargs):raise AssertionError('Worker started')
    monkeypatch.setattr(module.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError,match='per source run'):module.fit_predict(x,y,q,variant='pair',seed=12,controls=options)
