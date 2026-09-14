import numpy as np
import pytest
from sciona import otto_lasagne_120_crossfit as module


@pytest.mark.parametrize('variant',['two_hidden','three_hidden'])
def test_full_native_crossfit_reference_populations_and_learning(variant,monkeypatch):
    y=np.tile(np.arange(9),30);x=np.eye(9)[y]*20;f=np.arange(270)%5
    ids=[f'r{i}' for i in range(270)];q=np.eye(9)*20;qids=[f'q{i}' for i in range(9)]
    original=module.fit_predict;seen=[]
    def capture(x,y,q,**kwargs):
        seen.append((x.copy(),y.copy(),q.copy()));return original(x,y,q,**kwargs)
    monkeypatch.setattr(module,'fit_predict',capture)
    controls=dict(hidden=[16]*(2 if variant=='two_hidden' else 3),epochs=[20,21,22]*40,batch_size=64,learning_rate=.1,momentum=.9,ddof=1,representation='log1p')
    result=module.crossfit_lasagne(x,y,f,ids,q,qids,variant=variant,seed=12,controls=controls)
    assert len(seen)==6
    assert result['total_model_fits']==720
    for fold in range(5):
        np.testing.assert_array_equal(seen[fold][0],x[f!=fold])
        np.testing.assert_array_equal(seen[fold][1],y[f!=fold])
        np.testing.assert_array_equal(seen[fold][2],x[f==fold])
    np.testing.assert_array_equal(seen[-1][0],x)
    np.testing.assert_array_equal(result['oof'].argmax(axis=1),y)
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))
