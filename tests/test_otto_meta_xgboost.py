import numpy as np
import pytest
from sciona import otto_meta_xgboost as module
from sciona.otto_final_blend import _bag

CONTROLS=dict(rounds=8,max_depth=3,eta=.4,subsample=.8,colsample_bytree=.8)


def inputs():
    rng=np.random.default_rng(12);y=np.tile(np.arange(9),30)
    x=np.eye(9)[y]*10+rng.uniform(0,.2,size=(270,9))-2
    return x,y,np.eye(9)*10-1.9


def test_250_real_meta_models_and_strict_blend_contract(monkeypatch):
    original=module.xgb.train;seeds=[]
    def capture(params,*args,**kwargs):
        seeds.append(params['seed']);return original(params,*args,**kwargs)
    monkeypatch.setattr(module.xgb,'train',capture)
    x,y,q=inputs();bag=module.fit_bag(x,y,q,seed=12,controls=CONTROLS)
    assert bag.shape==(250,9,9) and seeds==list(range(12,262))
    np.testing.assert_array_equal(_bag(bag,250).argmax(axis=1),np.arange(9))
    np.testing.assert_allclose(bag.sum(axis=2),1.,rtol=0,atol=1e-12)
    assert np.any(np.std(bag,axis=0)>0)


def test_query_reordering_and_seed_wrap():
    x,y,q=inputs();kwargs=dict(seed=2**32-2,controls=CONTROLS)
    a=module.fit_bag(x,y,q,**kwargs);b=module.fit_bag(x,y,q[::-1],**kwargs)
    np.testing.assert_array_equal(a,b[:,::-1,:])


def test_invalid_labels_rejected_before_native_fit(monkeypatch):
    x,y,q=inputs();y[y==8]=7
    def forbidden(*args,**kwargs):raise AssertionError('Native fit started')
    monkeypatch.setattr(module.xgb,'train',forbidden)
    with pytest.raises(ValueError,match='nine-class'):module.fit_bag(x,y,q,seed=12,controls=CONTROLS)
