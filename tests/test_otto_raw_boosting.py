import numpy as np
import pytest
from sciona import otto_raw_boosting as module
from sciona.otto_raw_boosting_crossfit import crossfit_boosting

CONTROLS=dict(rounds=5,max_depth=2,eta=.2,subsample=.8,colsample_bytree=.8)


def fixture():
    y=np.repeat(np.arange(9),20);f=np.tile(np.arange(20)%5,9)
    x=np.eye(9)[y]*20
    return x,y,f,[f'synthetic-t{i}' for i in range(180)],np.eye(9)*20,[f'synthetic-q{i}' for i in range(9)]


def test_full_thirty_run_bag_matches_independent_average(monkeypatch):
    x,y,_,_,q,_=fixture();real=module.xgb.train;models=[];seeds=[]
    def capture(params,data,**kwargs):
        assert params['objective']=='multi:softprob' and params['num_class']==9
        seeds.append(params['seed']);model=real(params,data,**kwargs);models.append(model);return model
    monkeypatch.setattr(module.xgb,'train',capture)
    result=module.fit_predict(x,y,q,seed=12,controls=CONTROLS)
    assert seeds==list(range(12,42)) and len(models)==30
    query=module.xgb.DMatrix(q,nthread=1)
    expected=np.mean(np.array([m.predict(query) for m in models],dtype=np.float64),axis=0)
    np.testing.assert_allclose(result,expected,rtol=1e-14)
    np.testing.assert_array_equal(result.argmax(axis=1),np.arange(9))


def test_full_five_fold_bag_and_heldout_isolation():
    values=list(fixture())
    a=crossfit_boosting(*values,seed=12,controls=CONTROLS)
    assert a['bag_runs']==30 and a['total_model_fits']==180 and a['oof'].shape==(180,9)
    np.testing.assert_allclose(a['oof'].sum(axis=1),1.,atol=1e-6)
    held=values[2]==0;values[1]=values[1].copy();values[1][held]=(values[1][held]+1)%9
    values[4]=values[4]*2
    b=crossfit_boosting(*values,seed=12,controls=CONTROLS)
    np.testing.assert_array_equal(a['oof'][held],b['oof'][held])


def test_query_batch_independence_and_seed_wrap():
    x,y,_,_,q,_=fixture()
    a=module.fit_predict(x,y,q,seed=2**32-2,controls=CONTROLS)
    b=module.fit_predict(x,y,q[:1],seed=2**32-2,controls=CONTROLS)
    np.testing.assert_array_equal(a[:1],b)


def test_overlap_rejected_before_bagging(monkeypatch):
    import sciona.otto_raw_boosting_crossfit as wrapper
    values=list(fixture());values[5][0]=values[3][0]
    def forbidden(*args,**kwargs):raise AssertionError('Bagging started')
    monkeypatch.setattr(wrapper,'fit_predict',forbidden)
    with pytest.raises(ValueError):wrapper.crossfit_boosting(*values,seed=12,controls=CONTROLS)
