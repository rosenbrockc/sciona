import numpy as np
import pytest
from sciona import otto_ovr_boosting as module

CONTROLS=dict(rounds=12,max_depth=2,eta=.2,subsample=1.,colsample_bytree=1.)


def fixture():
    y=np.repeat(np.arange(9),12)
    x=np.eye(9)[y]*20
    return x,y,np.eye(9)*20


def test_zero_count_precedes_missing_conversion():
    result=module.features([[0,2,0],[1,2,3]])
    assert np.isnan(result[0,0]) and np.isnan(result[0,2])
    np.testing.assert_array_equal(result[:,-1],[2,0])
    np.testing.assert_array_equal(result[1,:3],[1,2,3])


def test_nine_independent_models_and_no_normalization(monkeypatch):
    real=module.xgb.train;labels=[]
    def capture(params,data,**kwargs):
        assert params['objective']=='binary:logistic' and params['nthread']==1
        labels.append(data.get_label().copy())
        return real(params,data,**kwargs)
    monkeypatch.setattr(module.xgb,'train',capture)
    x,y,q=fixture();scores=module.fit_predict(x,y,q,seed=12,controls=CONTROLS)
    assert len(labels)==9 and scores.shape==(9,9)
    for label,actual in enumerate(labels):np.testing.assert_array_equal(actual,y==label)
    np.testing.assert_array_equal(scores.argmax(axis=1),np.arange(9))
    assert not np.allclose(scores.sum(axis=1),1.)


def test_repeat_and_query_population_independence():
    x,y,q=fixture();before=x.copy()
    a=module.fit_predict(x,y,q,seed=12,controls=CONTROLS)
    b=module.fit_predict(x,y,q[:1],seed=12,controls=CONTROLS)
    np.testing.assert_array_equal(a[:1],b)
    np.testing.assert_array_equal(x,before)


@pytest.mark.parametrize('problem',['absent_class','bad_seed','bad_rounds','bad_fraction'])
def test_invalid_fit_rejected(problem,monkeypatch):
    x,y,q=fixture();c=dict(CONTROLS);seed=12
    if problem=='absent_class':y[y==8]=7
    if problem=='bad_seed':seed=True
    if problem=='bad_rounds':c['rounds']=0
    if problem=='bad_fraction':c['eta']=float('nan')
    def forbidden(*args,**kwargs):raise AssertionError('Boosting started')
    monkeypatch.setattr(module.xgb,'train',forbidden)
    with pytest.raises(ValueError):module.fit_predict(x,y,q,seed=seed,controls=c)
