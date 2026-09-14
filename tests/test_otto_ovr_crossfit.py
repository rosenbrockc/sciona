import numpy as np
import pytest
from sciona import otto_ovr_crossfit as module


def fixture():
    y=np.repeat(np.arange(9),20);f=np.tile(np.arange(20)%5,9)
    x=np.eye(9)[y]*20
    return x,y,f,[f'synthetic-t{i}' for i in range(180)],np.eye(9)*20,[f'synthetic-q{i}' for i in range(9)]


def run(values):
    return module.crossfit_boosting(*values,seed=12,controls=dict(rounds=8,max_depth=2,eta=.2,subsample=1.,colsample_bytree=1.))


def test_all_six_fit_populations_and_class_order(monkeypatch):
    real=module.fit_predict;seen=[]
    def capture(x,y,q,**kwargs):
        seen.append((x.copy(),y.copy(),q.copy()))
        return real(x,y,q,**kwargs)
    monkeypatch.setattr(module,'fit_predict',capture)
    values=fixture();x,y,f,_,q,_=values
    result=run(values)
    assert len(seen)==6 and result['binary_models']==54
    assert result['oof'].shape==(180,9) and result['query'].shape==(9,9)
    for fold in range(5):
        np.testing.assert_array_equal(seen[fold][0],x[f!=fold])
        np.testing.assert_array_equal(seen[fold][1],y[f!=fold])
        np.testing.assert_array_equal(seen[fold][2],x[f==fold])
    np.testing.assert_array_equal(seen[5][0],x)
    np.testing.assert_array_equal(seen[5][2],q)
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))


def test_own_fold_labels_and_query_cannot_affect_heldout_predictions():
    values=list(fixture());a=run(values);held=values[2]==0
    values[1]=values[1].copy();values[1][held]=(values[1][held]+1)%9
    values[4]=values[4]*2
    b=run(values)
    np.testing.assert_array_equal(a['oof'][held],b['oof'][held])


@pytest.mark.parametrize('problem',['overlap','missing_fold','missing_fitting_class'])
def test_invalid_split_fails_before_fitting(problem,monkeypatch):
    values=list(fixture())
    if problem=='overlap':values[5][0]=values[3][0]
    if problem=='missing_fold':values[2][values[2]==4]=3
    if problem=='missing_fitting_class':values[2][values[1]==8]=0
    def forbidden(*args,**kwargs):raise AssertionError('Fitting started')
    monkeypatch.setattr(module,'fit_predict',forbidden)
    with pytest.raises(ValueError):run(values)
