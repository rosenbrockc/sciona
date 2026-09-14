import numpy as np
import pytest
from sciona import otto_augmented_boosting as model
from sciona.otto_augmented_crossfit import crossfit_boosting

CONTROLS=dict(augmentation=dict(cluster_counts=list(range(2,9)),ddof=1,n_init=2,max_iter=100),
              boosting=dict(rounds=10,max_depth=2,eta=.2,subsample=1.,colsample_bytree=1.))


def fixture():
    y=np.repeat(np.arange(9),20);f=np.tile(np.arange(20)%5,9)
    x=np.eye(9)[y]*20+np.random.default_rng(4).uniform(0,.1,(180,9))
    return x,y,f,[f'synthetic-t{i}' for i in range(180)],np.array([x[y==c].mean(axis=0) for c in range(9)]),[f'synthetic-q{i}' for i in range(9)]


def run(values):return crossfit_boosting(*values,seed=12,controls=CONTROLS)


def test_each_augmentation_fit_excludes_heldout_population(monkeypatch):
    real=model.fit_augmentation;references=[]
    def capture(reference,**kwargs):
        references.append(reference.copy());return real(reference,**kwargs)
    monkeypatch.setattr(model,'fit_augmentation',capture)
    values=fixture();x,y,f,*_=values;result=run(values)
    assert result['model_fits']==6 and len(references)==6
    for fold in range(5):np.testing.assert_array_equal(references[fold],x[f!=fold])
    np.testing.assert_array_equal(references[5],x)
    assert result['oof'].shape==(180,9)
    np.testing.assert_allclose(result['oof'].sum(axis=1),1.,atol=1e-6)
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))


def test_query_and_heldout_labels_do_not_leak():
    values=list(fixture());a=run(values);held=values[2]==0
    values[1]=values[1].copy();values[1][held]=(values[1][held]+1)%9;values[4]=values[4]*3
    b=run(values)
    np.testing.assert_array_equal(a['oof'][held],b['oof'][held])


def test_query_population_alone_leaves_all_oof_unchanged():
    values=list(fixture());a=run(values)
    values[4]=values[4]+10
    b=run(values)
    np.testing.assert_array_equal(a['oof'],b['oof'])


def test_invalid_identity_fails_before_clustering(monkeypatch):
    values=list(fixture());values[5][0]=values[3][0]
    def forbidden(*args,**kwargs):raise AssertionError('Clustering started')
    monkeypatch.setattr(model,'fit_augmentation',forbidden)
    with pytest.raises(ValueError):run(values)
