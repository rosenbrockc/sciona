import numpy as np
import pytest
from sciona import otto_meta_tuning as module
from sciona import otto_meta_xgboost as xgb_module


def inputs():
    rng=np.random.default_rng(12);y=np.tile(np.arange(9),40)
    x=np.eye(9)[y]*10+rng.uniform(0,.2,size=(360,9))-2
    return x,y,np.arange(360)%4,[f'r{i}' for i in range(360)]


def candidates():
    return [dict(rounds=n,max_depth=3,eta=.4,subsample=.8,colsample_bytree=.8) for n in (1,8)]


def test_two_real_candidates_full_bags_and_pooled_scoring(monkeypatch):
    x,y,f,ids=inputs();original=xgb_module.fit_bag;seen=[]
    def capture(a,b,q,**kwargs):
        seen.append((a.copy(),b.copy(),q.copy()));return original(a,b,q,**kwargs)
    monkeypatch.setattr(xgb_module,'fit_bag',capture)
    result=module.tune(x,y,f,ids,family='xgboost',seed=12,candidates=candidates())
    assert len(seen)==8 and result['models_per_candidate']==1000
    for i,(a,b,q) in enumerate(seen):
        held=f==i%4
        np.testing.assert_array_equal(a,x[~held]);np.testing.assert_array_equal(b,y[~held]);np.testing.assert_array_equal(q,x[held])
    oracle=[sum(-np.log(max(float(row[label]),np.finfo(float).eps)) for row,label in zip(scores,y))/len(y) for scores in result['candidate_oof']]
    np.testing.assert_allclose(result['log_losses'],oracle)
    assert result['selected_index']==1
    np.testing.assert_array_equal(result['candidate_oof'][1].argmax(axis=1),y)


@pytest.mark.parametrize('problem',['duplicate','missing_class','five_folds'])
def test_invalid_partition_rejected_before_learner(problem,monkeypatch):
    x,y,f,ids=inputs()
    if problem=='duplicate':ids[1]=ids[0]
    elif problem=='missing_class':f[y==8]=0
    else:f[0]=4
    def forbidden(*args,**kwargs):raise AssertionError('Learner started')
    monkeypatch.setattr(module,'_learner',forbidden)
    with pytest.raises(ValueError):module.tune(x,y,f,ids,family='xgboost',seed=12,candidates=candidates())
