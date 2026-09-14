import numpy as np
import pytest
from sciona.otto_meta_tuning import tune
from sciona import otto_meta_adaboost,otto_meta_neural


@pytest.mark.parametrize('family',['adaboost_extratrees','lasagne_neural'])
def test_full_native_four_fold_candidate_selection(family,monkeypatch):
    rng=np.random.default_rng(12);y=np.tile(np.arange(9),20)
    x=np.eye(9)[y]*10+rng.uniform(0,.2,size=(180,9))-2
    folds=np.arange(180)%4;ids=[f'r{i}' for i in range(180)]
    if family=='adaboost_extratrees':
        implementation=otto_meta_adaboost;runs=250
        candidates=[dict(algorithm='SAMME',boost_rounds=4,learning_rate=.7,trees=3,max_depth=depth,min_samples_leaf=2,max_features=9) for depth in (1,3)]
    else:
        implementation=otto_meta_neural;runs=600
        candidates=[dict(hidden=[16],epochs=epochs,batch_size=64,learning_rate=.1,momentum=.9,ddof=1) for epochs in (1,20)]
    original=implementation.fit_bag;seen=[]
    def capture(a,b,q,**kwargs):
        seen.append((a.copy(),b.copy(),q.copy()))
        result=original(a,b,q,**kwargs)
        assert result.shape==(runs,len(q),9)
        return result
    monkeypatch.setattr(implementation,'fit_bag',capture)
    result=tune(x,y,folds,ids,family=family,seed=12,candidates=candidates)
    assert len(seen)==8 and result['models_per_candidate']==4*runs
    for index,(a,b,q) in enumerate(seen):
        held=folds==index%4
        np.testing.assert_array_equal(a,x[~held])
        np.testing.assert_array_equal(b,y[~held])
        np.testing.assert_array_equal(q,x[held])
    oracle=[sum(-np.log(max(float(row[label]),np.finfo(float).eps)) for row,label in zip(scores,y))/len(y) for scores in result['candidate_oof']]
    np.testing.assert_allclose(result['log_losses'],oracle)
    assert result['selected_index']==1
    np.testing.assert_array_equal(result['candidate_oof'][1].argmax(axis=1),y)
