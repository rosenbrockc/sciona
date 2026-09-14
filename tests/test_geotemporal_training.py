import copy
import numpy as np
import pytest
from sciona.geotemporal_training import fit,features,points


def sample():
    observations=[dict(x=float(i%2),y=0.,time=float(i),target=float(1+i%2)) for i in range(9)]
    context=[dict(x=0.,y=0.,time=float(i),value=float(i%3)) for i in range(12)]
    blocks=[i//3 for i in range(9)]
    c=dict(radius=3.,lookback=10.,period=12.,neighbors=2,gap=0.,seed=12,trees=32,max_depth=5,min_leaf=1,smoothing=.2)
    return observations,context,blocks,c


def test_full_forward_validation_and_refit():
    obs,context,blocks,c=sample();f=fit(obs,context,blocks,c)
    assert f.validation_predictions[:3]==[None]*3
    assert all(v is not None for v in f.validation_predictions[3:])
    assert len(f.folds)==2 and len(f.estimator.estimators_)==32
    assert f.validation_mse==pytest.approx(np.mean((np.array(f.validation_predictions[3:])-[r['target'] for r in obs[3:]])**2))
    assert [len(a['train_indices']) for a in f.folds]==[3,6]
    assert all(tree.tree_.n_node_samples[0]==9 for tree in f.estimator.estimators_)
    result=f.predict([dict(x=0.,y=0.,time=10.),dict(x=1.,y=0.,time=10.)])
    assert len(result)==2 and all(np.isfinite(v) and v>=0 for v in result)


def test_heldout_labels_never_enter_own_fold():
    obs,context,blocks,c=sample();first=fit(obs,context,blocks,c)
    for r in obs[3:6]:r['target']+=1000
    second=fit(obs,context,blocks,c)
    for key in ('training_features','validation_features','predictions'):
        np.testing.assert_array_equal(first.folds[0][key],second.folds[0][key])
    # Later expanding folds legitimately learn from the newly historical labels.
    assert not np.array_equal(first.folds[1]['training_features'],second.folds[1]['training_features'])


def test_future_context_cannot_change_earlier_fold():
    obs,context,blocks,c=sample();first=fit(obs,context,blocks,c)
    for r in context:
        if r['time']>5:r['value']=1000
    second=fit(obs,context,blocks,c)
    np.testing.assert_array_equal(first.folds[0]['predictions'],second.folds[0]['predictions'])


def test_repeat_and_copied_history():
    obs,context,blocks,c=sample();f=fit(obs,context,blocks,c);g=fit(obs,context,blocks,c)
    assert f.validation_predictions==g.validation_predictions
    obs[0]['target']=999;context[0]['value']=999;c['gap']=999
    assert f.history[0]['target']==1 and f.context[0]['value']==0 and f.controls['gap']==0


def test_past_query_rejected():
    obs,context,blocks,c=sample();f=fit(obs,context,blocks,c)
    with pytest.raises(ValueError):f.predict([dict(x=0.,y=0.,time=8.)])
