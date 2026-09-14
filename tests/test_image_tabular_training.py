import copy
import numpy as np
import pytest
import torch
from sciona.image_tabular_training import fit


def sample():
    labels=[i%2 for i in range(12)]
    images=[[[[float(y),.2,.3] for _ in range(2)] for _ in range(2)] for y in labels]
    numeric=[[float(y),None] for y in labels];categorical=[[f'c{y}'] for y in labels]
    return images,numeric,categorical,labels,[i//4 for i in range(12)],[f'g{i//2}' for i in range(12)],dict(seed=12,hidden=12,epochs=40,learning_rate=.02,dropout=.5,smoothing=.1)


def test_fold_training_oof_and_equal_ensemble():
    data=sample();state=torch.random.get_rng_state().clone();f=fit(*data)
    assert torch.equal(state,torch.random.get_rng_state())
    assert len(f.models)==3 and f.oof.shape==(12,) and np.isfinite(f.oof).all()
    for m in f.models:
        assert len(m.history)==41 and m.history[-1]<m.history[0]*.5
        assert m.features.scaler.n_samples_seen_==8
    expected=np.mean([m.predict(*data[:3]) for m in f.models],axis=0)
    np.testing.assert_allclose(f.predict(*data[:3]),expected)
    assert (f.oof>=.5).astype(int).tolist()==data[3]


def test_own_heldout_features_labels_cannot_change_fit():
    data=sample();data[-1]['epochs']=3;first=fit(*data)
    data[0][0]=[[[.4,.4,.4]]];data[1][0]=[999.,None];data[2][0]=['new'];data[3][0]=1
    second=fit(*data)
    a,b=first.models[0],second.models[0]
    assert a.history==b.history
    np.testing.assert_array_equal(a.features.scaler.mean_,b.features.scaler.mean_)
    for key,value in a.network.state_dict().items():torch.testing.assert_close(value,b.network.state_dict()[key],rtol=0,atol=0)


def test_repeat_and_query_batch_independence():
    data=sample();data[-1]['epochs']=2;first=fit(*data);second=fit(*data)
    np.testing.assert_array_equal(first.oof,second.oof)
    only=[v[:1] for v in data[:3]]
    np.testing.assert_allclose(first.predict(*only),first.predict(*data[:3])[:1],atol=1e-14)


def test_cross_fold_group_rejected():
    data=sample();data[5][4]=data[5][0]
    with pytest.raises(ValueError):fit(*data)


def test_fitting_fold_one_class_rejected():
    data=sample();data[3][:]=[0]*4+[1]*8
    with pytest.raises(ValueError):fit(*data)
