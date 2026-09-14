"""Native downstream variant routing and restored query mean checks."""
import numpy as np
import pytest
from unittest.mock import patch
from sklearn.model_selection import StratifiedKFold
from sciona import amex_tree as tree


def fixture():
    rng=np.random.default_rng(110);x=rng.normal(size=(1000,20));q=rng.normal(size=(8,20))
    labels=(x[:,0]>0).astype(int).tolist();s=rng.uniform(size=(1000,13));t=rng.uniform(size=(8,13));s[:5,0]=np.nan
    return x,s,labels,q,t


def test_native_variant_features_folds_and_predictions():
    x,s,y,q,t=fixture();plans=list(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y))
    native=tree.lgb.train;queries=[];calls=[]
    def inspect(params,dataset,**kwargs):
        index=len(calls);tr,va=plans[index%5];ref=x if index<5 else np.column_stack((x,s));query=q if index<5 else np.column_stack((q,t))
        np.testing.assert_array_equal(dataset.data,ref[tr]);np.testing.assert_array_equal(kwargs['valid_sets'][1].data,ref[va])
        assert params['feature_fraction']==.05 and params['bagging_fraction']==.75
        model=native(params,dataset,**kwargs);queries.append(model.predict(query,num_iteration=-1));calls.append(va)
        return model
    with patch.object(tree.lgb,'train',side_effect=inspect):r=tree.fit(x,s,y,q,t,rounds=8)
    assert r['models']==10
    for i,name in enumerate(('manual','manual_with_slots')):
        np.testing.assert_array_equal(np.sort(np.concatenate(calls[i*5:i*5+5])),np.arange(1000))
        np.testing.assert_allclose(r['variants'][name]['query_predictions'],np.mean(queries[i*5:i*5+5],axis=0),atol=1e-14)
        assert np.isfinite(r['variants'][name]['training_predictions']).all()


def test_invalid_slots_and_labels():
    x,s,y,q,t=fixture()
    with pytest.raises(ValueError):tree.fit(x,s[:,:12],y,q,t)
    s[0,0]=1.1
    with pytest.raises(ValueError):tree.fit(x,s,y,q,t)
