"""Actual grouped DART routing, row coverage and restored-model query oracle."""
import numpy as np
import pytest
from unittest.mock import patch
from sciona import amex_row_model as row


def test_native_grouped_crossfit_and_query_mean():
    rng=np.random.default_rng(108);labels=[0]*40+[1]*40
    xs=[rng.normal(loc=y,size=(13,3)) for y in labels];qs=[rng.normal(size=(n,3)) for n in (1,7,13)]
    # Encode synthetic row identity to recover and check actual native fold membership.
    flat=np.vstack(xs);flat[:,2]=np.arange(len(flat));xs=list(np.split(flat,80))
    real=row.lgb.train;calls=[];query_oracle=[]
    def inspect(params,reference,**kw):
        tr=reference.data[:,2].astype(int);va=kw['valid_sets'][1].data[:,2].astype(int)
        assert not set(tr//13)&set(va//13)
        np.testing.assert_array_equal(reference.label,np.asarray(labels)[tr//13])
        assert params['boosting']=='dart' and 'callbacks' not in kw
        model=real(params,reference,**kw);calls.append(va)
        query_oracle.append(model.predict(np.vstack(qs),num_iteration=-1))
        return model
    with patch.object(row.lgb,'train',side_effect=inspect):result=row.fit(xs,labels,qs,rounds=8)
    assert result['models']==5 and result['configured_rounds']==8
    np.testing.assert_array_equal(np.sort(np.concatenate(calls)),np.arange(1040))
    np.testing.assert_allclose(np.concatenate(result['query_predictions']),np.mean(query_oracle,axis=0),atol=1e-14)
    assert [len(p) for p in result['query_predictions']]==[1,7,13]
    assert all(np.all((p>=0)&(p<=1)) for p in result['training_predictions'])


def test_invalid_customer_counts_labels_and_lengths():
    x=[np.ones((2,2))]*10
    with pytest.raises(ValueError):row.fit(x,[0]*6+[1]*4,x)
    with pytest.raises(ValueError):row.fit(x,[False]*5+[True]*5,x)
    with pytest.raises(ValueError):row.fit(x,[0]*5+[1]*5,[np.ones((14,2))])
