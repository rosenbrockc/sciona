import numpy as np
import pytest
from sciona.m5_preprocessing import prepare
from sciona.m5_pooled_training import train
from sciona.m5_forecast import predict
from tests.test_m5_preprocessing import fixture


def test_native_six_family_full_horizon_and_exact_blend():
    prepared=prepare(**fixture());before=prepared['grid']['target'].copy()
    models=train(prepared,249,nonrecursive_first_day=0)
    result=predict(prepared,models,249)
    assert result['families'].shape==(6,2,28) and result['forecast'].shape==(2,28)
    assert np.isfinite(result['forecast']).all() and (result['forecast']>=0).all()
    np.testing.assert_equal(result['forecast'],sum(result['families'])/6)
    np.testing.assert_equal(prepared['grid']['target'],before)


def test_observed_future_target_rejects():
    prepared=prepare(**fixture());prepared['grid']['target'][-1]=1
    with pytest.raises(ValueError,match='Future targets'):
        predict(prepared,[],249)


def test_recursive_feedback_matches_scalar_recurrence_and_is_family_local():
    from types import SimpleNamespace
    from sciona.m5_family import frame,pools
    prepared=prepare(**fixture());models=[]
    class Learner:
        def __init__(self,recursive,offset):self.recursive=recursive;self.offset=offset
        def predict(self,features):
            return features['temporary_1_7'].to_numpy()+self.offset if self.recursive else np.full(len(features),10.)
    for recursive in (True,False):
        for offset,pooling in enumerate(('outlet','outlet_category','outlet_department'),1):
            for pool in pools(prepared,pooling):
                features,rows=frame(prepared,recursive=recursive,pooling=pooling,pool=pool,first_day=0)
                models.append(SimpleNamespace(recursive=recursive,pooling=pooling,pool=pool,
                    rows=rows,features=tuple(features.columns),model=Learner(recursive,offset)))
    result=predict(prepared,models,249)
    for family,offset in enumerate((1,2,3)):
        for identity in range(2):
            grid=prepared['grid']
            history=grid['target'][(grid['series']==identity)&(grid['day']<=249)].astype(float).tolist()
            expected=[]
            for day in range(28):
                value=sum(history[-7:])/7+offset
                history.append(value);expected.append(value)
            np.testing.assert_allclose(result['families'][family,identity],expected,rtol=1e-12)
    np.testing.assert_equal(result['families'][3:],np.full((3,2,28),10.))
