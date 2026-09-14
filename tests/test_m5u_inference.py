from datetime import date,timedelta
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from sciona.m5u_inference import repetitions,forecast
from sciona.m5u_targets import assemble


def test_repetitions_match_pandas_two_passes_including_clipping_and_truncation():
    weights=pd.Series([0.,.02,.3,.9,4.])
    for power in (1.5,2):
        expected=weights**power
        for _ in range(2):expected=(expected*97/expected.sum()).clip(2,35)
        np.testing.assert_array_equal(repetitions(weights,power=power,capacity=97,minimum=2,maximum=35),expected.astype(int))


def inputs(level):
    frame=pd.DataFrame({'raw_ewm_15':[4.,8.],'raw_ewm_30':[2.,4.],
                        'role_0':pd.Categorical([0,0])})
    p=dict(features=frame,days=np.array([79,79]),series=np.array([1,0]),levels=np.array([level,level]),
           scaled_columns=('raw_ewm_15','raw_ewm_30'),ewm_columns=('raw_ewm_15','raw_ewm_30'),
           volumes=np.array([2.,3.]),weights=np.array([.2,.8]))
    c=dict(dates=[(date(2000,2,1)+timedelta(days=i)).isoformat() for i in range(108)],
           holidays=np.zeros((108,1)),state_codes=[0],events=np.zeros((108,1)))
    history=np.ones((80,2))
    query=assemble(p,[0,1],[1,1],history,c,-1,out_of_sample=True,scale_range=0)
    class Model:
        feature_name_=list(query['features'])
        def predict(self,x):return x.raw_ewm_15.to_numpy()**2+x.days_forward.to_numpy()
    return p,history,c,[SimpleNamespace(bag=0,group=1,quantile=.5,model=Model())]


@pytest.mark.parametrize('level,speed,super_speed,batches',[(1,False,False,1),(11,False,False,28),(11,True,False,1),(11,False,True,28)])
def test_forecast_horizons_sorted_series_and_source_batch_modes(level,speed,super_speed,batches):
    p,h,c,m=inputs(level)
    result=forecast(m,p,h,c,level,79,[.5],scale_range=0,speed=speed,super_speed=super_speed,capacity=112)
    assert result['batches']==batches
    np.testing.assert_array_equal(result['series'],[0,1])
    expected=np.stack([64/3+np.arange(1,29)*3,8+np.arange(1,29)*2],axis=1)
    np.testing.assert_allclose(result['predictions'][0],expected)
    assert result['query_rows']>=56


def test_perturbed_repeats_are_averaged_after_inverse_scaling():
    p,h,c,m=inputs(1)
    result=forecast(m,p,h,c,1,79,[.5],scale_range=.7,capacity=112,seed=74)
    counts=repetitions(np.tile(p['weights'][[1,0]],28),power=2,capacity=112)
    rows=np.repeat(np.tile([1,0],28),counts)
    horizons=np.repeat(np.repeat(np.arange(1,29),2),counts)
    query=assemble(p,rows,horizons,h,c,-1,out_of_sample=True,scale_range=.7,seed=74)
    raw=m[0].model.predict(query['features'])*query['scales']
    oracle=pd.DataFrame({'series':p['series'][rows],'horizon':horizons,'value':raw}).groupby(['horizon','series']).value.mean().to_numpy().reshape(28,2)
    np.testing.assert_allclose(result['predictions'][0],oracle)
    baseline=forecast(m,p,h,c,1,79,[.5],scale_range=0,capacity=112,seed=74)
    assert not np.allclose(result['predictions'],baseline['predictions'])


def test_resource_rejection_happens_before_query_assembly():
    p,h,c,m=inputs(1)
    with patch('sciona.m5u_inference.assemble') as assembly:
        with pytest.raises(ValueError,match='resource limit'):
            forecast(m,p,h,c,1,79,[.5],scale_range=0,max_query_rows=20)
        assembly.assert_not_called()
