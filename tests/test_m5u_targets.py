from datetime import date,timedelta
import numpy as np
import pandas as pd
from sciona.m5u_targets import assemble


def inputs():
    n=80
    frame=pd.DataFrame({'raw_ewm_15':np.full(n,4.),'raw_ewm_30':np.full(n,2.),
                        'role_0':pd.Categorical(np.zeros(n,dtype=int))})
    population=dict(features=frame,days=np.arange(n),series=np.zeros(n,dtype=int),
        scaled_columns=('raw_ewm_15','raw_ewm_30'),ewm_columns=('raw_ewm_15','raw_ewm_30'),volumes=np.full(n,2.),weights=np.ones(n))
    calendar=dict(dates=[(date(2000,12,1)+timedelta(days=i)).isoformat() for i in range(n+28)],
        holidays=np.zeros((n+28,1)),state_codes=[0],events=np.ones((n+28,1)))
    return population,np.arange(n,dtype=float)[:,None],calendar


def test_duplicate_samples_target_join_and_group_filter():
    p,h,c=inputs();out=assemble(p,[1,1,40,79],[2,3,2,2],h,c,2001,scale_range=0)
    np.testing.assert_equal(out['rows'],[1,1]);np.testing.assert_equal(out['targets'],[1.5,2.])
    np.testing.assert_equal(out['features']['ewm_difference_0_1'].to_numpy(),[1.,1.])
    np.testing.assert_equal(out['features']['ewm_ratio_0_1'].to_numpy(),[2.,2.])
    assert isinstance(out['features']['role_0'].dtype,pd.CategoricalDtype)


def test_oos_retains_future_without_observed_targets():
    p,h,c=inputs();out=assemble(p,[79],[2],h,c,2001,out_of_sample=True,scale_range=0)
    np.testing.assert_equal(out['targets'],[-.5]);np.testing.assert_equal(out['groups'],[2001])
    np.testing.assert_equal(out['features']['event_ordinal'].to_numpy(),[15])
