from unittest.mock import patch
import numpy as np
import pytest
from sciona.m5u_training import schedule,train_batch,PARAMETERS


def test_full_source_search_effort_and_base_partitions():
    expected=[33,30,28,26,24,22,19,17,15,9,8,6,6,6]
    assert [schedule(level,1.)['iterations'] for level in PARAMETERS]==expected
    assert schedule(13,1.)['scale_range']==2.
    assert schedule(14,1.)['scale_range']==2.
    assert schedule(15,1.)['scale_range']==.5


def test_speed_and_super_speed_keep_distinct_sampling_and_search_factors():
    regular=schedule(1,.25);fast=schedule(1,.25,speed=True)
    fastest=schedule(1,.25,super_speed=True)
    assert fast['fraction']==regular['fraction']/2
    assert fastest['fraction']==pytest.approx(regular['fraction']/5)
    assert fast['iterations']==16 and fastest['iterations']==8
    assert schedule(1,.25,speed=True,super_speed=True)==fastest
    assert regular['fraction']==pytest.approx(.3*.8*4**.6)


def test_wrong_controller_is_rejected_before_sampling():
    prepared={'levels':np.array([11,13])}
    with patch('sciona.m5u_training.sample') as sampling:
        with pytest.raises(ValueError,match='aggregate controller'):
            train_batch(prepared,{}, {},-1)
        sampling.assert_not_called()


def test_combined_level_eleven_uses_controller_not_fitted_level_mask():
    prepared=dict(levels=np.array([11]),factors=np.array([.5]),history=np.ones((4,1)))
    population=dict(weights=np.ones(4),levels=np.full(4,11))
    bag=dict(targets=np.ones(4),features=np.ones((4,2)))
    # The learner boundary is inspected here; native execution is checked separately.
    import pandas as pd
    bag['features']=pd.DataFrame(bag['features'])
    with patch('sciona.m5u_training.sample',return_value=(np.array([0]),np.array([1]))), \
         patch('sciona.m5u_training.assemble',return_value=bag), \
         patch('sciona.m5u_training.train',return_value=[]) as learner:
        train_batch(prepared,population,{},-1)
    assert learner.call_args.args[3]==-1
    assert learner.call_args.kwargs['iterations']==8
