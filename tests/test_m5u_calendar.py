from datetime import date,timedelta
import numpy as np
import pytest
from sciona.m5u_calendar import features,state_features


def test_forecast_calendar_and_noise_alignment():
    dates=[(date(2000,1,25)+timedelta(days=i)).isoformat() for i in range(40)]
    h=np.arange(80).reshape(40,2)%2
    out=features([0,7],[7,2],dates,h,seed=4)
    np.testing.assert_equal(out['monthday'],[1,3]);np.testing.assert_equal(out['base_monthday'],[25,1])
    np.testing.assert_equal(out['weekday_change'],[0,2])
    np.testing.assert_equal(out['season'],2+np.random.RandomState(4).normal(0,1,2))
    np.testing.assert_equal(out['base_holiday_0'],h[[0,7],0])


def test_event_ordinal_is_rolling_not_month_reset():
    events=np.ones((40,2),dtype=int);events[30,1]=0
    out=state_features([28,29],[1,1],[7,8],[7,8],events)
    np.testing.assert_equal(out['event_ordinal'],[15,0])
    np.testing.assert_equal(out['base_event_ordinal'],[15,15])


def test_any_aggregate_state_bypasses_entire_block():
    assert state_features([0,0],[1,1],[-1,5],[],[])=={}


def test_missing_calendar_and_duplicate_state_reject():
    with pytest.raises(ValueError):state_features([0],[1],[2],[1],np.ones((3,1)))
    with pytest.raises(ValueError):state_features([0],[1],[1],[1,1],np.ones((3,2)))
