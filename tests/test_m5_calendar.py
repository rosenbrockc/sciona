import numpy as np
import pandas as pd
import pytest
from sciona.m5_calendar import features


def test_gregorian_and_iso_year_boundary():
    values=['2016-01-01','2016-01-03','2016-01-04','2016-02-29','2017-01-01']
    out=features(values)
    np.testing.assert_equal(out['iso_week'],[53,53,1,9,52])
    np.testing.assert_equal(out['relative_year'],[0,0,0,0,1])
    np.testing.assert_equal(out['week_of_month'],[1,1,1,5,1])
    np.testing.assert_equal(out['weekend'],[0,1,0,0,1])


def test_all_columns_against_pandas_calendar_oracle():
    index=pd.date_range('2000-01-01',periods=1500)
    actual=features([value.date().isoformat() for value in index])
    expected=[index.day,index.isocalendar().week,index.month,index.year-index.year.min(),
              (index.day+6)//7,index.dayofweek,index.dayofweek>=5]
    for array,value in zip(actual.values(),expected):
        assert array.dtype==np.int8
        np.testing.assert_equal(array,np.asarray(value,dtype=np.int8))


@pytest.mark.parametrize('values',[[],['bad'],['2001-02-29'],[None],['1900-01-01','2100-01-01']])
def test_invalid_dates_reject(values):
    with pytest.raises(ValueError):features(values)
