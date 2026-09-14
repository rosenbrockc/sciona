"""Web Traffic calendar lags; MIT source adaptation.

Derived from Arturus/kaggle-web-traffic; see docs/licenses/WebTraffic-MIT.txt.
Legacy permissive .loc date lookup is expressed by explicit reindex.
"""
from typing import List
import numpy as np
import pandas as pd

def lag_indexes(begin, end) -> List[pd.Series]:
    """
    Calculates indexes for 3, 6, 9, 12 months backward lag for the given date range
    :param begin: start of date range
    :param end: end of date range
    :return: List of 4 Series, one for each lag. For each Series, index is date in range(begin, end), value is an index
     of target (lagged) date in a same Series. If target date is out of (begin,end) range, index is -1
    """
    dr = pd.date_range(begin, end)
    base_index = pd.Series(np.arange(0, len(dr)), index=dr)

    def lag(offset):
        dates = dr - offset
        return pd.Series(data=base_index.reindex(dates).fillna(-1).astype(np.int16).values, index=dr)
    return [lag(pd.DateOffset(months=m)) for m in (3, 6, 9, 12)]
