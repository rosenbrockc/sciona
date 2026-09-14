"""Assemble independent M5 preprocessing from caller-supplied semantic roles.

This in-memory boundary produces aligned blocks for subsequent model-family
selection. Runtime role coding and calendar data are supplied by the caller.
"""
from datetime import date
import numpy as np
import pandas as pd

from sciona.m5_grid import build as build_grid
from sciona.m5_join import left_indices
from sciona.m5_calendar import features as date_features
from sciona.m5_price_block import build as price_features
from sciona.m5_precision import downcast
from sciona.m5_lags import features as lag_features
from sciona.m5_encoding import encode

GROUPINGS=((0,),(1,),(2,),(3,),(0,2),(0,3),(1,2),(1,3),(4,),(4,0),(4,1))


def prepare(history, first_day, pairs, roles, calendar, prices, groupings):
    """Build grid, categorical roles, prices, calendar, lags and target moments.

    The target-encoding cutoff is 28 days before the last history day. Future
    targets are never accepted as input. Seven calendar categorical roles and
    five hierarchy roles correspond to the source's preprocessing inventory;
    their meaning and codes are explicit runtime configuration.
    """
    history=np.asarray(history);roles=np.asarray(roles)
    if (history.ndim!=2 or roles.shape!=(history.shape[0],5) or roles.dtype.kind not in 'iu'):
        raise ValueError('Expected history with five aligned integer hierarchy roles')
    if (not isinstance(groupings,(list,tuple)) or any(not isinstance(g,(list,tuple)) for g in groupings)
            or tuple(tuple(g) for g in groupings)!=GROUPINGS):
        raise ValueError('Eleven source grouping combinations in reviewed role order are required')
    if not isinstance(calendar,dict) or set(calendar)!={'day','week','date','categories'}:
        raise ValueError('Calendar boundary fields differ')
    days=np.asarray(calendar['day']);weeks=np.asarray(calendar['week'])
    category_values=np.asarray(calendar['categories'],dtype=object)
    if (days.ndim!=1 or weeks.shape!=days.shape or len(calendar['date'])!=len(days)
            or category_values.shape!=(len(days),7)):
        raise ValueError('Calendar rows and seven categorical roles must align')
    try:
        dates=[date.fromisoformat(v) for v in calendar['date']]
    except (ValueError,TypeError) as error:
        raise ValueError('Calendar dates must be ISO dates') from error
    grid=build_grid(history,first_day,pairs,days,weeks,prices['pair'],prices['week'])
    grid['target']=downcast(grid['target'])
    calendar_rows=left_indices(grid['day'][:,None],days[:,None])
    assert (calendar_rows>=0).all()
    selected_dates=[calendar['date'][i] for i in calendar_rows]
    selected_roles=roles[grid['series']]
    categorical={}
    for column in range(5):
        # Preserve the vocabulary formed before release filtering.
        categories=np.unique(roles[:,column])
        categorical[f'role_{column}']=pd.Categorical(selected_roles[:,column],categories=categories)
    for column in range(7):
        values=category_values[calendar_rows,column].tolist()
        nonmissing=[v for v in values if v is not None and not (isinstance(v,float) and np.isnan(v))]
        if nonmissing and not (all(isinstance(v,str) for v in nonmissing)
                               or all(isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) for v in nonmissing)):
            raise ValueError('Each category role must contain homogeneous strings or numbers')
        categorical[f'calendar_{column}']=pd.Categorical(values)
    price_calendar=dict(week=weeks,month=np.array([v.month for v in dates]),year=np.array([v.year for v in dates]))
    price_block=price_features(np.asarray(pairs)[grid['series']],grid['week'],prices,price_calendar)
    cutoff=int(first_day)+history.shape[1]-1-28
    return dict(grid=grid,categorical=categorical,prices=price_block,
        calendar=date_features(selected_dates),lags=lag_features(grid['target'],grid['series']),
        encoding=encode(grid['target'],grid['day'],selected_roles,cutoff,groupings),encoding_cutoff=cutoff,
        encoding_groupings=GROUPINGS)
