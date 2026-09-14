"""Independent base/forecast calendar features with explicit runtime roles."""
from datetime import date
import numpy as np


def _positions(base,horizon,length):
    base,horizon=np.asarray(base),np.asarray(horizon)
    if (base.ndim!=1 or not base.size or horizon.shape!=base.shape or base.dtype.kind not in 'iu'
            or horizon.dtype.kind not in 'iu' or (base<0).any() or (horizon<1).any() or (horizon>28).any()):
        raise ValueError('Aligned integer base positions and 1..28 horizons required')
    future=base.astype(np.int64)+horizon.astype(np.int64)
    if (future>=length).any():raise ValueError('Forecast exceeds calendar coverage')
    return base.astype(np.int64),horizon.astype(np.int64),future


def features(base,horizon,dates,holidays,*,seed=0):
    if type(dates) is not list or not dates or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Date list and bounded seed required')
    try:parsed=[date.fromisoformat(v) for v in dates]
    except (ValueError,TypeError) as error:raise ValueError('Valid ISO calendar dates required') from error
    holidays=np.asarray(holidays)
    if (holidays.ndim!=2 or holidays.shape[0]!=len(parsed) or holidays.dtype.kind not in 'iuf'
            or not np.isfinite(holidays).all()):raise ValueError('Aligned finite holiday matrix required')
    base,horizon,future=_positions(base,horizon,len(parsed))
    weekday=np.array([v.weekday() for v in parsed]);monthday=np.array([v.day for v in parsed])
    months=np.array([v.month for v in parsed],dtype=float)
    result=dict(weekday=weekday[future],monthday=monthday[future],base_weekday=weekday[base],
        weekday_change=(horizon%7).astype(np.int8),base_monthday=monthday[base],
        season=months[future]+np.random.RandomState(seed).normal(0,1,len(base)))
    for col in range(holidays.shape[1]):
        result[f'base_holiday_{col}']=holidays[base,col].copy()
        result[f'holiday_{col}']=holidays[future,col].copy()
    return result


def state_features(base,horizon,states,state_codes,events,*,aggregate_code=-1):
    states,state_codes,events=np.asarray(states),np.asarray(state_codes),np.asarray(events)
    if states.ndim!=1 or states.dtype.kind not in 'iu' or type(aggregate_code) is not int:
        raise ValueError('Integer state codes required')
    if states.shape!=np.asarray(base).shape:raise ValueError('State rows must align')
    # Preserve source whole-input bypass for a mixed aggregate population.
    if (states==aggregate_code).any():return {}
    if (state_codes.ndim!=1 or state_codes.dtype.kind not in 'iu' or len(set(state_codes))!=len(state_codes)
            or events.ndim!=2 or events.shape[1]!=len(state_codes) or events.dtype.kind not in 'iuf'
            or not np.isin(events,[0,1]).all()):raise ValueError('Unique states and binary event calendar required')
    base,horizon,future=_positions(base,horizon,len(events))
    lookup={int(v):i for i,v in enumerate(state_codes)}
    if any(int(s) not in lookup for s in states):raise ValueError('Missing state calendar')
    columns=np.array([lookup[int(s)] for s in states])
    ordinal=np.empty(events.shape,dtype=np.int8)
    for day in range(len(events)):
        ordinal[day]=events[max(0,day-14):day+1].sum(axis=0)*events[day]
    return dict(base_event=events[base,columns].astype(np.int8),base_event_ordinal=ordinal[base,columns],
                event=events[future,columns].astype(np.int8),event_ordinal=ordinal[future,columns])
