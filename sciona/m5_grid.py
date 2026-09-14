"""Independent day-major forecast grid and source release-week filtering.

Caller-supplied integer pair codes identify price groups. No dataset schema,
calendar, identifiers or records are embedded here. The 28 future days carry
missing targets; missing release groups are filtered as in the source join.
"""
import numpy as np


def _integer_vector(value):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in 'iu':
        raise ValueError('Expected integer vector')
    return value


def build(history, first_day, pairs, calendar_days, calendar_weeks, price_pairs, price_weeks):
    history = np.asarray(history)
    pairs, calendar_days, calendar_weeks, price_pairs, price_weeks = [
        _integer_vector(v) for v in (pairs, calendar_days, calendar_weeks, price_pairs, price_weeks)]
    if (history.ndim != 2 or not all(history.shape) or history.dtype.kind not in 'iuf'
            or pairs.size != history.shape[0] or len(np.unique(pairs)) != pairs.size
            or calendar_days.shape != calendar_weeks.shape or price_pairs.shape != price_weeks.shape):
        raise ValueError('Invalid history or aligned role populations')
    if isinstance(first_day, (bool,np.bool_)) or not isinstance(first_day, (int,np.integer)):
        raise ValueError('First day must be an integer')
    history = history.astype(np.float64)
    if np.isinf(history).any() or (history < 0).any():
        raise ValueError('Targets must be nonnegative finite values or NaN')
    if len(np.unique(calendar_days)) != calendar_days.size:
        raise ValueError('Calendar days must be unique')
    day_count = history.shape[1] + 28
    days = np.arange(int(first_day),int(first_day)+day_count,dtype=np.int64)
    calendar = dict(zip(map(int,calendar_days),map(int,calendar_weeks)))
    if any(int(d) not in calendar for d in days):
        raise ValueError('Calendar must cover history and all 28 future days')
    release = {}
    for pair, week in zip(price_pairs,price_weeks):
        pair,week = int(pair),int(week)
        release[pair] = min(release.get(pair,week),week)
    series = np.tile(np.arange(history.shape[0]),day_count)
    row_days = np.repeat(days,history.shape[0])
    row_weeks = np.repeat([calendar[int(day)] for day in days],history.shape[0])
    row_release = np.array([release.get(int(pairs[s]),np.nan) for s in series])
    keep = row_weeks >= row_release
    if not keep.any():
        raise ValueError('Release filtering removed the entire grid')
    retained_release = row_release[keep]
    offset = retained_release - retained_release.min()
    if offset.max() > np.iinfo(np.int16).max:
        raise ValueError('Release offset exceeds int16 range')
    targets = np.concatenate((history.T.ravel(),np.full(28*history.shape[0],np.nan)))
    return dict(series=series[keep],day=row_days[keep],week=row_weeks[keep],
                target=targets[keep],release=offset.astype(np.int16))
