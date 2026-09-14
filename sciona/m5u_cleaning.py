"""Independent source date-population filtering and missing-feature sentinel."""
import numpy as np


def retained_rows(days,calendar_years,calendar_months,minimum_day,*,stacked=True):
    """Return positions retained by source cleaning for stacked or pivot input.

    Day values index the full caller-supplied calendar. Pivot input applies the
    lower boundary only when that boundary row exists, and slices in row order.
    """
    days,years,months=[np.asarray(v) for v in (days,calendar_years,calendar_months)]
    if (days.ndim!=1 or years.ndim!=1 or not years.size or months.shape!=years.shape
            or any(v.dtype.kind not in 'iu' for v in (days,years,months))
            or (days<0).any() or (days>=len(years)).any() or (months<1).any() or (months>12).any()):
        raise ValueError('Aligned integer day positions and full calendar roles required')
    if type(minimum_day) is not int or not 0<=minimum_day<len(years) or type(stacked) is not bool:
        raise ValueError('Invalid minimum day or table mode')
    keep=np.ones(len(days),dtype=bool)
    if stacked:keep &= days>=minimum_day
    else:
        if len(np.unique(days))!=len(days):raise ValueError('Pivot day index must be unique')
        boundary=np.flatnonzero(days==minimum_day)
        if boundary.size:keep[:boundary[0]]=False
    keep &= years[days]!=years.min()
    keep &= ~np.isin(months[days],[10,11,12,1])
    return np.flatnonzero(keep)


def missing_sentinel(values):
    """Replace NaN with -10; retain finite values and infinities unchanged."""
    values=np.asarray(values)
    if values.dtype.kind not in 'iuf':raise ValueError('Real numeric features required')
    result=values.astype(float).copy()
    result[np.isnan(result)]=-10
    return result
