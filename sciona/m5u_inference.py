"""Repeated, perturbed forecast queries and per-series forecast averaging."""
import numpy as np
from sciona.m5u_targets import assemble
from sciona.m5u_prediction import predict


def repetitions(weights, *, power, capacity=3_000_000, minimum=20, maximum=2500):
    """Two normalize-and-clip passes, then source integer truncation."""
    weights=np.asarray(weights)
    if (weights.ndim!=1 or not weights.size or weights.dtype.kind not in 'iuf'
            or not np.isfinite(weights).all() or (weights<0).any() or not (weights>0).any()
            or power not in (1.5,2) or type(capacity) is not int or capacity<1
            or type(minimum) is not int or type(maximum) is not int or not 1<=minimum<=maximum):
        raise ValueError('Finite nonnegative weights and positive repeat limits required')
    with np.errstate(over='ignore',invalid='ignore'):
        counts=np.power(weights.astype(float),power)
        for _ in range(2):
            counts=np.clip(counts*capacity/counts.sum(),minimum,maximum)
    if not np.isfinite(counts).all():raise ValueError('Repeat allocation overflow')
    return counts.astype(np.int64)


def forecast(models,population,history,calendar,level,base_day,quantiles,*,
             scale_range,validation=False,speed=False,super_speed=False,seed=0,
             capacity=3_000_000,max_query_rows=5_000_000):
    """Return quantile x 28 horizons x sorted series before level restoration.

    The capacity parameter controls the source repetition budget. The separate
    resource limit rejects excess work without silently changing repeat counts.
    Explicit per-batch RNG seeds do not reproduce historical global RNG state.
    """
    if (type(level) is not int or level<1 or type(base_day) is not int or base_day<0
            or any(type(x) is not bool for x in (validation,speed,super_speed))
            or type(seed) is not int or not 0<=seed<2**32-29
            or type(max_query_rows) is not int or max_query_rows<1):
        raise ValueError('Invalid forecast configuration')
    rows=np.flatnonzero((population['days']==base_day)&(population['levels']==level))
    if not rows.size:raise ValueError('No forecast origins at requested day and level')
    series=population['series'][rows]
    if len(np.unique(series))!=len(series):raise ValueError('Duplicate forecast origin for series')
    order=np.argsort(series);rows=rows[order];series=series[order]
    fast=speed or super_speed
    minimum,maximum=(1,250) if fast else (20,2500)
    together=level<=9 or speed  # SUPER_SPEED alone does not select this source branch.
    batches=[]
    for horizons in ([np.arange(1,29)] if together else [np.array([h]) for h in range(1,29)]):
        origins=np.tile(rows,len(horizons))
        horizon=np.repeat(horizons,len(rows))
        counts=repetitions(population['weights'][origins],power=2 if together else 1.5,
                           capacity=capacity,minimum=minimum,maximum=maximum)
        if sum(int(x) for x in counts)>max_query_rows:
            raise ValueError('Forecast batch exceeds explicit query resource limit')
        batches.append((origins,horizon,counts))
    output=np.empty((len(quantiles),28,len(rows)))
    queries=0
    for batch,(origins,horizon,counts) in enumerate(batches):
        query_rows=np.repeat(origins,counts);query_horizons=np.repeat(horizon,counts)
        query=assemble(population,query_rows,query_horizons,history,calendar,-1,
                       out_of_sample=True,scale_range=scale_range,seed=seed+batch)
        if not np.array_equal(query['rows'],query_rows) or not np.array_equal(query['horizons'],query_horizons):
            raise ValueError('Forecast assembly changed repeated query alignment')
        values=predict(models,query['features'],query['scales'],quantiles,validation=validation)
        starts=np.r_[0,np.cumsum(counts)[:-1]]
        means=np.add.reduceat(values,starts,axis=1)/counts
        output[:,np.unique(horizon)-1,:]=means.reshape(len(quantiles),-1,len(rows))
        queries+=len(query_rows)
    return dict(predictions=output,series=series,query_rows=queries,batches=len(batches))
