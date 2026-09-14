"""Join sampled history rows to forecast targets, calendar roles and scales."""
from datetime import date
import numpy as np
import pandas as pd
from sciona.m5u_exponential import crosses
from sciona.m5u_calendar import features as calendar_features,state_features
from sciona.m5u_scaling import transform


def assemble(population,rows,horizons,history,calendar,validation_group,*,out_of_sample=False,scale_range=.1,seed=0):
    rows,horizons,history=np.asarray(rows),np.asarray(horizons),np.asarray(history)
    if (rows.ndim!=1 or not rows.size or horizons.shape!=rows.shape or rows.dtype.kind not in 'iu'
            or horizons.dtype.kind not in 'iu' or (rows<0).any() or (rows>=len(population['features'])).any()
            or history.ndim!=2 or history.dtype.kind not in 'iuf' or np.isinf(history).any()
            or (horizons<1).any() or (horizons>28).any()):raise ValueError('Aligned sampled indices, horizons and real history required')
    if type(out_of_sample) is not bool or type(seed) is not int or not 0<=seed<2**32-1:
        raise ValueError('Invalid execution mode or seed')
    base=population['days'][rows];series=population['series'][rows];future=base+horizons
    if not out_of_sample:
        eligible=future<len(history)
        rows,horizons,base,series,future=[v[eligible] for v in (rows,horizons,base,series,future)]
    if not rows.size:raise ValueError('No eligible forecast targets')
    if (series<0).any() or (series>=history.shape[1]).any():raise ValueError('Series outside target history')
    frame=population['features'].iloc[rows].reset_index(drop=True).copy()
    cross,pairs=crosses(frame[list(population['ewm_columns'])].to_numpy())
    scaled=list(population['scaled_columns'])
    frame['days_forward']=horizons.astype(np.int8)
    for i,(a,b) in enumerate(pairs):
        difference=f'ewm_difference_{a}_{b}';ratio=f'ewm_ratio_{a}_{b}'
        frame[difference]=cross[:,2*i];frame[ratio]=cross[:,2*i+1];scaled.append(difference)
    additions=calendar_features(base,horizons,calendar['dates'],calendar['holidays'],seed=seed)
    additions.update(state_features(base,horizons,np.asarray(frame['role_0']),calendar['state_codes'],calendar['events']))
    for name,value in additions.items():frame[name]=value
    target=np.full(len(rows),np.nan);known=future<len(history)
    target[known]=history[future[known],series[known]]
    if out_of_sample:target=np.where(np.isnan(target),-1.,target)
    groups=np.array([date.fromisoformat(calendar['dates'][int(d)]).year for d in future])
    numeric=[name for name in frame if not isinstance(frame[name].dtype,pd.CategoricalDtype)]
    result=transform(frame[numeric].to_numpy(),target,population['volumes'][rows],groups,
        [numeric.index(name) for name in scaled],validation_group,out_of_sample=out_of_sample,scale_range=scale_range,seed=seed+1)
    retained=result['rows'];output=frame.iloc[retained].copy().reset_index(drop=True)
    for name in scaled:output[name]=result['features'][:,numeric.index(name)]
    return dict(features=output,targets=result['targets'],groups=result['groups'],scales=result['scales'],
                rows=rows[retained],horizons=horizons[retained],weights=population['weights'][rows[retained]])
