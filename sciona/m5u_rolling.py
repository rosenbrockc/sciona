"""Independent full-window history moments and quantiles for uncertainty features."""
import numpy as np


def statistic(values,window,kind):
    values=np.asarray(values)
    if (values.ndim!=2 or not all(values.shape) or values.dtype.kind not in 'iuf'
            or np.isinf(values).any() or type(window) is not int or window<1):
        raise ValueError('Real time-by-series matrix and positive integer window required')
    if kind not in ('mean','median','std','skew','kurt','q10','q90'):raise ValueError('Unknown statistic')
    out=np.full(values.shape,np.nan)
    for end in range(window-1,len(values)):
        sample=values[end-window+1:end+1].astype(float)
        valid=~np.isnan(sample).any(axis=0)
        x=sample[:,valid]
        if not x.size:continue
        if kind=='mean':result=np.mean(x,axis=0)
        elif kind=='median':result=np.median(x,axis=0)
        elif kind in ('q10','q90'):result=np.quantile(x,.1 if kind=='q10' else .9,axis=0)
        elif kind=='std':result=np.std(x,axis=0,ddof=1) if window>1 else np.full(x.shape[1],np.nan)
        else:
            n=window;centered=x-x.mean(axis=0);m2=np.mean(centered**2,axis=0)
            with np.errstate(divide='ignore',invalid='ignore'):
                if kind=='skew':
                    result=np.sqrt(n*(n-1))/(n-2)*np.mean(centered**3,axis=0)/m2**1.5 if n>2 else np.full(x.shape[1],np.nan)
                    if n>2:result=np.where(m2==0,0.,result)
                else:
                    result=((n*n-1)*np.mean(centered**4,axis=0)/m2**2-3*(n-1)**2)/((n-2)*(n-3)) if n>3 else np.full(x.shape[1],np.nan)
                    if n>3:result=np.where(m2==0,-3.,result)
        out[end,valid]=result
    return out


def blocks(raw,scaled,*,reduced=False):
    from sciona.m5u_exponential import histories
    raw,scaled=np.asarray(raw),np.asarray(scaled)
    if raw.shape!=scaled.shape or type(reduced) is not bool:raise ValueError('Aligned raw/scaled history required')
    result={f'raw_ewm_{w}':v for w,v in histories(raw,reduced=reduced).items()}
    if not reduced:result.update({f'scaled_ewm_{w}':v for w,v in histories(scaled).items()})
    for w in ((28,) if reduced else (7,14,28,56,112)):
        result[f'nonzero_{w}']=statistic((np.nan_to_num(raw,nan=0)!=0).astype(float),w,'mean')
    if not reduced:
        for lag in range(1,11):
            shifted=np.zeros(raw.shape,dtype=float)
            if lag<len(raw):shifted[lag:]=np.nan_to_num(raw[:-lag],nan=0)
            result[f'lag_{lag}']=shifted
    populations=[('raw',raw)] if reduced else [('raw',raw),('scaled',scaled)]
    for name,values in populations:
        for w in ((28,) if reduced else (7,14,21,28,56,112)):
            for kind in ('mean','median'):result[f'{name}_{kind}_{w}']=statistic(values,w,kind)
    for name,values in populations:
        for w in ((28,) if reduced else (7,14,28,84,168)):
            result[f'{name}_std_{w}']=statistic(values,w,'std')
            if w>=10 and not reduced:
                for kind in ('skew','kurt'):result[f'{name}_{kind}_{w}']=statistic(values,w,kind)
    for name,values in populations:
        for w in ((28,) if reduced else (14,28,56)):
            for kind in ('q10','q90'):result[f'{name}_{kind}_{w}']=statistic(values,w,kind)
    return result
