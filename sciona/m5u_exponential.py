"""Independent exponentially weighted history and pairwise feature crosses."""
import numpy as np


def mean(values,span,*,minimum=0,adjust=True):
    """EWM down the time axis; missing observations retain absolute-position decay."""
    values=np.asarray(values)
    if (values.ndim!=2 or not all(values.shape) or values.dtype.kind not in 'iuf'
            or np.isinf(values).any()):raise ValueError('Nonempty real time-by-series matrix required')
    if type(span) is not int or span<1 or type(minimum) is not int or minimum<0 or type(adjust) is not bool:
        raise ValueError('Invalid exponential mean controls')
    decay=1-2/(span+1);new_weight=1. if adjust else 1-decay
    output=np.full(values.shape,np.nan);average=np.full(values.shape[1],np.nan)
    weight=np.ones(values.shape[1]);count=np.zeros(values.shape[1],dtype=int)
    for row,current in enumerate(values.astype(float)):
        observed=~np.isnan(current);count+=observed
        established=~np.isnan(average)
        weight[established]*=decay
        update=established&observed
        # Avoid unnecessary cancellation when the observations are identical.
        changed=update&(average!=current)
        average[changed]=(weight[changed]*average[changed]+new_weight*current[changed])/(weight[changed]+new_weight)
        if adjust:weight[update]+=new_weight
        else:weight[update]=1
        average[~established&observed]=current[~established&observed]
        enough=count>=max(1,minimum)
        output[row,enough]=average[enough]
    return output


def histories(values,*,reduced=False):
    if type(reduced) is not bool:raise ValueError('Boolean reduced feature flag required')
    return {span:mean(values,span,minimum=int(np.ceil(span**.8)))
            for span in (3,7,15,30,100) if not reduced or span>=15}


def crosses(values):
    """For each ordered column pair, append difference then raw ratio."""
    values=np.asarray(values)
    if values.ndim!=2 or not values.shape[0] or values.dtype.kind not in 'iuf' or np.isinf(values).any():
        raise ValueError('Real row-by-feature matrix required')
    columns=[];pairs=[]
    with np.errstate(divide='ignore',invalid='ignore'):
        for first in range(values.shape[1]):
            for second in range(first+1,values.shape[1]):
                a,b=values[:,first].astype(float),values[:,second].astype(float)
                columns.extend((a-b,a/b));pairs.append((first,second))
    return (np.column_stack(columns) if columns else np.empty((len(values),0))),pairs
