"""Independent randomized volume scaling and final target/group filtering."""
import numpy as np


def transform(features,targets,volumes,groups,scaled_columns,validation_group,*,out_of_sample=False,scale_range=.1,seed=0):
    features,targets,volumes,groups=[np.asarray(v) for v in (features,targets,volumes,groups)]
    if (features.ndim!=2 or not all(features.shape) or features.dtype.kind not in 'iuf'
            or any(v.shape!=(len(features),) for v in (targets,volumes,groups))
            or targets.dtype.kind not in 'iuf' or volumes.dtype.kind not in 'iuf' or groups.dtype.kind not in 'iu'
            or np.isinf(features).any() or np.isinf(targets).any()
            or not np.isfinite(volumes).all() or (volumes<=0).any()):
        raise ValueError('Aligned real features/targets, positive volumes and integer groups required')
    if (type(out_of_sample) is not bool or type(validation_group) is not int
            or type(seed) is not int or not 0<=seed<2**32
            or isinstance(scale_range,bool) or not isinstance(scale_range,(int,float))
            or not np.isfinite(scale_range) or scale_range<0):
        raise ValueError('Invalid scaling controls')
    if (not isinstance(scaled_columns,(list,tuple)) or len(set(scaled_columns))!=len(scaled_columns)
            or any(type(i) is not int or not 0<=i<features.shape[1] for i in scaled_columns)):
        raise ValueError('Distinct scaled feature indices required')
    scales=volumes.astype(float).copy()
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        if scale_range>0:
            scales*=np.exp(scale_range*np.random.RandomState(seed).normal(0,.5,len(features)))
    if not np.isfinite(scales).all() or (scales<=0).any():raise ValueError('Random scale is outside finite positive range')
    result=features.astype(float).copy()
    with np.errstate(over='ignore',invalid='ignore'):
        for column in scaled_columns:
            values=result[:,column]
            result[:,column]=np.where(values==-10,values,values/scales)
        y=targets.astype(float)/scales
    if np.isinf(result).any() or np.isinf(y).any():raise ValueError('Scaled values overflow')
    keep=np.ones(len(features),dtype=bool) if out_of_sample else (~np.isnan(y)&(groups!=validation_group))
    rows=np.flatnonzero(keep)
    return dict(features=result[keep],targets=y[keep],groups=groups[keep].copy(),scales=scales[keep],rows=rows)
