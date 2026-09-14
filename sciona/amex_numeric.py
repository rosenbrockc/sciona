"""Independent numeric denoising and sequence summaries for Amex reconstruction.

Caller supplies ordered numeric sequences, with NaN representing missing values.
No fixed source column names or real records are embedded. Categorical mappings,
rank construction and sequence-model features belong to later components.
"""
import numpy as np


def _matrix(values):
    x=np.asarray(values)
    if x.ndim!=2 or not x.shape[0] or not x.shape[1] or x.dtype.kind not in 'iuf' or np.isinf(x).any():raise ValueError('Nonempty numeric matrix with optional NaN required')
    return x.astype(np.float64,copy=True)


def denoise_numeric(values):
    x=_matrix(values)
    with np.errstate(over='ignore',invalid='ignore'):result=np.floor(x*100.)
    if np.isinf(result).any():raise ValueError('Denoising arithmetic overflow')
    return result


def summarize(values,*,last_window=None,differences=False,ranked=False):
    x=_matrix(values)
    if type(differences) is not bool or type(ranked) is not bool or (ranked and differences):raise ValueError('Valid feature mode required')
    if last_window is not None:
        if type(last_window) is not int or last_window<1:raise ValueError('Positive last-window length required')
        x=x[-last_window:]
    if differences:x=np.vstack((np.full((1,x.shape[1]),np.nan),np.diff(x,axis=0)))
    names=['last'] if ranked else ['mean','std','min','max','sum']+(['last'] if last_window is None else [])
    output=np.full((x.shape[1],len(names)),np.nan)
    for column in range(x.shape[1]):
        v=x[:,column];v=v[~np.isnan(v)]
        if len(v):
            stats=dict(mean=v.mean(),std=v.std(ddof=1) if len(v)>1 else np.nan,min=v.min(),max=v.max(),sum=v.sum(),last=v[-1])
        else:stats=dict(mean=np.nan,std=np.nan,min=np.nan,max=np.nan,sum=0.,last=np.nan)
        output[column]=[stats[n] for n in names]
    if not ranked:
        with np.errstate(over='ignore',invalid='ignore'):output=np.floor_divide(output,.01)
    if np.isinf(output).any():raise ValueError('Aggregate arithmetic overflow')
    return output.reshape(-1)
