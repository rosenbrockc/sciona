"""Independent Amex sequence encoding, prediction slots and neural batch packing.

Inputs use already denoised numbers and mapped categories. Joint vocabulary is
caller population dependent. Prediction slots and neural sequences intentionally
use opposite padding directions, matching the source's distinct consumers.
"""
import numpy as np
from sciona.amex_numeric import _matrix


def encode_series(numerics,categories):
    if type(numerics) is not list or not numerics or len(numerics)!=len(categories):raise ValueError('Aligned populations required')
    xs=[_matrix(x) for x in numerics];cs=[_matrix(c) for c in categories]
    nw=xs[0].shape[1];cw=cs[0].shape[1]
    if any(x.shape[1]!=nw or c.shape!=(len(x),cw) for x,c in zip(xs,cs)):raise ValueError('Aligned sequence feature widths required')
    population=np.vstack(cs);vocab=[np.unique(population[~np.isnan(population[:,j]),j]) for j in range(cw)]
    encoded=[]
    for x,c in zip(xs,cs):
        channels=[x]+[(c[:,j,None]==v).astype(float) for j,v in enumerate(vocab)]
        values=np.concatenate(channels,axis=1)/100.
        values[np.isnan(values)]=0.
        encoded.append(values)
    return dict(sequences=encoded,vocabulary=vocab)


def prediction_slots(sequences):
    if type(sequences) is not list or not sequences:raise ValueError('Prediction sequence population required')
    out=np.full((len(sequences),13),np.nan)
    for i,values in enumerate(sequences):
        p=np.asarray(values)
        if p.ndim!=1 or not 1<=len(p)<=13 or p.dtype.kind not in 'iuf' or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('One through thirteen finite probabilities required')
        out[i,-len(p):]=p
    return out


def pack_neural(sequences,features):
    if type(sequences) is not list or not sequences:raise ValueError('Sequence population required')
    xs=[_matrix(x) for x in sequences];f=_matrix(features)
    if len(f)!=len(xs) or any(len(x)>13 or x.shape[1]!=xs[0].shape[1] or not np.isfinite(x).all() for x in xs) or not np.isfinite(f).all():raise ValueError('Aligned finite neural inputs required')
    series=np.zeros((len(xs),13,xs[0].shape[1]),np.float32);mask=np.zeros((len(xs),13),np.float32)
    for i,x in enumerate(xs):series[i,:len(x)]=x;mask[i,:len(x)]=1
    complement=np.where(f!=0,1.-f+.001,0.)
    packed=np.concatenate((f,complement),axis=1).astype(np.float32)
    if not np.isfinite(series).all() or not np.isfinite(packed).all():raise ValueError('Float32 packing overflow')
    return dict(series=series,mask=mask,features=packed)
