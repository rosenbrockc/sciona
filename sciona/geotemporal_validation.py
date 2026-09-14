"""Expanding forward time blocks with an explicit exclusion gap."""
import math
import numpy as np


def forward_splits(times,blocks,*,gap=0.):
    if type(times) is not list or not times or any(type(t) not in (int,float) or not math.isfinite(t) for t in times):raise ValueError('Finite times required')
    if type(blocks) is not list or len(blocks)!=len(times) or any(type(b) is not int or b<0 for b in blocks):raise ValueError('Aligned nonnegative integer blocks required')
    if type(gap) not in (int,float) or not math.isfinite(gap) or gap<0:raise ValueError('Nonnegative gap required')
    unique=sorted(set(blocks))
    if len(unique)<2 or unique!=list(range(len(unique))):raise ValueError('Contiguous time blocks starting at zero required')
    times=np.asarray(times,dtype=np.float64);blocks=np.asarray(blocks)
    for left,right in zip(unique[:-1],unique[1:]):
        if times[blocks==left].max()>=times[blocks==right].min():raise ValueError('Blocks must be strictly ordered in time')
    splits=[]
    for block in unique[1:]:
        valid=np.flatnonzero(blocks==block)
        train=np.flatnonzero((blocks<block)&(times<times[valid].min()-gap))
        if not len(train):raise ValueError('Gap leaves an empty fitting population')
        splits.append((train,valid))
    return splits
