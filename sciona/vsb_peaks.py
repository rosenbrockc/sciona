"""Independent reconstruction of VSB's directional peak and knee semantics.

Preserves asymmetric scan advancement, forward/reverse midpoint intersection,
nonconsecutive threshold counting and negative knee slicing. This is not the
usual symmetric maximum filter or a conventional consecutive plateau detector.
"""
import numpy as np
from sciona.vsb_baseline import subtract_baseline


def _signal(values):
    x=np.asarray(values)
    if x.ndim!=1 or not x.size or x.dtype.kind not in 'iuf' or not np.isfinite(x).all():
        raise ValueError('Nonempty finite real signal required')
    return x.astype(np.float64,copy=True)


def _directional(x,window):
    rises=np.flatnonzero(np.diff(x)>0)+1
    boundaries=np.r_[np.flatnonzero(np.diff(x)!=0),len(x)-1]
    accepted=[];cursor=1
    while True:
        candidate_position=np.searchsorted(rises,cursor)
        if candidate_position==len(rises):break
        start=int(rises[candidate_position])
        if start>=len(x)-1:break
        end=min(int(boundaries[np.searchsorted(boundaries,start)]),len(x)-2)
        stop=min(end+window,len(x)-1)
        higher=np.flatnonzero(x[end+1:stop]>x[start])
        cursor=int(end+1+higher[0]) if higher.size else stop
        if x[cursor]<x[start]:accepted.append((start+end)//2)
    return np.asarray(accepted,dtype=np.int64)


def window_maxima(values,*,window=25):
    x=_signal(values)
    if type(window) is not int or window<1:raise ValueError('Positive integer window required')
    forward=_directional(x,window)
    reverse=len(x)-1-_directional(x[::-1],window)
    return np.intersect1d(forward,reverse)


def plateau_location(gradient,*,threshold=-.01,count=1000):
    g=np.asarray(gradient)
    if g.ndim!=1 or g.dtype.kind not in 'iuf' or not np.isfinite(g).all():raise ValueError('Finite real gradient required')
    if type(count) is not int or count<1:raise ValueError('Positive count required')
    if isinstance(threshold,(bool,np.bool_)) or not isinstance(threshold,(int,float,np.integer,np.floating)) or not np.isfinite(threshold):raise ValueError('Finite threshold required')
    hits=np.flatnonzero(g>threshold)
    return int(hits[count-1]-count) if len(hits)>=count else 0


def select_peaks(values,*,window=25):
    residual=subtract_baseline(values)
    positions=window_maxima(np.abs(residual),window=window)
    if len(positions)<2:raise ValueError('At least two candidate peaks required for source knee calculation')
    heights=np.abs(residual[positions]);order=np.argsort(heights)[::-1]
    positions=positions[order];heights=heights[order]
    gradient=np.convolve(np.diff(heights),np.ones(9)/9)[8:-8]
    knee=plateau_location(gradient)-4
    selected=positions[:knee];amplitudes=heights[:knee]
    chronological=np.argsort(selected)
    return dict(positions=selected[chronological],heights=amplitudes[chronological],residual=residual,knee=knee,candidate_count=len(positions))
