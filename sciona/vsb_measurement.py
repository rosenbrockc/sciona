"""Independent VSB three-signal phase and full measurement aggregation.

Input columns form complete consecutive triples. The fundamental is one cycle
per supplied signal; variable length is an explicit independent generalization.
Missing aggregate groups retain NaN for the downstream tree learner.
"""
import numpy as np
from sciona.vsb_descriptors import process_signal


def fundamental(values):
    x=np.asarray(values)
    if x.ndim!=2 or len(x)<3 or not x.shape[1] or x.dtype.kind not in 'iuf' or not np.isfinite(x).all():raise ValueError('Finite sample-by-signal matrix required')
    result=np.fft.rfft(x.astype(np.float64),axis=0)[1]
    if not np.isfinite(result).all():raise ValueError('Fourier arithmetic overflow')
    return result


def phase_quadrants(positions,coefficient,*,length):
    p=np.asarray(positions)
    if type(length) is not int or length<3 or p.ndim!=1 or (p.size and p.dtype.kind not in 'iu') or np.any(p<0) or np.any(p>=length):raise ValueError('Valid phase positions required')
    if not np.isscalar(coefficient) or not np.isfinite(coefficient):raise ValueError('Finite Fourier coefficient required')
    degrees=(np.degrees(2*np.pi*p/length+np.angle(coefficient))+90)%360
    # Right-closed bins exclude exactly zero; -1 denotes the missing quadrant.
    return np.searchsorted([0.,90.,180.,270.,360.],degrees,side='left')-1


def aggregate(groups,heights,descriptors,quadrants,*,measurement_count):
    g=np.asarray(groups);h=np.asarray(heights);f=np.asarray(descriptors);q=np.asarray(quadrants)
    if type(measurement_count) is not int or measurement_count<1:raise ValueError('Positive measurement count required')
    if g.ndim!=1 or (g.size and g.dtype.kind not in 'iu') or np.any(g<0) or np.any(g>=measurement_count):raise ValueError('Valid measurement indices required')
    if h.shape!=g.shape or h.dtype.kind not in 'iuf' or not np.isfinite(h).all() or np.any(h<=0):raise ValueError('Aligned positive peak heights required')
    if f.shape!=(len(g),4) or f.dtype.kind not in 'iuf' or np.isinf(f).any() or not np.isfinite(f[:,2:]).all():raise ValueError('Aligned descriptors required')
    if q.shape!=g.shape or (q.size and q.dtype.kind not in 'iu') or not np.isin(q,[-1,0,1,2,3]).all():raise ValueError('Valid aligned quadrants required')
    keep=~((f[:,0]>.33333)&(h>50))
    result=np.full((measurement_count,9),np.nan)
    for index in range(measurement_count):
        total=(g==index)&keep;first=total&np.isin(q,[0,2]);second=total&np.isin(q,[1,3])
        for col,mask in enumerate((first,total,second)):
            if mask.any():result[index,col]=mask.sum()
        if first.any():
            values=h[first];result[index,3]=values.mean()
            if len(values)>1:result[index,4]=values.std(ddof=1)
            for col,values in enumerate((f[first,1],f[first,0],np.abs(f[first,2]),f[first,3]),start=5):
                finite=values[np.isfinite(values)]
                if len(finite):result[index,col]=finite.mean()
    if np.isinf(result).any():raise ValueError('Aggregate arithmetic overflow')
    return result


def measurement_features(values):
    x=np.asarray(values);coefficients=fundamental(x)
    if x.shape[1]%3:raise ValueError('Complete three-signal measurements required')
    groups=[];heights=[];descriptors=[];quadrants=[]
    for column in range(x.shape[1]):
        peaks=process_signal(x[:,column]);n=len(peaks['positions'])
        groups.extend([column//3]*n);heights.extend(peaks['heights']);descriptors.extend(peaks['descriptors'])
        quadrants.extend(phase_quadrants(peaks['positions'],coefficients[column],length=len(x)))
    return aggregate(np.asarray(groups,dtype=int),np.asarray(heights,dtype=float),np.asarray(descriptors,dtype=float).reshape(-1,4),np.asarray(quadrants,dtype=int),measurement_count=x.shape[1]//3)
