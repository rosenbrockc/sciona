"""Independent reconstruction of source Amex greedy histogram boundaries.

The compressed branch deliberately discards its last provisional split before
adding infinity, matching source behavior. Missing values digitize to zero;
positive bin codes are divided by the maximum code in the joint population.
"""
import numpy as np
from sciona.amex_numeric import _matrix


def boundaries(values,counts,*,max_bins=255,min_count=3):
    v=np.asarray(values);c=np.asarray(counts)
    if v.ndim!=1 or not len(v) or v.dtype.kind not in 'iuf' or not np.isfinite(v).all() or np.any(np.diff(v)<=0):raise ValueError('Sorted distinct finite values required')
    if c.shape!=v.shape or c.dtype.kind not in 'iu' or np.any(c<=0):raise ValueError('Positive integer histogram counts required')
    if type(max_bins) is not int or max_bins<1 or type(min_count) is not int or min_count<0:raise ValueError('Valid bin controls required')
    v=v.astype(float);c=[int(x) for x in c];splits=[]
    if len(v)<=max_bins:
        mass=0
        for i in range(len(v)-1):
            mass+=c[i]
            if mass>=min_count:splits.append(i);mass=0
    else:
        total=sum(c);budget=max(1,min(max_bins,total//min_count)) if min_count else max_bins
        heavy=np.asarray(c)>=total/budget
        slots=budget-int(heavy.sum());remaining=sum(n for n,big in zip(c,heavy) if not big)
        if slots<=0:raise ValueError('Undefined source residual bin budget')
        target=remaining/slots;mass=0
        for i in range(len(v)-1):
            if not heavy[i]:remaining-=c[i]
            mass+=c[i]
            close=heavy[i] or mass>=target or (heavy[i+1] and mass>=max(1.,target*.5))
            if close:
                splits.append(i)
                if len(splits)>=budget-1:break
                mass=0
                if not heavy[i]:
                    slots-=1
                    if slots<=0:raise ValueError('Undefined source residual bin budget')
                    target=remaining/slots
        splits=splits[:-1]
    with np.errstate(over='ignore'):cuts=np.asarray([(v[i]+v[i+1])/2 for i in splits],float)
    if not np.isfinite(cuts).all():raise ValueError('Boundary arithmetic overflow')
    return np.r_[cuts,np.inf]


def normalize(values,*,max_bins=255):
    x=_matrix(values);out=np.empty_like(x);learned=[]
    for j in range(x.shape[1]):
        valid=x[~np.isnan(x[:,j]),j];v,c=np.unique(valid,return_counts=True)
        if not len(v):raise ValueError('All-missing feature has undefined source histogram')
        cuts=boundaries(v,c,max_bins=max_bins);code=np.digitize(x[:,j],np.r_[-np.inf,cuts])
        code[code==len(cuts)+1]=0
        out[:,j]=code/code.max();learned.append(cuts)
    return dict(values=out,boundaries=learned)
