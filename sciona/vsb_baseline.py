"""Independent floating-point realization of recursive baseline subtraction.

For retention a, the baseline obeys b[t]=a*b[t-1]+(1-a)*x[t],
with b[0]=x[0]. The returned residual is x-b. This is a first-order
high-pass filter, implemented through SciPy's transfer-function interface.
"""
import numpy as np
from scipy.signal import lfilter


def subtract_baseline(values,*,retention=.99):
    x=np.asarray(values)
    if x.ndim!=1 or not x.size or x.dtype.kind not in 'iuf' or not np.isfinite(x).all():
        raise ValueError('A nonempty finite real signal is required')
    if isinstance(retention,(bool,np.bool_)) or not isinstance(retention,(int,float,np.integer,np.floating)) or not np.isfinite(retention) or not 0<=retention<=1:
        raise ValueError('Retention must lie between zero and one')
    x=x.astype(np.float64,copy=True);a=float(retention)
    if a==0:return np.zeros_like(x)
    residual,_=lfilter([a,-a],[1.,-a],x,zi=[-a*x[0]])
    residual[0]=0.
    if not np.isfinite(residual).all():raise ValueError('Baseline arithmetic overflow')
    return residual
