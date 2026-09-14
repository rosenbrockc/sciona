"""Independent VSB waveform descriptors with an explicit variable-length boundary.

Neighbor ratios, signed opposite-extremum offset, and sawtooth MSE preserve
source mathematics. Windows clip to the actual input length rather than the
source's fixed length. Missing adjacent samples retain NaN for later handling.
"""
import numpy as np
from sciona.vsb_peaks import _signal,select_peaks


def peak_descriptors(positions,residual,*,small_radius=5,large_radius=25):
    x=_signal(residual);p=np.asarray(positions)
    if p.ndim!=1 or (p.size and p.dtype.kind not in 'iu') or np.any(p<0) or np.any(p>=len(x)):
        raise ValueError('In-range integer peak positions required')
    for radius in (small_radius,large_radius):
        if type(radius) is not int or radius<1:raise ValueError('Positive integer radii required')
    result=np.full((len(p),4),np.nan,dtype=np.float64)
    for row,position in enumerate(p):
        index=int(position);height=x[index]
        if height==0:raise ValueError('Nonzero peak amplitude required')
        for column,neighbor in enumerate((index+1,index-1)):
            if 0<=neighbor<len(x):result[row,column]=abs(x[neighbor]/height)
        near=x[max(0,index-small_radius):min(len(x),index+small_radius+1)]
        result[row,2]=np.argmin(near*np.sign(height))-small_radius
        start=max(0,index-large_radius);end=min(len(x),index+large_radius+1)
        normalized=x[start:end]/height
        relative=np.arange(start,end)-index
        template=np.where((relative>=0)&(relative<=3),1.-2.*relative/3.,0.)
        if np.argmax(normalized)!=np.argmax(template):raise ValueError('Peak and sawtooth maximum must align')
        result[row,3]=np.mean(np.square(normalized-template))
    if np.isinf(result).any():raise ValueError('Descriptor arithmetic overflow')
    return result


def process_signal(values,*,window=25):
    peaks=select_peaks(values,window=window)
    return dict(positions=peaks['positions'],heights=peaks['heights'],
        descriptors=peak_descriptors(peaks['positions'],peaks['residual']))
