"""Independent fixed-window and one-sided PSD features for synthetic qualification.

Arrays are channel/time. Windows never cross recording boundaries. The caller
must split subjects before invoking feature/training populations.
"""
import math
import numpy as np


def positive_int(value):
    if type(value) is not int or value<1:raise ValueError('Positive integer required')
    return value


def signal_array(signal):
    if type(signal) is not list or not signal or any(type(row) is not list or not row for row in signal):raise ValueError('Nonempty channel/time lists required')
    if len({len(row) for row in signal})!=1 or any(type(v) not in (int,float) for row in signal for v in row):raise ValueError('Rectangular numeric signal required')
    try:values=np.asarray(signal,dtype=np.float64)
    except (ValueError,OverflowError) as error:raise ValueError('Invalid numeric signal') from error
    if not np.isfinite(values).all():raise ValueError('Finite signal required')
    return values


def window_signal(signal,*,window_size,stride):
    values=signal_array(signal);positive_int(window_size);positive_int(stride)
    if stride>window_size:raise ValueError('Stride must not leave gaps')
    if values.shape[1]<window_size:raise ValueError('Recording shorter than one full window')
    starts=list(range(0,values.shape[1]-window_size+1,stride))
    windows=np.stack([values[:,start:start+window_size] for start in starts])
    return windows,starts,values.shape[1]-(starts[-1]+window_size)


def psd_spectrogram(windows,*,sample_rate,n_fft,hop):
    """Periodic Hann, segment-mean detrending, density scaling, log1p power.

Returns log PSD, frequencies and segment-center times. No cross-population fitted
statistics. Trailing samples insufficient for a full FFT segment are omitted.
"""
    x=np.asarray(windows,dtype=np.float64)
    if x.ndim!=3 or not all(x.shape) or not np.isfinite(x).all():raise ValueError('Finite window/channel/time array required')
    if type(sample_rate) not in (int,float) or not math.isfinite(sample_rate) or sample_rate<=0:raise ValueError('Positive sample rate required')
    positive_int(n_fft);positive_int(hop)
    if n_fft<4 or n_fft>x.shape[-1] or hop>n_fft:raise ValueError('Invalid FFT segment controls')
    taper=.5-.5*np.cos(2*np.pi*np.arange(n_fft)/n_fft)
    energy=np.sum(taper*taper);pieces=[]
    starts=list(range(0,x.shape[-1]-n_fft+1,hop))
    for start in starts:
        segment=x[:,:,start:start+n_fft]
        centered=segment-segment.mean(axis=-1,keepdims=True)
        transformed=np.fft.rfft(centered*taper,axis=-1)
        power=(transformed.real**2+transformed.imag**2)/(sample_rate*energy)
        power[:,:,1:-1 if n_fft%2==0 else None]*=2
        pieces.append(np.log1p(power))
    result=np.stack(pieces,axis=-1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite PSD')
    return result,np.fft.rfftfreq(n_fft,d=1/sample_rate),(np.asarray(starts)+n_fft/2)/sample_rate
