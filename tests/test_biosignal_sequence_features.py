import numpy as np
import pytest
from sciona.biosignal_sequence_features import window_signal,psd_spectrogram


def test_window_alignment_and_tail():
    windows,starts,tail=window_signal([list(range(11)),list(range(20,31))],window_size=4,stride=3)
    assert starts==[0,3,6] and tail==1
    np.testing.assert_array_equal(windows[1],[[3,4,5,6],[23,24,25,26]])
    windows[0,0,0]=999
    assert windows[1,0,0]==3

@pytest.mark.parametrize('n_fft',[7,8])
def test_explicit_dft_density_and_parseval(n_fft):
    rng=np.random.default_rng(3);signal=rng.normal(size=(1,1,n_fft))
    actual,freq,times=psd_spectrogram(signal,sample_rate=32.,n_fft=n_fft,hop=2)
    taper=.5-.5*np.cos(2*np.pi*np.arange(n_fft)/n_fft)
    centered=signal[0,0]-signal.mean();expected=[]
    for k in range(n_fft//2+1):
        coefficient=sum(centered[j]*taper[j]*np.exp(-2j*np.pi*k*j/n_fft) for j in range(n_fft))
        factor=1 if k==0 or (n_fft%2==0 and k==n_fft//2) else 2
        expected.append(factor*abs(coefficient)**2/(32*np.sum(taper**2)))
    np.testing.assert_allclose(np.expm1(actual[0,0,:,0]),expected,rtol=1e-12,atol=1e-14)
    assert sum(expected)*32/n_fft==pytest.approx(np.sum((centered*taper)**2)/np.sum(taper**2))
    np.testing.assert_allclose(freq,np.arange(n_fft//2+1)*32/n_fft)
    np.testing.assert_allclose(times,[n_fft/64])


def test_constant_offset_and_population_independence():
    x=np.arange(32,dtype=float).reshape(1,1,32)
    first=psd_spectrogram(x,sample_rate=64.,n_fft=8,hop=4)[0]
    np.testing.assert_allclose(first,psd_spectrogram(x+100,sample_rate=64.,n_fft=8,hop=4)[0])
    batch=np.concatenate([x,x*100])
    np.testing.assert_array_equal(first,psd_spectrogram(batch,sample_rate=64.,n_fft=8,hop=4)[0][:1])
    assert first.shape==(1,1,5,7)

@pytest.mark.parametrize('signal',[[[True]],[[float('nan')]],[[1,2],[3]],[]])
def test_bad_signal(signal):
    with pytest.raises(ValueError):window_signal(signal,window_size=4,stride=2)


def test_invalid_fft_controls():
    with pytest.raises(ValueError):psd_spectrogram(np.ones((1,1,8)),sample_rate=0,n_fft=8,hop=4)
    with pytest.raises(ValueError):psd_spectrogram(np.ones((1,1,8)),sample_rate=32,n_fft=9,hop=4)
