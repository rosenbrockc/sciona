"""Synthetic periodic signals with separate timing and spectral measurements."""
import numpy as np


def periodic_signal_measurements():
    sample_rate=8192.;samples=8192
    times=np.arange(samples)/sample_rate
    measured_periods=[];spectral_frequencies=[]
    for frequency in range(8,264):
        signal=np.sin(2*np.pi*frequency*times+0.37)
        indices=np.flatnonzero((signal[:-1]<=0)&(signal[1:]>0))
        crossings=(indices-signal[indices]/(signal[indices+1]-signal[indices]))/sample_rate
        if len(crossings)<3:raise ValueError('insufficient synthetic cycles')
        measured_periods.append(float(np.mean(np.diff(crossings))))
        power=np.abs(np.fft.rfft(signal))**2
        peak=int(np.argmax(power[1:])+1)
        spectral_frequencies.append(float(np.fft.rfftfreq(samples,1/sample_rate)[peak]))
    return np.asarray(measured_periods),np.asarray(spectral_frequencies)
