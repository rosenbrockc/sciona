"""Independent waveform/spectrogram branches and bounded spectral masking."""
import torch
from torch import nn


class DualSignalHead(nn.Module):
    def __init__(self,channels,frequency_bins,width=8):
        super().__init__()
        if any(type(v) is not int or v<1 for v in (channels,frequency_bins,width)):raise ValueError('Positive architecture dimensions required')
        self.channels=channels;self.frequency_bins=frequency_bins
        self.waveform=nn.Sequential(nn.Conv1d(channels,width,5,padding=2),nn.GELU(),nn.Conv1d(width,width,3,padding=1),nn.GELU())
        self.spectral=nn.Sequential(nn.Conv2d(channels,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU())
        # Average only over time so absolute frequency position remains available.
        self.classifier=nn.Linear(width+width*frequency_bins,1)

    def forward(self,waveform,spectrogram):
        if waveform.ndim!=3 or spectrogram.ndim!=4 or waveform.shape[:2]!=spectrogram.shape[:2] or waveform.shape[1]!=self.channels or spectrogram.shape[2]!=self.frequency_bins:
            raise ValueError('Aligned batch/channel waveform and spectrogram required')
        if not all(waveform.shape) or not all(spectrogram.shape) or not torch.isfinite(waveform).all() or not torch.isfinite(spectrogram).all():raise ValueError('Finite nonempty branches required')
        wave=self.waveform(waveform).mean(-1)
        spectral=self.spectral(spectrogram).mean(-1).flatten(1)
        return self.classifier(torch.cat((wave,spectral),dim=1)).squeeze(-1)


def mask_spectrogram(spectrogram,*,max_frequency,max_time,generator):
    if spectrogram.ndim!=4 or not all(spectrogram.shape) or not torch.isfinite(spectrogram).all():raise ValueError('Finite batch/channel/frequency/time array required')
    for value,size in ((max_frequency,spectrogram.shape[2]),(max_time,spectrogram.shape[3])):
        if type(value) is not int or not 0<=value<size:raise ValueError('Mask maximum must leave at least one bin')
    result=spectrogram.clone()
    for row in result:
        for axis,maximum in ((1,max_frequency),(2,max_time)):
            width=int(torch.randint(maximum+1,(),generator=generator))
            if width:
                start=int(torch.randint(row.shape[axis]-width+1,(),generator=generator))
                slices=[slice(None)]*3;slices[axis]=slice(start,start+width)
                row[tuple(slices)]=0
    return result
