"""Recording-balanced scaling and class/recording-balanced signal training."""
from dataclasses import dataclass
import numpy as np
import torch
from sciona.biosignal_sequence_features import window_signal,psd_spectrogram
from sciona.biosignal_sequence_network import DualSignalHead,mask_spectrogram


def features(signals,controls):
    if type(signals) is not list or not signals:raise ValueError('Nonempty recording population required')
    waves=[];spectra=[];counts=[];tails=[];channels=None
    for signal in signals:
        wave,_,tail=window_signal(signal,window_size=controls['window_size'],stride=controls['stride'])
        if channels is None:channels=wave.shape[1]
        if channels!=wave.shape[1]:raise ValueError('Recording channel counts differ')
        spectrum,_,_=psd_spectrogram(wave,sample_rate=controls['sample_rate'],n_fft=controls['n_fft'],hop=controls['hop'])
        waves.append(wave);spectra.append(spectrum);counts.append(len(wave));tails.append(tail)
    return np.concatenate(waves),np.concatenate(spectra),counts,tails


def window_weights(labels,counts):
    if type(labels) is not list or len(labels)!=len(counts) or any(type(v) is not int or v not in (0,1) for v in labels) or set(labels)!={0,1}:raise ValueError('Aligned recording binary labels with both classes required')
    if any(type(n) is not int or n<1 for n in counts):raise ValueError('Positive recording window counts required')
    return np.concatenate([np.full(n,1/(2*labels.count(label)*n)) for label,n in zip(labels,counts)])


@dataclass
class Scaling:
    wave_mean: object
    wave_scale: object
    spectral_mean: object
    spectral_scale: object

    @classmethod
    def fit(cls,wave,spectral,counts):
        weights=np.concatenate([np.full(n,1/(len(counts)*n)) for n in counts])
        def moments(x):
            axes=tuple(range(2,x.ndim));shape=(1,x.shape[1])+((1,)*(x.ndim-2))
            mean=np.sum(x.mean(axis=axes)*weights[:,None],axis=0).reshape(shape)
            variance=np.sum(((x-mean)**2).mean(axis=axes)*weights[:,None],axis=0).reshape(shape)
            scale=np.sqrt(variance);scale=np.where(scale>1e-12,scale,1.)
            if not np.isfinite(mean).all() or not np.isfinite(scale).all():raise ValueError('Nonfinite fitted scaling')
            return mean,scale
        return cls(*moments(wave),*moments(spectral))

    def transform(self,wave,spectral):
        if wave.shape[1]!=self.wave_mean.shape[1] or spectral.shape[1]!=self.spectral_mean.shape[1]:raise ValueError('Fitted channel counts differ')
        values=((wave-self.wave_mean)/self.wave_scale,(spectral-self.spectral_mean)/self.spectral_scale)
        if any(not np.isfinite(x).all() for x in values):raise ValueError('Nonfinite normalized features')
        return tuple(torch.tensor(x,dtype=torch.float64) for x in values)


@dataclass
class SignalModel:
    network: object
    scaling: Scaling
    controls: dict
    history: list
    counts: list

    def predict(self,signals):
        wave,spectral,counts,tails=features(signals,self.controls)
        x,s=self.scaling.transform(wave,spectral);values=[]
        self.network.eval()
        with torch.no_grad():
            for start in range(0,len(x),self.controls['batch_size']):
                end=start+self.controls['batch_size']
                values.extend(torch.sigmoid(self.network(x[start:end],s[start:end])).tolist())
        values=np.asarray(values)
        if not np.isfinite(values).all():raise ValueError('Nonfinite prediction')
        offsets=np.cumsum([0]+counts)
        return [float(values[a:b].mean()) for a,b in zip(offsets[:-1],offsets[1:])],counts,tails


def fit(signals,labels,controls):
    c=dict(controls)
    for name in ('seed','width','epochs','batch_size'):
        if type(c[name]) is not int or c[name]<(0 if name=='seed' else 1):raise ValueError('Invalid integer training control')
    if c['seed']>=2**32:raise ValueError('Invalid seed')
    if type(c['learning_rate']) not in (int,float) or not np.isfinite(c['learning_rate']) or c['learning_rate']<=0:raise ValueError('Invalid learning rate')
    wave,spectral,counts,_=features(signals,c)
    weights=torch.tensor(window_weights(labels,counts),dtype=torch.float64)
    scaling=Scaling.fit(wave,spectral,counts);x,s=scaling.transform(wave,spectral)
    target=torch.tensor(np.repeat(labels,counts),dtype=torch.float64)
    generator=torch.Generator().manual_seed(c['seed'])
    # Validate augmentation controls before model fitting.
    mask_spectrogram(s[:1],max_frequency=c['max_frequency'],max_time=c['max_time'],generator=torch.Generator())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed']);model=DualSignalHead(x.shape[1],s.shape[2],c['width']).double()
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate']);history=[]
    for epoch in range(c['epochs']+1):
        model.eval();total=0.
        with torch.no_grad():
            for start in range(0,len(x),c['batch_size']):
                end=start+c['batch_size']
                losses=torch.nn.functional.binary_cross_entropy_with_logits(model(x[start:end],s[start:end]),target[start:end],reduction='none')
                total+=float((losses*weights[start:end]).sum())
        if not np.isfinite(total):raise ValueError('Nonfinite loss')
        history.append(total)
        if epoch==c['epochs']:break
        model.train();optimizer.zero_grad(set_to_none=True)
        # Accumulation preserves global per-recording/class weights regardless
        # of unequal final batch size; one optimizer update per complete epoch.
        for start in range(0,len(x),c['batch_size']):
            end=start+c['batch_size']
            augmented=mask_spectrogram(s[start:end],max_frequency=c['max_frequency'],max_time=c['max_time'],generator=generator)
            losses=torch.nn.functional.binary_cross_entropy_with_logits(model(x[start:end],augmented),target[start:end],reduction='none')
            (losses*weights[start:end]).sum().backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Invalid training gradients')
        optimizer.step()
    model.eval()
    return SignalModel(model,scaling,c,history,counts)
