"""Independent CPU swap-denoising autoencoder for pending Porto reconstruction.

Consumes an already-prepared, declared unlabeled population. Linear decoder and
clean-target MSE follow the source. SGD settings, initialization and extraction
layers are explicit; no historical CUDA/optimizer equivalence is asserted.
"""
from dataclasses import dataclass,field
import numpy as np
import torch
from sciona.porto_transforms import _matrix,swap_noise


class Autoencoder(torch.nn.Module):
    def __init__(self,width,hidden):
        super().__init__()
        self.layers=torch.nn.ModuleList(torch.nn.Linear(a,b) for a,b in zip([width]+hidden[:-1],hidden))
        self.output=torch.nn.Linear(hidden[-1],width)

    def activations(self,x):
        values=[]
        for layer in self.layers:
            x=torch.relu(layer(x));values.append(x)
        return values

    def forward(self,x):return self.output(self.activations(x)[-1])


@dataclass(frozen=True,repr=False)
class FittedDAE:
    model: Autoencoder=field(repr=False)
    width: int
    feature_layers: tuple
    clean_mse: tuple
    noisy_training_mse: tuple

    def _input(self,values):
        x=_matrix(values)
        if x.shape[1]!=self.width:raise ValueError('Feature width differs from fitted autoencoder')
        with np.errstate(over='ignore'):x=x.astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite float32 autoencoder input')
        return torch.from_numpy(x)

    def transform(self,values):
        self.model.eval()
        with torch.no_grad():
            outputs=self.model.activations(self._input(values))
            result=torch.cat([outputs[i] for i in self.feature_layers],dim=1).numpy().copy()
        if not np.isfinite(result).all():raise ValueError('Nonfinite learned features')
        return result

    def reconstruct(self,values):
        self.model.eval()
        with torch.no_grad():result=self.model(self._input(values)).numpy().copy()
        if not np.isfinite(result).all():raise ValueError('Nonfinite reconstruction')
        return result


def fit_population(values,*,seed,controls):
    fields={'hidden','feature_layers','epochs','batch_size','learning_rate','decay','swap_probability','momentum'}
    if type(controls) is not dict or set(controls)!=fields:raise ValueError('Explicit autoencoder controls required')
    hidden=controls['hidden'];layers=controls['feature_layers']
    if type(hidden) is not list or not hidden or any(type(n) is not int or n<1 for n in hidden):raise ValueError('Positive hidden widths required')
    if type(layers) is not list or not layers or any(type(i) is not int or not 0<=i<len(hidden) for i in layers) or len(set(layers))!=len(layers):raise ValueError('Distinct explicit extraction layers required')
    for key in ('epochs','batch_size'):
        if type(controls[key]) is not int or controls[key]<1:raise ValueError('Positive integer training controls required')
    for key in ('learning_rate','decay'):
        if type(controls[key]) not in (int,float) or not np.isfinite(controls[key]) or not 0<controls[key]<=1:raise ValueError('Positive bounded rate and decay required')
    for key in ('swap_probability','momentum'):
        if type(controls[key]) not in (int,float) or not np.isfinite(controls[key]) or not 0<=controls[key]<1:raise ValueError('Noise and momentum must be in [0,1)')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Bounded integer seed required')
    x=_matrix(values)
    with np.errstate(over='ignore'):x=x.astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite float32 population')
    rng=np.random.default_rng(seed);clean=torch.from_numpy(x);history=[];noisy_history=[]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=Autoencoder(x.shape[1],hidden)
        optimizer=torch.optim.SGD(model.parameters(),lr=controls['learning_rate'],momentum=controls['momentum'])
        with torch.no_grad():history.append(float(torch.nn.functional.mse_loss(model(clean),clean)))
        for epoch in range(controls['epochs']):
            model.train();order=rng.permutation(len(x));total=0.
            for start in range(0,len(x),controls['batch_size']):
                rows=order[start:start+controls['batch_size']]
                noisy=swap_noise(x[rows],x,probability=controls['swap_probability'],rng=rng).astype(np.float32)
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.mse_loss(model(torch.from_numpy(noisy)),clean[rows])
                if not torch.isfinite(loss):raise ValueError('Nonfinite denoising loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite autoencoder gradient')
                optimizer.step();total+=float(loss.detach())*len(rows)
            for group in optimizer.param_groups:group['lr']*=controls['decay']
            model.eval()
            with torch.no_grad():error=float(torch.nn.functional.mse_loss(model(clean),clean))
            if not np.isfinite(error):raise ValueError('Nonfinite fitted autoencoder')
            history.append(error);noisy_history.append(total/len(x))
    return FittedDAE(model,x.shape[1],tuple(layers),tuple(history),tuple(noisy_history))
