"""Independent one-dimensional Fourier operator for periodic scalar fields.

Learned low-frequency complex multipliers plus local channel mixing. This is an
explicit project architecture, not a reproduction of the paper's trained models.
"""
import torch
from torch import nn


def positive_int(value):
    if type(value) is not int or value < 1: raise ValueError('Expected positive integer')
    return value


class SpectralConv1d(nn.Module):
    def __init__(self, inputs, outputs, modes):
        super().__init__()
        self.inputs=positive_int(inputs);self.outputs=positive_int(outputs);self.modes=positive_int(modes)
        # Real storage lets ordinary dtype conversion preserve both components.
        self.weight=nn.Parameter(torch.randn(inputs,outputs,modes,2)/(inputs*outputs))

    def forward(self,x):
        if x.ndim != 3 or x.shape[1] != self.inputs or x.shape[-1] < 2 or not x.is_floating_point():
            raise ValueError('Expected batch/channel/periodic-grid floating tensor')
        if self.modes > x.shape[-1]//2+1: raise ValueError('Retained modes exceed real FFT bins')
        if not torch.isfinite(x).all(): raise ValueError('Nonfinite operator input')
        transformed=torch.fft.rfft(x,dim=-1,norm='backward')
        weights=torch.view_as_complex(self.weight.contiguous())
        low=torch.einsum('bik,iok->bok',transformed[:,:,:self.modes],weights)
        zeros=low.new_zeros(x.shape[0],self.outputs,transformed.shape[-1]-self.modes)
        return torch.fft.irfft(torch.cat((low,zeros),dim=-1),n=x.shape[-1],dim=-1,norm='backward')


class FourierOperator1d(nn.Module):
    """Lift state and broadcast nondimensional diffusion time, then spectral blocks.

Uniform periodic coordinates are implicit in sample ordering. No learned absolute
position channels, preserving circular translation equivariance.
"""
    def __init__(self,width=16,modes=6,depth=3):
        super().__init__()
        width=positive_int(width);depth=positive_int(depth)
        self.lift=nn.Conv1d(2,width,1)
        self.spectral=nn.ModuleList([SpectralConv1d(width,width,modes) for _ in range(depth)])
        self.local=nn.ModuleList([nn.Conv1d(width,width,1) for _ in range(depth)])
        self.head=nn.Sequential(nn.Conv1d(width,width,1),nn.GELU(),nn.Conv1d(width,1,1))

    def forward(self,states,diffusion_times):
        if states.ndim != 2 or diffusion_times.shape != (states.shape[0],):
            raise ValueError('Expected batch/grid states and one diffusion time per state')
        if not states.is_floating_point() or not diffusion_times.is_floating_point() or not torch.isfinite(states).all() or not torch.isfinite(diffusion_times).all() or (diffusion_times < 0).any():
            raise ValueError('Expected finite states and nonnegative diffusion times')
        time=diffusion_times[:,None].expand_as(states)
        x=self.lift(torch.stack((states,time),dim=1))
        for spectral,local in zip(self.spectral,self.local): x=torch.nn.functional.gelu(spectral(x)+local(x))
        return self.head(x)[:,0]
