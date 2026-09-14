"""Synthetic numerical checks independent of FFT implementation details."""
import math
import numpy as np
import pytest
import torch
from sciona.physical_operator_network import SpectralConv1d,FourierOperator1d

@pytest.mark.parametrize('length',[7,8])
def test_spectral_matches_explicit_dft(length):
    torch.manual_seed(12)
    layer=SpectralConv1d(2,3,3).double()
    x=torch.randn(2,2,length,dtype=torch.float64)
    actual=layer(x).detach().numpy()
    weight=torch.view_as_complex(layer.weight).detach().numpy()
    expected=np.zeros((2,3,length))
    for b in range(2):
        for o in range(3):
            for t in range(length):
                value=0j
                for k in range(3):
                    coefficient=sum(sum(float(x[b,i,j])*np.exp(-2j*np.pi*k*j/length) for j in range(length))*weight[i,o,k] for i in range(2))
                    contribution=coefficient*np.exp(2j*np.pi*k*t/length)
                    value += contribution.real if k==0 else 2*contribution.real
                expected[b,o,t]=value.real/length
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

def test_full_mode_identity_and_nyquist():
    layer=SpectralConv1d(1,1,5).double()
    with torch.no_grad():layer.weight[...,0].fill_(1);layer.weight[...,1].zero_()
    x=torch.tensor([[[(-1.)**i for i in range(8)]]],dtype=torch.float64)
    torch.testing.assert_close(layer(x),x)

def test_high_frequency_removed():
    layer=SpectralConv1d(1,1,2).double()
    with torch.no_grad():layer.weight[...,0].fill_(1);layer.weight[...,1].zero_()
    x=torch.cos(2*math.pi*3*torch.arange(16,dtype=torch.float64)/16)[None,None,:]
    torch.testing.assert_close(layer(x),torch.zeros_like(x),atol=1e-14,rtol=0)

def test_spectral_gradients():
    torch.manual_seed(1)
    layer=SpectralConv1d(1,1,2).double()
    x=torch.randn(1,1,5,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(layer,(x,))
    layer(x).square().mean().backward()
    assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum()>0

def test_operator_translation_and_batch_independence():
    torch.manual_seed(2)
    net=FourierOperator1d(width=4,modes=3,depth=2).double()
    x=torch.randn(3,12,dtype=torch.float64);t=torch.tensor([0.,.1,.2],dtype=torch.float64)
    y=net(x,t)
    torch.testing.assert_close(net(x.roll(2,-1),t),y.roll(2,-1),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(net(x[:1],t[:1]),y[:1],atol=1e-12,rtol=1e-12)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())

@pytest.mark.parametrize('modes',[0,True,1.5])
def test_invalid_modes(modes):
    with pytest.raises(ValueError):SpectralConv1d(1,1,modes)

def test_too_many_modes():
    with pytest.raises(ValueError):SpectralConv1d(1,1,5)(torch.ones(1,1,4))

def test_invalid_time():
    with pytest.raises(ValueError):FourierOperator1d()(torch.ones(2,16),torch.tensor([0.,-1.]))
