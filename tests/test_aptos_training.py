"""Independent Smooth-L1 update oracle and optimizer coverage contracts."""
import numpy as np
import pytest
import torch
from torch import nn

from sciona.aptos_training import train_epochs


def model():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1)).double()


def test_manual_piecewise_loss_and_sgd_gradient():
    net = model()
    with torch.no_grad():
        net[-1].weight.copy_(torch.tensor([[.2, -.1, .3]]))
        net[-1].bias.fill_(.1)
    x = torch.tensor([[.5, 1., -.5], [1., -.5, .25]], dtype=torch.float64).reshape(2, 3, 1, 1)
    y = torch.tensor([[.3], [3.]], dtype=torch.float64)
    w, b = net[-1].weight.detach().numpy().copy(), net[-1].bias.detach().numpy().copy()
    inputs = x.numpy().reshape(2, 3)
    delta = inputs @ w.T + b - y.numpy()
    expected_loss = np.where(np.abs(delta) < 1., .5 * delta ** 2, np.abs(delta) - .5).mean()
    derivative = np.clip(delta, -1., 1.) / len(delta)
    optimizer = torch.optim.SGD(net.parameters(), lr=.05)
    losses = train_epochs(net, optimizer, lambda epoch: [(x, y)], epochs=1)
    np.testing.assert_allclose(losses, [expected_loss], rtol=0, atol=1e-14)
    np.testing.assert_allclose(net[-1].weight.detach().numpy(), w - .05 * derivative.T @ inputs, rtol=0, atol=1e-14)
    np.testing.assert_allclose(net[-1].bias.detach().numpy(), b - .05 * derivative.sum(0), rtol=0, atol=1e-14)


def test_missing_optimizer_parameter_rejected():
    net = model()
    optimizer = torch.optim.SGD([net[-1].weight], lr=.1)
    with pytest.raises(ValueError, match='every trainable'):
        train_epochs(net, optimizer, lambda epoch: [], epochs=1)


def test_empty_epoch_rejected():
    net = model()
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    with pytest.raises(ValueError, match='observations'):
        train_epochs(net, optimizer, lambda epoch: [], epochs=1)


def test_vector_targets_rejected_before_broadcast():
    net = model()
    optimizer = torch.optim.SGD(net.parameters(), lr=.1)
    with pytest.raises(ValueError, match='soft targets'):
        train_epochs(net, optimizer, lambda epoch: [(torch.ones(2, 3, 1, 1), torch.ones(2))], epochs=1)
