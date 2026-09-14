"""Synthetic arithmetic and training checks for source-defined GeM."""
import math
import pytest
import torch

from sciona.aptos_pooling import GeM, gem


@pytest.mark.parametrize('exponent', [1., 2., 3., 4.5])
def test_independent_scalar_power_mean(exponent):
    values = [-2., 0., 0.2, 1., 3., 5.]
    x = torch.tensor(values, dtype=torch.float64).reshape(1, 1, 2, 3)
    expected = (math.fsum(max(v, 1e-6) ** exponent for v in values) / len(values)) ** (1 / exponent)
    result = gem(x, exponent)
    assert result.shape == (1, 1, 1, 1)
    assert float(result) == pytest.approx(expected, rel=1e-14)


def test_input_and_exponent_gradients():
    x = torch.tensor([-.2, .1, .7, 2.], dtype=torch.float64, requires_grad=True).reshape(1, 1, 2, 2)
    p = torch.tensor([3.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: gem(a, b), (x, p), eps=1e-6, atol=1e-6)
    dx, dp = torch.autograd.grad(gem(x, p).sum(), (x, p))
    assert dx[0, 0, 0, 0] == 0 and torch.isfinite(dp).all() and dp.abs().sum() > 0


def test_exponent_is_trained_and_checkpointed():
    module = GeM().double()
    x = torch.arange(1., 25., dtype=torch.float64).reshape(2, 3, 2, 2)
    old = module.p.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=.01)
    module(x).sum().backward()
    optimizer.step()
    assert not torch.equal(module.p, old)
    restored = GeM().double()
    restored.load_state_dict(module.state_dict(), strict=True)
    assert torch.equal(restored(x), module(x))
    assert list(dict(module.named_parameters())) == ['p']
