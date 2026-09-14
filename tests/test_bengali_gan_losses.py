import pytest
import torch

from sciona.bengali_gan_losses import discriminator_objective, generator_objective, least_squares


def test_generator_known_objective_and_gradient_weights():
    # Constant synthetic arrays yield an analytic result independent of PyTorch losses.
    a = torch.full((1,3,2,2), .25, requires_grad=True)
    b = torch.full((1,3,2,2), -.5, requires_grad=True)
    pa = torch.tensor([.5, 1.5], requires_grad=True)
    pb = torch.tensor([0., 2.], requires_grad=True)
    font = torch.tensor(2., requires_grad=True)
    terms = generator_objective(fake_a_prediction=pa, fake_b_prediction=pb,
        reconstructed_a=a, real_a=torch.zeros_like(a), reconstructed_b=b,
        real_b=torch.zeros_like(b), weighted_font_loss=font)
    assert terms['total'].item() == 10.75
    terms['total'].backward()
    torch.testing.assert_close(a.grad, torch.full_like(a, 10/12))
    torch.testing.assert_close(b.grad, torch.full_like(b, -10/12))
    torch.testing.assert_close(pa.grad, torch.tensor([-.5,.5]))
    torch.testing.assert_close(pb.grad, torch.tensor([-1.,1.]))
    assert font.grad.item() == 1


def test_discriminator_half_weight_per_domain_and_no_sigmoid():
    values = [torch.tensor([x], requires_grad=True) for x in [2., 3., -1., -2.]]
    terms = discriminator_objective(real_a_prediction=values[0], replay_a_prediction=values[1],
        real_b_prediction=values[2], replay_b_prediction=values[3])
    assert terms['domain_a'].item() == 5
    assert terms['domain_b'].item() == 4
    terms['total'].backward()
    assert [v.grad.item() for v in values] == [1.,3.,-2.,-2.]


def test_invalid_predictions_rejected():
    for value in [torch.tensor(float('nan')), torch.empty(0), torch.ones(1, dtype=torch.float64)]:
        with pytest.raises(ValueError):
            least_squares(value, real=True)
    with pytest.raises(ValueError):
        least_squares(torch.ones(1), real=1)


def test_cycle_shape_broadcast_rejected():
    with pytest.raises(ValueError, match='matching'):
        generator_objective(fake_a_prediction=torch.ones(1), fake_b_prediction=torch.ones(1),
            reconstructed_a=torch.ones(2,3,2,2), real_a=torch.ones(1,3,2,2),
            reconstructed_b=torch.ones(1,3,2,2), real_b=torch.ones(1,3,2,2),
            weighted_font_loss=torch.tensor(1.))
