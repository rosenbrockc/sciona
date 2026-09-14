import pytest
import torch

from sciona.dsb_losses import detector_loss
from sciona.dsb_losses import LearnedNoisyOR, classifier_loss


def test_learned_baseline_initialization_and_gradient():
    model = LearnedNoisyOR().double()
    assert model.baseline.item() == -30.
    probabilities = torch.tensor([[.01, .02], [.4, .6]], dtype=torch.float64, requires_grad=True)
    case = model(probabilities)
    losses = classifier_loss(case, probabilities, torch.ones(2, dtype=torch.float64), torch.ones_like(probabilities))
    losses['total'].backward()
    assert model.baseline.grad.item() < 0
    assert bool(torch.isfinite(probabilities.grad).all())


def test_miss_penalty_strict_threshold_epsilon_and_reduction():
    probabilities = torch.tensor([[0., .03], [.02, .01]], dtype=torch.float64, requires_grad=True)
    known = torch.tensor([[1., 1.], [0., 1.]], dtype=torch.float64)
    losses = classifier_loss(torch.full((2,), .5, dtype=torch.float64), probabilities,
                             torch.ones(2, dtype=torch.float64), known)
    expected = -(torch.log(torch.tensor(.001, dtype=torch.float64)) + torch.log(torch.tensor(.011, dtype=torch.float64))) / 4
    torch.testing.assert_close(losses['miss'], expected)
    losses['miss'].backward()
    assert probabilities.grad[0, 1] == probabilities.grad[1, 0] == 0
    torch.testing.assert_close(probabilities.grad[0, 0], torch.tensor(-250., dtype=torch.float64))


def test_classifier_gradcheck_away_from_threshold():
    model = LearnedNoisyOR().double()
    probabilities = torch.tensor([[.01, .2]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: classifier_loss(model(p), p,
        torch.ones(1, dtype=torch.float64), torch.ones_like(p))['total'], (probabilities,))


def sample():
    output = torch.linspace(-.8, 1.1, 60, dtype=torch.float64).reshape(2, 6, 5).requires_grad_()
    labels = torch.zeros_like(output)
    labels[..., 0] = -1
    labels[:, 0, 0] = 1
    labels[:, 1, 0] = 0
    return output, labels


def test_batch_scaled_mining_and_ignored_anchor_gradients():
    output, labels = sample()
    result = detector_loss(output, labels, num_hard=2)
    assert result['negative_count'] == 4
    assert result['positive_count'] == 2
    result['total'].backward()
    assert torch.all(output.grad[:, 1] == 0)
    assert torch.all(output.grad[labels[..., 0] == -1][:, 1:] == 0)
    assert torch.isfinite(output.grad).all()


def test_validation_keeps_all_negatives():
    output, labels = sample()
    assert detector_loss(output, labels, training=False)['negative_count'] == 8
    assert detector_loss(output, labels, num_hard=0)['negative_count'] == 8


def test_no_positive_half_weight_and_zero_regression():
    output = torch.zeros((1, 2, 5), dtype=torch.float64, requires_grad=True)
    labels = torch.zeros_like(output)
    labels[..., 0] = -1
    result = detector_loss(output, labels)
    torch.testing.assert_close(result['total'], torch.tensor(.5, dtype=torch.float64) * torch.log(torch.tensor(2., dtype=torch.float64)))
    assert torch.all(result['regression'] == 0)
    result['total'].backward()
    torch.testing.assert_close(output.grad[..., 0], torch.full((1, 2), .125, dtype=torch.float64))


def test_detector_loss_gradcheck_away_from_mining_ties():
    output, labels = sample()
    assert torch.autograd.gradcheck(lambda x: detector_loss(x, labels)['total'], (output,))


def test_empty_negative_mean_is_rejected():
    output, labels = sample()
    labels[..., 0] = 1
    with pytest.raises(ValueError, match='negative anchor'):
        detector_loss(output, labels)
