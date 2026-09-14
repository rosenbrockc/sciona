import torch

from sciona.dsb_network import DetectorNet, CaseNet
from sciona.dsb_losses import detector_loss, classifier_loss


def test_detector_coordinates_influence_output_and_loss_backpropagates():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = DetectorNet().eval()
    inputs = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
    coordinates = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
    output = model(inputs, coordinates)
    assert output.shape == (2, 4, 4, 4, 3, 5)
    labels = torch.zeros_like(output)
    labels[..., 0] = -1
    labels[:, 1, 1, 1, 0, 0] = 1
    detector_loss(output, labels)['total'].backward()
    assert coordinates.grad.abs().sum() > 0
    assert inputs.grad.abs().sum() > 0


def test_case_network_learned_baseline_and_classifier_loss_backpropagate():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = CaseNet(topk=2).eval()
    inputs = torch.randn(1, 2, 1, 16, 16, 16, requires_grad=True)
    coordinates = torch.randn(1, 2, 3, 4, 4, 4, requires_grad=True)
    detector, case, proposals = model(inputs, coordinates)
    assert detector.shape == (1, 2, 4 * 4 * 4 * 3 * 5)
    assert case.shape == (1,) and proposals.shape == (1, 2)
    assert model.baseline.item() == -30.
    classifier_loss(case, proposals, torch.ones(1), torch.ones_like(proposals))['total'].backward()
    assert model.baseline.grad.abs().sum() > 0
    assert model.fc2.weight.grad.abs().sum() > 0
    assert model.NoduleNet.preBlock[0].weight.grad.abs().sum() > 0
