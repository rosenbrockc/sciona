import pytest
import torch

from sciona.contrails_evaluation import evaluate


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x, x


def batch(size, prediction):
    x = torch.full((size, 1, 1, 1), prediction)
    y = torch.ones_like(x)
    return dict(x=x, y=y, y_sym=y, label=y)


def test_hard_dice_is_global_across_unequal_batches():
    model = IdentityModel().train()
    result = evaluate(model, [batch(1, 100.), batch(3, -100.)])
    assert result['score'] == .4  # one TP, one predicted positive, four true positives
    assert model.training


def test_original_loss_keeps_source_validation_weight():
    result = evaluate(IdentityModel(), [batch(1, 0.)])
    expected = torch.log(torch.tensor(2.)).item() * 1.05
    assert result['loss'] == pytest.approx(expected)


def test_error_restores_model_mode():
    model = IdentityModel().train()
    with pytest.raises(ValueError):
        evaluate(model, [])
    assert model.training
