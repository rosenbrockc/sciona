import numpy as np
import pytest
import torch

from sciona.contrails_fold import train_fold


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(1, 2, 1)

    def forward(self, x):
        y = self.layer(x)
        return y[:, :1], y[:, 1:]


def fixture():
    torch.manual_seed(107)
    model = SyntheticModel()
    x = torch.randn(1, 1, 256, 256)
    label = (x > 0).float()
    batch = dict(x=x, y=label, y_sym=label, label=label, w=torch.ones(1))
    cfg = dict(train=dict(weight_decay=.01, accumulate=1), data=dict(augment_prob=.95),
               val=dict(th=.45, per_epoch=1), test=dict(th=.45),
               scheduler=[{'const': {'lr': .001, 'epoch_end': 2}}])
    return model, batch, cfg


def test_fold_captures_terminal_state_and_keeps_threshold_sweep_diagnostic():
    model, batch, cfg = fixture()
    result = train_fold(model, cfg, [batch] * 4, [batch], [batch], initialization='random')
    assert result['checkpoint_selection'] == 'terminal'
    assert [r['optimizer_steps'] for r in result['epochs']] == [3, 3]
    assert [r['epoch'] for r in result['observations']] == [.75, 1.75]
    assert not model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(result['checkpoint'][key], value, rtol=0, atol=0)
    with torch.no_grad():
        _, logits = model(batch['x'])
    probability = logits.sigmoid().numpy()
    label = batch['label'].numpy()
    expected = []
    for threshold in np.arange(.1, .6, .01):
        hard = probability > threshold
        expected.append(2 * np.sum(hard * label) / (np.sum(hard) + np.sum(label)))
    np.testing.assert_array_equal(result['diagnostic']['scores'], expected)


def test_no_update_fold_is_not_accepted_as_training():
    model, batch, cfg = fixture()
    cfg['train']['accumulate'] = 2
    with pytest.raises(ValueError, match='without an optimizer update'):
        train_fold(model, cfg, [batch], [batch], [batch], initialization='random')
