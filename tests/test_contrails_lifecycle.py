"""Orchestration contract only; full-model execution has a separate validator."""
import numpy as np
import pytest
import torch

import sciona.contrails_lifecycle as lifecycle
from sciona.contrails_prediction import VARIANTS


@pytest.mark.parametrize('variant,fold_count', [('v43', 4), ('v47', 10)])
def test_every_terminal_checkpoint_contributes_in_branch_order(monkeypatch, variant, fold_count):
    calls = []

    class Model(torch.nn.Module):
        def __init__(self, cfg, **kwargs):
            super().__init__()
            self.register_buffer('value', torch.tensor(0.))

        def forward(self, x):
            return None, self.value.expand(1, 1, 256, 256)

    def train(model, cfg, train_loader, val_loader, scoring_loader, **kwargs):
        calls.append((train_loader, val_loader))
        return {'checkpoint': {'value': torch.tensor(-.4 + .1 * len(calls))}, 'epochs': []}

    for name in ['TemporalTraining', 'SingleTraining', 'TemporalInference', 'SingleInference']:
        monkeypatch.setattr(lifecycle, name, Model)
    monkeypatch.setattr(lifecycle, 'train_fold', train)
    monkeypatch.setattr(lifecycle, 'make_loader', lambda examples, indices, **kwargs: tuple(indices))
    monkeypatch.setattr(lifecycle, 'prepare_inference', lambda *args, **kwargs: torch.zeros(3, 2, 2))
    result = lifecycle.run_lifecycle([{}] * 20, [{}], [None], initialization='random', variant=variant)
    assert len(calls) == fold_count
    assert all(not set(train) & set(val) for train, val in calls)
    (wt, nt), (ws, ns) = VARIANTS[variant]
    values = 1 / (1 + np.exp(-(-.4 + .1 * np.arange(1, fold_count + 1))))
    expected = wt * values[:nt].mean() + ws * values[nt:].mean()
    np.testing.assert_allclose(result['probabilities'], expected, rtol=1e-6)
    assert [f['branch'] for f in result['folds']] == ['temporal'] * nt + ['single'] * ns
    assert all('checkpoint' not in f['training'] for f in result['folds'])
