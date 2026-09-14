import json
import random

import numpy as np
import pytest
import torch

import sciona.contrails_runtime as runtime
from sciona.contrails_defaults import defaults


def payload():
    thermal = np.zeros((4, 3, 256, 256), np.float32)
    target = np.zeros((1, 256, 256), np.float32)
    def record(key):
        return dict(key=key, thermal=thermal, label=target, annotation_mean=target)
    return dict(version=1, training=[record(f'synthetic-{i}') for i in range(10)],
                scoring=[record('synthetic-score')],
                prediction=[dict(key='synthetic-predict', thermal=thermal)],
                initialization=dict(policy='random'), config=dict(seed=431, variant='v43', epoch_limit=1))


def test_json_roundtrip_seeded_execution_and_rng_restoration(monkeypatch):
    data = payload()
    data = json.loads(json.dumps(data, default=lambda x: x.tolist()))
    def fake(**kwargs):
        assert len(kwargs['training']) == 10
        assert kwargs['training'][0]['thermal'].dtype == np.float32
        value = np.random.random() + random.random() + torch.rand(()).item()
        return dict(probabilities=np.full((1, 1, 1, 1), value), masks=['-'],
                    folds=[], variant=kwargs['variant'], initialization=kwargs['initialization'], epoch_limit=1)
    monkeypatch.setattr(runtime, 'run_lifecycle', fake)
    np.random.seed(999)
    random.seed(999)
    torch.manual_seed(999)
    state = torch.random.get_rng_state().clone()
    a = runtime.execute(data)
    b = runtime.execute(data)
    assert json.dumps(a, allow_nan=False) == json.dumps(b, allow_nan=False)
    assert np.random.random() == np.random.RandomState(999).random()
    assert random.random() == random.Random(999).random()
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize('case', ['version', 'unknown', 'duplicate', 'seed', 'epochs', 'shape'])
def test_invalid_envelope_rejected_before_training(case, monkeypatch):
    data = payload()
    if case == 'version': data['version'] = True
    if case == 'unknown': data['config']['path'] = 'unaccepted'
    if case == 'duplicate': data['prediction'][0]['key'] = data['training'][0]['key']
    if case == 'seed': data['config']['seed'] = -1
    if case == 'epochs': data['config']['epoch_limit'] = 36
    if case == 'shape': data['prediction'][0]['thermal'] = []
    monkeypatch.setattr(runtime, 'run_lifecycle', lambda **kwargs: pytest.fail('Training must not start'))
    with pytest.raises(ValueError):
        runtime.execute(data)


def test_defaults_are_independent_and_packaged_as_python():
    a = defaults()
    a['single']['model']['decoder_channels'].clear()
    assert defaults()['single']['model']['decoder_channels'] == [256, 128, 64, 32, 16]
