"""Orchestration tests with tiny stand-ins; not backbone training evidence."""
import numpy as np
import pytest
import torch
from torch import nn

import sciona.aptos_pipeline as pipeline
from sciona.aptos_ensemble import MODEL_KEYS
from sciona.aptos_models import FAMILIES


def arguments(tmp_path):
    keys = ['synthetic-base', 'synthetic-average', 'synthetic-group', 'synthetic-pseudo']
    return dict(base=([keys[0]], [4]), average=([keys[1]], [0]), grouped=([keys[2]], [2]),
        pseudo_keys=[keys[3]], query_keys=[keys[3]],
        images={key: np.full((3, 5, 3), 40 + i * 30, dtype=np.uint8) for i, key in enumerate(keys)},
        pretrained={family: ('unused', '0' * 64) for family in FAMILIES},
        first_stage_epochs={key: 5 for key in MODEL_KEYS}, seeds={key: 71 + i for i, key in enumerate(MODEL_KEYS)},
        batch_size=2, learning_rate=1e-4, lower_deviation=.5, upper_deviation=.5,
        tie_policy='upper', work_root=tmp_path)


def test_all_first_stage_fits_precede_refinement_and_continuation(tmp_path, monkeypatch):
    def tiny(*args, **kwargs):
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1))
    calls = []
    def record(model, optimizer, batches, *, epochs):
        seen = sum(x.shape[0] for x, _ in batches(0))
        calls.append((epochs, seen))
        return [0.] * epochs
    monkeypatch.setattr(pipeline, 'build_from_checkpoint', tiny)
    monkeypatch.setattr(pipeline, 'build_uninitialized', tiny)
    monkeypatch.setattr(pipeline, 'train_epochs', record)
    result = pipeline.run_reference(**arguments(tmp_path))
    assert calls == [(5, 1)] * 8 + [(10, 4)] * 8
    assert result['checkpoint_replays_exact'] and result['temporary_checkpoints_removed']
    assert not list(tmp_path.iterdir())
    expected = np.mean([result['second_stage_predictions'][key][1] for key in MODEL_KEYS], axis=0)
    np.testing.assert_array_equal(result['scores'], expected)
    assert result['second_stage_targets'][0] == 4.


@pytest.mark.parametrize('fault', ['query_overlap', 'repeated_seed', 'budget', 'missing_model'])
def test_invalid_workflow_boundary_rejects_before_model_construction(tmp_path, monkeypatch, fault):
    args = arguments(tmp_path)
    if fault == 'query_overlap': args['query_keys'] = ['synthetic-base']
    elif fault == 'repeated_seed': args['seeds'][MODEL_KEYS[1]] = args['seeds'][MODEL_KEYS[0]]
    elif fault == 'budget': args['first_stage_epochs'][MODEL_KEYS[0]] = 1
    else: del args['first_stage_epochs'][MODEL_KEYS[0]]
    monkeypatch.setattr(pipeline, 'build_from_checkpoint', lambda *a, **kw: pytest.fail('Constructed model on invalid inputs'))
    with pytest.raises(ValueError):
        pipeline.run_reference(**args)
