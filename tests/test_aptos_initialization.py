"""Synthetic checkpoints exercise initialization failures without downloads."""
import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sciona.aptos_initialization import build_from_checkpoint
from sciona.aptos_models import FAMILIES
from sciona.aptos_pooling import GeM


def fixture_checkpoint(tmp_path, monkeypatch, family):
    factory, pool, width, _ = FAMILIES[family]
    classes = 1001 if family.startswith('inception_') else 1000

    class Tiny(nn.Module):
        def __init__(self, count):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            setattr(self, pool, nn.AdaptiveAvgPool2d(1))
            self.last_linear = nn.Linear(width, count)

    def construct(*, num_classes, pretrained):
        assert num_classes == classes and pretrained is None
        return Tiny(num_classes)

    monkeypatch.setitem(sys.modules, 'pretrainedmodels', SimpleNamespace(**{factory: construct}))
    original = Tiny(classes)
    with torch.no_grad():
        original.backbone.weight.fill_(0.375)
    path = tmp_path / 'synthetic.pt'
    torch.save(original.state_dict(), path)
    return path, original, pool


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('checkpoint_format', ['torch', 'safetensors'])
def test_exact_backbone_loaded_before_head_replacement(tmp_path, monkeypatch, family, checkpoint_format):
    path, original, pool = fixture_checkpoint(tmp_path, monkeypatch, family)
    if checkpoint_format == 'safetensors':
        from safetensors.torch import save_file
        save_file(original.state_dict(), path)
    model = build_from_checkpoint(family, path, expected_sha256=digest(path),
                                  checkpoint_format=checkpoint_format)
    torch.testing.assert_close(model.backbone.weight, original.backbone.weight, rtol=0, atol=0)
    assert isinstance(getattr(model, pool), GeM)
    assert model.last_linear.out_features == 1
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('failure', ['hash', 'missing', 'extra', 'shape', 'nonfinite', 'wrapped'])
def test_rejects_bad_checkpoint(tmp_path, monkeypatch, failure):
    path, original, _ = fixture_checkpoint(tmp_path, monkeypatch, 'seresnext50')
    state = original.state_dict()
    if failure == 'missing':
        del state['backbone.weight']
    elif failure == 'extra':
        state['unexpected'] = torch.zeros(1)
    elif failure == 'shape':
        state['last_linear.weight'] = torch.zeros(1, 2048)
    elif failure == 'nonfinite':
        state['backbone.weight'][0, 0] = float('nan')
    elif failure == 'wrapped':
        state = {'state_dict': state}
    torch.save(state, path)
    expected = '0' * 64 if failure == 'hash' else digest(path)
    with pytest.raises((ValueError, RuntimeError)):
        build_from_checkpoint('seresnext50', path, expected_sha256=expected)


@pytest.mark.parametrize('digest_value', [None, 'abc', 'A' * 64])
def test_invalid_digest_before_file_access(digest_value):
    with pytest.raises(ValueError, match='SHA256'):
        build_from_checkpoint('seresnext50', '/nonexistent', expected_sha256=digest_value)


def test_unknown_format_before_file_access():
    with pytest.raises(ValueError, match='format'):
        build_from_checkpoint('seresnext50', '/nonexistent', expected_sha256='0' * 64,
                              checkpoint_format='automatic')
