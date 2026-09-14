"""Fail closed before model construction for invalid local artifact contracts."""
import pytest

from sciona.cassava_pretrained import load_reference_backbone, build_backbone


@pytest.mark.parametrize('family,digest', [('other', '0' * 64), ('vit', ''), ('resnext', 'A' * 64)])
def test_invalid_reference_contract(tmp_path, family, digest):
    with pytest.raises(ValueError):
        load_reference_backbone(family, tmp_path / 'missing', expected_sha256=digest)


def test_rejects_tampered_bytes_before_deserialization(tmp_path):
    path = tmp_path / 'synthetic.bin'
    path.write_bytes(b'synthetic invalid model')
    with pytest.raises(ValueError, match='integrity mismatch'):
        load_reference_backbone('vit', path, expected_sha256='0' * 64)


def test_missing_file_and_symlink_rejected(tmp_path):
    path = tmp_path / 'missing'
    with pytest.raises(ValueError, match='regular local'):
        load_reference_backbone('vit', path, expected_sha256='0' * 64)
    real = tmp_path / 'synthetic.bin'
    real.write_bytes(b'synthetic invalid model')
    path.symlink_to(real)
    with pytest.raises(ValueError, match='regular local'):
        load_reference_backbone('resnext', path, expected_sha256='0' * 64)


@pytest.mark.parametrize('settings', [
    {'pretrained': True}, {'pretrained': 1},
    {'pretrained': False, 'weights_path': 'unused'},
    {'pretrained': False, 'expected_sha256': '0' * 64},
])
def test_no_implicit_download_or_ignored_artifact(settings):
    for family in ('vit', 'resnext'):
        with pytest.raises(ValueError):
            build_backbone(family, **settings)


def test_classifiers_require_pretrained_evidence():
    from sciona.cassava_vit_model import Classifier as ViT
    from sciona.cassava_resnext import Classifier as ResNeXt
    for classifier in (ViT, ResNeXt):
        with pytest.raises(ValueError, match='local weights'):
            classifier(pretrained=True)


@pytest.mark.parametrize('family', ['vit', 'resnext'])
def test_fold_forwards_reviewed_artifact(tmp_path, monkeypatch, family):
    import importlib
    import cv2
    import numpy as np
    module = importlib.import_module(f'sciona.cassava_{family}_fold')
    ok, encoded = cv2.imencode('.jpg', np.full((600, 800, 3), 71, np.uint8))
    assert ok
    image = encoded.tobytes()
    class ReachedConstructor(Exception):
        pass
    def classifier(**settings):
        assert settings == {'pretrained': True, 'weights_path': tmp_path / 'reviewed',
                            'expected_sha256': '1' * 64}
        raise ReachedConstructor
    monkeypatch.setattr(module, 'Classifier', classifier)
    with pytest.raises(ReachedConstructor):
        module.train_fold([image], [1], [image], [1], pretrained=True,
            weights_path=tmp_path / 'reviewed', expected_sha256='1' * 64,
            batch_size=1, workers=0, output_directory=tmp_path / 'fold')
