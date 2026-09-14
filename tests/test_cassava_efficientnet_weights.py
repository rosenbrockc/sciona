import hashlib
from pathlib import Path

import pytest

from sciona.cassava_efficientnet_weights import verified_weights
from sciona.cassava_efficientnet_model import build_model


def test_snapshot_uses_verified_bytes_and_cleans_up_on_failure(tmp_path):
    source = tmp_path / 'synthetic.h5'
    source.write_bytes(b'synthetic model bytes')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):
        with verified_weights(source, expected_sha256=digest) as filename:
            snapshot = Path(filename)
            source.write_bytes(b'changed after verification')
            assert snapshot.read_bytes() == b'synthetic model bytes'
            raise RuntimeError('synthetic downstream failure')
    assert not snapshot.exists()


def test_rejects_tampering_and_symlinks(tmp_path):
    source = tmp_path / 'synthetic.h5'
    source.write_bytes(b'synthetic model bytes')
    with pytest.raises(ValueError, match='integrity mismatch'):
        with verified_weights(source, expected_sha256='0' * 64):
            pytest.fail('tampered weights accepted')
    link = tmp_path / 'link.h5'
    link.symlink_to(source)
    with pytest.raises(ValueError, match='regular local'):
        with verified_weights(link, expected_sha256='0' * 64):
            pytest.fail('symlink accepted')


@pytest.mark.parametrize('weights,digest', [('imagenet', None), ('missing', ''), (None, '0' * 64)])
def test_model_rejects_invalid_contract_before_tensorflow_import(weights, digest):
    with pytest.raises(ValueError):
        build_model(weights=weights, expected_sha256=digest)


def test_explicit_untrained_diagnostic():
    with verified_weights(None) as value:
        assert value is None
