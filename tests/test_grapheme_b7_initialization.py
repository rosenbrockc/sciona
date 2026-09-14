import pytest

from sciona.grapheme_b7_initialization import load_standard_b7


def test_changed_artifact_is_rejected_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic-corrupt-checkpoint.bin'
    path.write_bytes(b'synthetic invalid checkpoint payload')

    def forbidden(*args, **kwargs):
        raise AssertionError('unverified content reached deserialization')

    monkeypatch.setattr('sciona.grapheme_b7_initialization.torch.load', forbidden)
    with pytest.raises(ValueError, match='content differs'):
        load_standard_b7(path)
