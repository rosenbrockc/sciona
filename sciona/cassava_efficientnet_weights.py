"""Private, verified local H5 weight snapshots; no automatic downloads."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import tempfile


@contextmanager
def verified_weights(weights, *, expected_sha256=None):
    """Require a reviewed hash for pretrained bytes; None is diagnostic only.

Integrity verification does not grant redistribution rights. Deserialize a
private snapshot of the verified bytes so a later source-file change cannot
alter the model being loaded.
"""
    if weights is None:
        if expected_sha256 is not None:
            raise ValueError('Untrained initialization cannot ignore a weights digest')
        yield None
        return
    if (not isinstance(expected_sha256, str)
            or not re.fullmatch('[0-9a-f]{64}', expected_sha256)):
        raise ValueError('Pretrained initialization requires a reviewed SHA-256')
    path = Path(weights)
    if path.is_symlink() or not path.is_file():
        raise ValueError('A regular local H5 weights file is required')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Pretrained weights integrity mismatch')
    with tempfile.TemporaryDirectory(prefix='cassava-verified-weights-') as directory:
        snapshot = Path(directory) / 'weights.h5'
        snapshot.write_bytes(content)
        yield str(snapshot)
