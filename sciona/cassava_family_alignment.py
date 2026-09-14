"""Identity-safe assembly of the four source-defined family outputs."""
import numpy as np

from sciona.cassava_fold_contract import _keys
from sciona.cassava_ensemble import combine


def assemble(expected_keys, *, vit, resnext, efficientnet, cropnet):
    """Align runtime identities before final score arithmetic.

Each family supplies (keys, normalized five-class probabilities), with its
fold/TTA reduction already complete. Identity values stay private at runtime.
"""
    expected = _keys(expected_keys)
    aligned = []
    for entry in (vit, resnext, efficientnet, cropnet):
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError('Each family must supply identities and probabilities')
        keys, values = entry
        keys = _keys(keys)
        values = np.asarray(values)
        if set(keys) != set(expected) or values.shape != (len(keys), 5):
            raise ValueError('Every family must cover exactly the requested identities')
        positions = {key: i for i, key in enumerate(keys)}
        aligned.append(values[[positions[key] for key in expected]])
    return combine(*aligned)
