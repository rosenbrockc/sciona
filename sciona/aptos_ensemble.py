"""Identity-aligned eight-model regression blend for the APTOS reconstruction.

The winning writeup specifies the architecture pairs, arithmetic mean and cut
points. Historical floating-point precision and threshold equality are not
specified. This reference uses float64 reduction and requires an explicit tie
policy; neither choice is claimed as historical implementation parity.
"""
import numpy as np

MODEL_KEYS = tuple((family, replica) for family in
    ('inception_resnet_v2', 'inception_v4', 'seresnext50', 'seresnext101') for replica in (0, 1))
CUT_POINTS = (0.7, 1.5, 2.5, 3.5)


def _keys(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError('Nonempty unique runtime identities required')
    return tuple(values)


def blend(expected_keys, predictions):
    """Average exactly two outputs from each of the four specified families.

    Each model maps to (runtime identities, floating regression vector). All
    models must cover the same requested identities. Scores are not clipped
    to the ordinal label range, and private identities are not returned/logged.
    """
    expected = _keys(expected_keys)
    if not isinstance(predictions, dict) or set(predictions) != set(MODEL_KEYS):
        raise ValueError('Exactly the eight specified model outputs required')
    aligned = []
    for model in MODEL_KEYS:
        entry = predictions[model]
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError('Each model must supply identities and regression values')
        keys, values = entry
        keys = _keys(keys)
        array = np.asarray(values)
        if (set(keys) != set(expected) or array.shape != (len(keys),)
                or array.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or not np.isfinite(array).all()):
            raise ValueError('Aligned finite floating regression vectors required')
        positions = {key: i for i, key in enumerate(keys)}
        aligned.append(array[[positions[key] for key in expected]].astype(np.float64))
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.stack(aligned).mean(axis=0)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite ensemble arithmetic')
    return result


def classify(scores, *, tie_policy):
    """Map finite scores to ordinals 0..4 with an explicit boundary policy.

    'upper' places a score exactly at a cut point in the higher class; 'lower'
    places it in the lower class. The writeup does not resolve this convention.
    """
    values = np.asarray(scores)
    if (tie_policy not in ('upper', 'lower') or values.ndim != 1 or not values.size
            or values.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(values).all()):
        raise ValueError('Finite floating score vector and explicit upper/lower tie policy required')
    return np.searchsorted(CUT_POINTS, values, side='right' if tie_policy == 'upper' else 'left')
