"""Synthetic blend identity, arithmetic and boundary-policy checks."""
import math
import numpy as np
import pytest
from sciona.aptos_ensemble import MODEL_KEYS, CUT_POINTS, blend, classify


def predictions():
    return {model: (['synthetic-b', 'synthetic-a'], np.array([index + 2., index - 3.]))
            for index, model in enumerate(MODEL_KEYS)}


def test_identity_alignment_and_independent_scalar_mean():
    inputs = predictions()
    result = blend(['synthetic-a', 'synthetic-b'], inputs)
    expected = [math.fsum(i - 3. for i in range(8)) / 8,
                math.fsum(i + 2. for i in range(8)) / 8]
    np.testing.assert_array_equal(result, expected)
    assert result[1] > 4  # No implicit clipping of regression scores.
    np.testing.assert_array_equal(inputs[MODEL_KEYS[0]][1], [2., -3.])


@pytest.mark.parametrize('fault', ['missing_model', 'foreign_model', 'missing_identity', 'duplicate_identity', 'rank', 'nan'])
def test_incomplete_or_invalid_predictions_reject(fault):
    values = predictions()
    if fault == 'missing_model':
        values.pop(MODEL_KEYS[0])
    elif fault == 'foreign_model':
        values[('other_family', 0)] = values.pop(MODEL_KEYS[0])
    elif fault == 'missing_identity':
        values[MODEL_KEYS[0]] = (['synthetic-a'], np.array([1.]))
    elif fault == 'duplicate_identity':
        values[MODEL_KEYS[0]] = (['synthetic-a'] * 2, np.array([1., 2.]))
    elif fault == 'rank':
        values[MODEL_KEYS[0]] = (['synthetic-a', 'synthetic-b'], np.ones((2, 1)))
    else:
        values[MODEL_KEYS[0]][1][0] = np.nan
    with pytest.raises(ValueError):
        blend(['synthetic-a', 'synthetic-b'], values)


def test_explicit_threshold_equality_and_neighbors():
    cuts = np.array(CUT_POINTS)
    np.testing.assert_array_equal(classify(cuts, tie_policy='upper'), [1, 2, 3, 4])
    np.testing.assert_array_equal(classify(cuts, tie_policy='lower'), [0, 1, 2, 3])
    for policy in ['upper', 'lower']:
        np.testing.assert_array_equal(classify(np.nextafter(cuts, -np.inf), tie_policy=policy), [0, 1, 2, 3])
        np.testing.assert_array_equal(classify(np.nextafter(cuts, np.inf), tie_policy=policy), [1, 2, 3, 4])
    with pytest.raises(ValueError):
        classify(cuts, tie_policy='unspecified')
