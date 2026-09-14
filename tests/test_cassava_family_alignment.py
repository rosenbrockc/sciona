import numpy as np
import pytest

from sciona.cassava_family_alignment import assemble


def test_independent_family_orders_preserve_predictions():
    keys = ['synthetic-a', 'synthetic-b', 'synthetic-c']
    arrays = [np.random.default_rng(i).dirichlet(np.ones(5), size=3) for i in range(4)]
    families = {}
    for name, array, order in zip(('vit', 'resnext', 'efficientnet', 'cropnet'), arrays,
                                  ([2, 0, 1], [1, 2, 0], [0, 2, 1], [2, 1, 0]), strict=True):
        families[name] = ([keys[i] for i in order], array[order])
    result = assemble(keys, **families)
    expected = np.array([[(arrays[0][r, c] + arrays[1][r, c]) / 2
                         + arrays[2][r, c] + arrays[3][r, c] for c in range(5)] for r in range(3)])
    np.testing.assert_array_equal(result['scores'], expected)
    np.testing.assert_array_equal(result['labels'], expected.argmax(-1))
    np.testing.assert_allclose(result['scores'].sum(-1), 3.)


@pytest.mark.parametrize('invalid', [
    (['synthetic-a'], np.full((1, 5), .2)),
    (['synthetic-a', 'synthetic-a'], np.full((2, 5), .2)),
    (['synthetic-a', 'foreign'], np.full((2, 5), .2)),
    (['synthetic-a', 'synthetic-b'], np.full((2, 6), 1/6)),
    (['synthetic-a', 'synthetic-b'], np.full((2, 5), np.nan)),
    (['synthetic-a', 'synthetic-b'], np.full((2, 5), .3)),
])
def test_invalid_family_fails_before_assembly(invalid):
    keys = ['synthetic-a', 'synthetic-b']
    families = {name: (keys, np.full((2, 5), .2)) for name in ('vit', 'resnext', 'efficientnet', 'cropnet')}
    families['efficientnet'] = invalid
    with pytest.raises(ValueError):
        assemble(keys, **families)
