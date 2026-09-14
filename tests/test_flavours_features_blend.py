import numpy as np
from sciona.flavours_features import prepare_features, classifier_views
from sciona.flavours_blend import blend


def test_physical_feature_routing_and_corrected_ratio():
    base = np.arange(92.).reshape(2, 46)
    prepared = prepare_features(base, np.full((2, 3), 5.), np.full((2, 3), 3.),
                                [5., 5.], [2., 2.], [1., 1.], 7)
    np.testing.assert_allclose(prepared['regression'][:, -4:], [[6.5,13.,2.,12.]]*2)
    np.testing.assert_array_equal(prepared['restricted'], base[:, [i for i in range(46) if i != 7]])
    views = classifier_views(prepared, [3.25, 6.5])
    np.testing.assert_allclose(views['corrected'][:, -4:],
                               [[6.5,3.25,3.25,2.], [6.5,6.5,0.,1.]])


def test_blend_endpoints_and_second_group_routing():
    scores = {f'xgb{i}': np.array([0., 1.]) for i in range(1,6)}
    expected = .5 + .2 * .5**.6 + .0001 * .5**.01
    np.testing.assert_allclose(blend(scores, [0.,1.]), [0.,expected])
    # Isolate the neural contribution: only classifier4 and neural are nonzero.
    scores = {f'xgb{i}': np.array([float(i == 4)]) for i in range(1,6)}
    np.testing.assert_allclose(blend(scores, [1.]), [.5*(.85/3.85)**3.9])
