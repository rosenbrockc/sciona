"""Synthetic mapping and transductive-scaling contract checks."""
import numpy as np
from sciona.santander_preparation import prepare_neural_populations


def test_sorted_training_vocabulary_and_unknown_query_category():
    train = np.array([[1.], [1.], [2.], [3.], [4.], [4.], [5.]])
    labels = np.array([1,1,0,1,0,1,0])
    query = np.array([[2.], [6.], [6.], [7.]])
    r = prepare_neural_populations(train, labels, query)
    np.testing.assert_array_equal(r['training_categories'][:,0], [3,3,1,4,3,2,4])
    np.testing.assert_array_equal(r['test_categories'][:,0], [2,4,4,4])
    # Here training codes are only zero; query presence produces an unseen code.
    r = prepare_neural_populations(np.array([[1.],[2.]]), np.array([0,1]), np.array([[1.],[2.]]))
    np.testing.assert_array_equal(r['training_categories'][:,0], [1,1])
    np.testing.assert_array_equal(r['test_categories'][:,0], [0,0])


def test_combined_population_standardization_matches_hand_moments():
    train = np.array([[0.], [2.]])
    query = np.array([[4.], [4.]])
    r = prepare_neural_populations(train, np.array([0,1]), query)
    raw = np.concatenate((r['training_raw'], r['test_raw']))[:,0]
    original = np.array([0., 2., 4., 4.])
    np.testing.assert_allclose(raw, (original-2.5)/np.sqrt(2.75), rtol=1e-6)
    assert not r['real_test_mask'].any()
    # Excluded test rows still participate in continuous scaling.
    assert r['training_raw'][0,0] != -1
    # All substituted values equal training mean, so scaled replacement is zero.
    assert not r['training_substituted'].any() and not r['test_substituted'].any()
    assert r['training_raw'].dtype == np.float32


def test_columns_get_independent_token_maps_and_inputs_are_unchanged():
    train = np.array([[1.,8.],[1.,9.],[2.,10.]])
    query = np.array([[2.,8.],[3.,8.]])
    labels = np.array([0,1,1])
    old = [x.copy() for x in (train, labels, query)]
    r = prepare_neural_populations(train, labels, query)
    np.testing.assert_array_equal(r['training_categories'], [[3,1],[2,2],[1,2]])
    np.testing.assert_array_equal(r['test_categories'], [[3,0],[0,0]])
    for a,b in zip(old,(train,labels,query)): np.testing.assert_array_equal(a,b)
    r['labels'][0] = 1
    assert labels[0] == 0
