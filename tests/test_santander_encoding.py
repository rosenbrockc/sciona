"""Hand-calculated synthetic evidence for source-compared encoding."""
import numpy as np
import pytest
from sciona.santander_encoding import encode_populations


def example():
    return np.array([[1.],[1.],[2.],[3.],[4.],[4.],[5.]]), np.array([1,1,0,1,0,1,0]), np.array([[2.],[6.],[6.],[7.]])


def test_hand_encoding_self_exclusion_and_mean_override():
    data = example()
    before = [x.copy() for x in data]
    result = encode_populations(*data)
    np.testing.assert_array_equal(result['training_categories'][:,0], [2,2,0,4,2,1,4])
    np.testing.assert_array_equal(result['test_categories'][:,0], [1,4,4,4])
    np.testing.assert_array_equal(result['real_test_mask'], [True,False,False,True])
    np.testing.assert_allclose(result['training_substituted'][:,0], [1,1,2,20/7,4,4,20/7])
    np.testing.assert_allclose(result['test_substituted'][:,0], [2,20/7,20/7,20/7])
    for old, value in zip(before, data): np.testing.assert_array_equal(old, value)


def test_both_class_presence_and_own_label_exclusion():
    train = np.ones((3,1)); labels = np.array([0,0,1]); query = np.ones((2,1))
    result = encode_populations(train, labels, query)
    np.testing.assert_array_equal(result['training_categories'][:,0], [3,3,1])
    np.testing.assert_array_equal(result['test_categories'][:,0], [3,3])


def test_label_reference_is_global_not_fold_isolated():
    train, labels, query = example()
    original = encode_populations(train, labels, query)
    labels[0] = 0
    changed = encode_populations(train, labels, query)
    assert original['training_categories'][1,0] != changed['training_categories'][1,0]
    # Self-exclusion keeps this row independent of its own label.
    assert original['training_categories'][0,0] == changed['training_categories'][0,0]
    np.testing.assert_array_equal(original['training_substituted'], changed['training_substituted'])


def test_row_permutations_preserve_results():
    train, labels, query = example()
    base = encode_populations(train, labels, query)
    a = np.array([6,2,0,4,1,5,3]); b = np.array([3,1,0,2])
    result = encode_populations(train[a], labels[a], query[b])
    for suffix in ('categories','substituted'):
        np.testing.assert_allclose(result['training_'+suffix], base['training_'+suffix][a])
        np.testing.assert_allclose(result['test_'+suffix], base['test_'+suffix][b])


@pytest.mark.parametrize('case', ['width','empty','integer','nonfinite','labels'])
def test_invalid_population(case):
    train, labels, query = example()
    if case == 'width': query = np.ones((2,2))
    if case == 'empty': train = train[:0]
    if case == 'integer': query = query.astype(int)
    if case == 'nonfinite': train[0,0] = np.nan
    if case == 'labels': labels[0] = 2
    with pytest.raises(ValueError): encode_populations(train, labels, query)
