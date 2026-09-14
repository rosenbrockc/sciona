import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.ensemble_partitions import (
    select_ensemble_population, collect_population_scores, ordered_prediction_ids,
)


def inputs():
    # Clip values are irrelevant to selection; actual numerical providers own
    # orientation/rate/duration checks. Distinct objects expose accidental mixing.
    train = [[object(), object()] for _ in range(3)]
    prediction = [[object()] * n for n in [1, 3, 2]]
    labels = [np.array([0, 1]) for _ in range(3)]
    ids = [np.array(values) for values in [[8], [1, 7, 3], [6, 2]]]
    return train, prediction, labels, ids


def test_population_selection_preserves_objects_labels_and_identity_order():
    args = inputs()
    selected = select_ensemble_population(*args, 1)
    assert selected[0] is args[0][1] and selected[1] is args[1][1]
    np.testing.assert_array_equal(selected[2], [0, 1])
    np.testing.assert_array_equal(selected[3], [1, 7, 3])


def test_collect_and_andriy_identity_adapter_preserve_unequal_population_counts():
    _, clips, _, ids = inputs()
    scores = [np.array([.8]), np.array([.1, .7, .3]), np.array([.6, .2])]
    merged, identity = collect_population_scores(scores[0], ids[0], scores[1], ids[1], scores[2], ids[2])
    np.testing.assert_array_equal(merged, [.8, .1, .7, .3, .6, .2])
    np.testing.assert_array_equal(identity, [8, 1, 7, 3, 6, 2])
    np.testing.assert_array_equal(ordered_prediction_ids(clips, ids), identity)


@pytest.mark.parametrize('failure', ['population_count', 'index', 'labels', 'identity_count', 'cross_duplicate', 'overflow'])
def test_population_selection_rejects_inconsistent_family_contract(failure):
    train, prediction, labels, ids = inputs()
    index = 0
    if failure == 'population_count': prediction.pop()
    elif failure == 'index': index = True
    elif failure == 'labels': labels[2] = np.array([1, 1])
    elif failure == 'identity_count': ids[2] = np.array([6])
    elif failure == 'cross_duplicate': ids[2] = np.array([8, 2])
    else: ids[0] = np.array([2**63], dtype=np.uint64)
    with pytest.raises(ValueError):
        select_ensemble_population(train, prediction, labels, ids, index)


def test_andriy_identity_binding_rejects_wrong_population_split_with_same_total():
    _, prediction, _, ids = inputs()
    ids[0], ids[2] = ids[2], ids[0]
    with pytest.raises(ValueError):
        ordered_prediction_ids(prediction, ids)


def test_population_collection_rejects_duplicate_identity_across_populations():
    with pytest.raises(ValueError, match='unique across populations'):
        collect_population_scores(np.array([.1]), np.array([1]), np.array([.2]), np.array([2]), np.array([.3]), np.array([1]))
