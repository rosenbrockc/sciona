"""Cross-fitting isolation checks using synthetic data and a mean-model stand-in."""
import numpy as np
import pytest

from sciona.flavours_mass_correction import correct_mass


def test_each_rows_own_target_is_excluded_from_its_oof_prediction(monkeypatch):
    class MeanModel:
        def __init__(self, matrix):
            self.mean = matrix.get_label().mean()

        def predict(self, matrix):
            return np.full(matrix.num_row(), self.mean)

        def num_boosted_rounds(self):
            return 2800

    monkeypatch.setattr('sciona.flavours_mass_correction.xgb.train',
                        lambda params, matrix, num_boost_round: MeanModel(matrix))
    features = np.arange(90, dtype=float).reshape(15, 6)
    target = np.arange(15, dtype=float)
    baseline = correct_mass(features, target, features[:2])
    for row in range(15):
        changed = target.copy()
        changed[row] += 1000
        result = correct_mass(features, changed, features[:2])
        assert result['oof_mass'][row] == baseline['oof_mass'][row]
        assert np.all(result['query_mass'] > baseline['query_mass'])


@pytest.mark.parametrize('case', ['short', 'width', 'target', 'infinite', 'empty_query'])
def test_invalid_boundaries_fail_before_training(case):
    train = np.ones((10, 3))
    target = np.ones(10)
    query = np.ones((2, 3))
    if case == 'short': train = train[:4]
    elif case == 'width': query = query[:, :2]
    elif case == 'target': target[0] = np.nan
    elif case == 'infinite': train[0, 0] = np.inf
    else: query = query[:0]
    with pytest.raises(ValueError):
        correct_mass(train, target, query)
