import math
import pickle
import numpy as np
import pytest
from sciona.santa_agent_features import BanditHistory, exploration_ratio, choose_action
from sciona.santa_threshold_posterior import infer_threshold


def test_initial_ten_features():
    a, b = BanditHistory().features()
    assert a.shape == b.shape == (100, 10) and a.dtype == np.float32
    np.testing.assert_array_equal(a[:, :6], 0.)
    np.testing.assert_array_equal(a[:, 6], 50.)
    np.testing.assert_array_equal(a[:, 7:9], 0.)
    np.testing.assert_array_equal(a[:, 9], 1.)
    assert b[0, 6] == pytest.approx(np.mean(1.02**np.arange(101.)))


def test_shared_actions_apply_pre_round_likelihood_then_double_decay():
    h = BanditHistory()
    h.observe(4, 4, 1)
    h.observe(4, 4, 0)
    posterior = infer_threshold([dict(prior_pulls=0,reward=1),dict(prior_pulls=2,reward=0)])
    a, b = h.features()
    assert a[4, 5] == a[4, 7] == 2
    assert a[4, 6] == pytest.approx(posterior['expected_threshold'])
    assert b[4, 6] == pytest.approx(posterior['expected_transformed_threshold'])
    assert a[4, 8] == pytest.approx(1 + .97**-2)
    assert a[4, 9] == 1 and a[5, 9] == 3
    assert a[0, 1] == 1 and a[0, 2] == 1 and a[0, 3] == 2


def test_opponent_only_pulls_do_not_update_own_posterior():
    h = BanditHistory()
    h.observe(0, 8, 1)
    a, b = h.features()
    assert a[8, 6] == 50 and a[8, 5] == 0 and a[8, 7] == 1
    assert b[8, 6] == b[9, 6]


def test_sorted_dispersion_matches_pairwise_minimum_oracle():
    h = BanditHistory()
    for i in range(137):
        h.observe(i % 100, (i*i + 3) % 100, i % 2)
    a, _ = h.features()
    pairwise = sum(min(x, y) for i,x in enumerate(h.opponent) for y in h.opponent[i+1:])
    assert a[0, 4] == pytest.approx(pairwise / sum(i % 100 for i in range(137)))
    balanced = BanditHistory()
    for i in range(100): balanced.observe(0, i, 0)
    assert balanced.features()[0][0, 4] == 1


def test_feature_outputs_are_detached():
    h = BanditHistory()
    before = pickle.dumps(h)
    a, b = h.features()
    a[:] = b[:] = -1
    assert pickle.dumps(h) == before


def test_schedule_endpoints_and_monotonicity():
    assert exploration_ratio(0) == pytest.approx(1.2)
    assert exploration_ratio(2000) == pytest.approx(0.)
    ratios = [exploration_ratio(i) for i in range(2001)]
    assert all(a > b for a,b in zip(ratios, ratios[1:]))
    assert exploration_ratio(1000) == pytest.approx(1.2*((1/(1+math.exp(5)))-(1/(1+math.exp(10))))/(.5-(1/(1+math.exp(10)))))


def test_prediction_blend_decay_and_seeded_tie():
    h = BanditHistory()
    h.observe(0, 0, 1)
    normal = np.arange(100, dtype=float)
    transformed = np.power(1.02, normal + 5)
    result = choose_action(normal, transformed, h, seed=42)
    expected = normal + 5*exploration_ratio(1)
    expected[0] *= .97**2
    np.testing.assert_allclose(result['scores'], expected)
    assert result['action'] == 99
    h = BanditHistory()
    first = choose_action(np.ones(100)*50, np.ones(100)*1.02**50, h, seed=12)
    assert first == choose_action(np.ones(100)*50, np.ones(100)*1.02**50, h, seed=12)
    assert first['action'] == int(np.random.default_rng(12).permutation(100)[0])
    assert choose_action(np.zeros(100), np.ones(100)*.5, h, seed=12)['action'] == 0


@pytest.mark.parametrize('actions', [(True,0,1), (100,0,1), (0,-1,1), (0,0,2)])
def test_invalid_update_is_atomic(actions):
    h = BanditHistory();before = pickle.dumps(h)
    with pytest.raises(ValueError):h.observe(*actions)
    assert pickle.dumps(h) == before
