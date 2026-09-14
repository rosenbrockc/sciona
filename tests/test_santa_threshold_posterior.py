import math
import numpy as np
import pytest
from sciona.santa_threshold_posterior import infer_threshold, reward_probabilities


def test_uniform_initial_belief():
    result = infer_threshold([])
    assert result['expected_threshold'] == pytest.approx(50.)
    expected = sum(1.02**i for i in range(101)) / 101
    assert result['expected_transformed_threshold'] == pytest.approx(expected)
    assert result['inverse_transformed_expectation'] == pytest.approx(math.log(expected, 1.02))
    assert 58 < result['inverse_transformed_expectation'] < 59


def test_integer_draw_probability_exhaustive():
    for pulls in [0, 1, 2, 11, 100, 4000]:
        decay = .97**pulls
        oracle = [sum(draw < threshold * decay for draw in range(101)) / 101 for threshold in range(101)]
        np.testing.assert_array_equal(reward_probabilities(pulls), oracle)


@pytest.mark.parametrize('reward', [0, 1])
def test_single_observation_hand_bayes(reward):
    probabilities = np.arange(101) / 101.
    weights = probabilities if reward else 1. - probabilities
    weights /= weights.sum()
    result = infer_threshold([dict(prior_pulls=0, reward=reward)])
    np.testing.assert_allclose(result['probabilities'], weights, atol=1e-16)
    assert result['expected_threshold'] == pytest.approx(sum(i*w for i,w in enumerate(weights)))


def test_opponent_pulls_change_likelihood_without_opponent_rewards():
    observations = [dict(prior_pulls=0, reward=1), dict(prior_pulls=4, reward=0)]
    weights = reward_probabilities(0) * (1-reward_probabilities(4))
    weights /= weights.sum()
    result = infer_threshold(observations)
    np.testing.assert_allclose(result['probabilities'], weights)
    assert result != infer_threshold([dict(prior_pulls=0, reward=1),dict(prior_pulls=1, reward=0)])


def test_long_history_stays_normalized():
    observations = [dict(prior_pulls=2*i, reward=1) for i in range(2000)]
    result = infer_threshold(observations)
    assert np.isfinite(result['probabilities']).all()
    assert sum(result['probabilities']) == pytest.approx(1.)
    assert result['probabilities'][0] == 0


@pytest.mark.parametrize('observations', [None, [dict(prior_pulls=True,reward=0)],
    [dict(prior_pulls=0,reward=True)], [dict(prior_pulls=-1,reward=0)],
    [dict(prior_pulls=4001,reward=0)], [dict(prior_pulls=0,reward=0,opponent_reward=1)],
    [dict(prior_pulls=2,reward=0),dict(prior_pulls=2,reward=1)]])
def test_invalid_observations(observations):
    with pytest.raises(ValueError):
        infer_threshold(observations)
