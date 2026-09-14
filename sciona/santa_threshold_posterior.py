"""Discrete initial-threshold inference for a decaying two-player bandit.

Independent numerical component. An observation records only this player's
binary reward and the total number of selections by both players BEFORE that
round. Opponent rewards are unavailable. This is not a complete winning agent.
"""
import math
import numpy as np


def reward_probabilities(prior_pulls):
    if type(prior_pulls) is not int or not 0 <= prior_pulls <= 4000:
        raise ValueError('Prior pulls must be an integer in [0,4000]')
    # The environment draws uniformly from integer values 0 through 100 and
    # awards a reward when the draw is strictly below the decayed threshold.
    return np.ceil(np.arange(101, dtype=np.float64) * (0.97 ** prior_pulls)) / 101.


def infer_threshold(observations):
    if type(observations) not in (list, tuple) or len(observations) > 2000:
        raise ValueError('At most 2000 own reward observations required')
    log_weights = np.full(101, -math.log(101.), dtype=np.float64)
    previous = -1
    for observation in observations:
        if type(observation) is not dict or set(observation) != {'prior_pulls', 'reward'}:
            raise ValueError('Invalid own reward observation')
        pulls, reward = observation['prior_pulls'], observation['reward']
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError('Binary integer reward required')
        probability = reward_probabilities(pulls)
        if pulls <= previous:
            raise ValueError('Own observations require increasing prior pull counts')
        previous = pulls
        likelihood = probability if reward else 1. - probability
        with np.errstate(divide='ignore'):
            log_weights += np.log(likelihood)
        maximum = float(log_weights.max())
        if not math.isfinite(maximum):
            raise ValueError('Impossible reward history')
        log_weights -= maximum + math.log(float(np.exp(log_weights - maximum).sum()))
    weights = np.exp(log_weights)
    weights /= weights.sum()
    grid = np.arange(101, dtype=np.float64)
    transformed_mean = float(weights @ np.power(1.02, grid))
    return dict(probabilities=weights.tolist(), expected_threshold=float(weights @ grid),
                expected_transformed_threshold=transformed_mean,
                inverse_transformed_expectation=math.log(transformed_mean) / math.log(1.02))
