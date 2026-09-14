"""Independent ten-feature state and prediction rule for the Santa winner method."""
import math
import numpy as np
from sciona.santa_threshold_posterior import infer_threshold


class BanditHistory:
    """Only own rewards and both public actions enter agent-visible state."""
    def __init__(self):
        self.step = 0
        self.own = np.zeros(100, dtype=np.int64)
        self.opponent = np.zeros(100, dtype=np.int64)
        self.modified = np.zeros(100, dtype=np.float32)
        self.last_opponent = np.full(100, -1, dtype=np.int64)
        self.previous_opponent = None
        self.consecutive = 0
        self.histories = [[] for _ in range(100)]
        self.normal = np.full(100, 50., dtype=np.float32)
        self.transformed = np.full(100, np.mean(1.02 ** np.arange(101.)), dtype=np.float32)

    def observe(self, own_action, opponent_action, reward):
        if self.step >= 2000:
            raise ValueError('Episode history limit exceeded')
        if any(type(a) is not int or not 0 <= a < 100 for a in (own_action, opponent_action)):
            raise ValueError('Actions must be integers in [0,99]')
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError('Binary integer own reward required')
        # Both rewards use the same pre-round decay, including shared actions.
        pulls = self.own + self.opponent
        history = self.histories[own_action] + [dict(prior_pulls=int(pulls[own_action]), reward=reward)]
        posterior = infer_threshold(history)
        self.histories[own_action] = history
        self.normal[own_action] = posterior['expected_threshold']
        self.transformed[own_action] = posterior['expected_transformed_threshold']
        self.modified[opponent_action] += 1. / (.97 ** int(pulls[opponent_action]))
        self.own[own_action] += 1
        self.opponent[opponent_action] += 1
        self.last_opponent[opponent_action] = self.step
        self.consecutive += int(self.previous_opponent == opponent_action)
        self.previous_opponent = opponent_action
        self.step += 1

    def features(self):
        # This is the source's normalized sorted-count statistic, not the
        # conventional Gini coefficient. Balanced selection reaches one.
        sorted_counts = np.sort(self.opponent)
        numerator = int(sorted_counts @ np.arange(99, -1, -1))
        cycles, remainder = divmod(self.step, 100)
        denominator = cycles * 4950 + remainder * (remainder - 1) // 2
        dispersion = numerator / denominator if denominator else 0.
        normal = np.empty((100, 10), dtype=np.float32)
        normal[:, 0] = self.step
        normal[:, 1] = self.consecutive
        normal[:, 2] = np.count_nonzero(self.opponent)
        normal[:, 3] = self.opponent.max()
        normal[:, 4] = dispersion
        normal[:, 5] = self.own
        normal[:, 6] = self.normal
        normal[:, 7] = self.opponent
        normal[:, 8] = self.modified
        normal[:, 9] = self.step - self.last_opponent
        transformed = normal.copy()
        transformed[:, 6] = self.transformed
        return normal, transformed


def exploration_ratio(step):
    if type(step) is not int or not 0 <= step <= 2000:
        raise ValueError('Step must be an integer in [0,2000]')
    end = 1. / (1. + math.exp(10.))
    current = 1. / (1. + math.exp(10. * step / 2000.))
    return 1.2 * (current - end) / (.5 - end)


def choose_action(normal_predictions, transformed_predictions, history, *, seed):
    if type(history) is not BanditHistory or history.step >= 2000:
        raise ValueError('Active bandit history required')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Explicit unsigned seed required')
    normal = np.asarray(normal_predictions, dtype=np.float64)
    transformed = np.asarray(transformed_predictions, dtype=np.float64)
    if normal.shape != (100,) or transformed.shape != (100,) or not np.isfinite(normal).all() or not np.isfinite(transformed).all() or (transformed <= 0).any():
        raise ValueError('Finite model predictions and positive transformed predictions required')
    ratio = exploration_ratio(history.step)
    initial = ratio * (np.log(transformed) / math.log(1.02)) + (1. - ratio) * normal
    scores = initial * np.power(.97, history.own + history.opponent)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite blended scores')
    order = np.random.default_rng(seed).permutation(100)
    # The source keeps action zero if every predicted score is nonpositive.
    action = int(order[np.argmax(scores[order])]) if scores.max() > 0 else 0
    return dict(action=action, scores=scores.tolist(), exploration_ratio=ratio)
