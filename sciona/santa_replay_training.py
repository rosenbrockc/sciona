"""Causal replay feature generation with whole-game validation separation.

Hidden initial thresholds enter labels only. Real replays, identities and paths
are runtime inputs and must never be included in publication evidence.
"""
import numpy as np
from sciona.santa_agent_features import BanditHistory


def validate_replay(replay):
    fields = {'episode_id', 'thresholds', 'actions', 'rewards', 'perspectives'}
    if type(replay) is not dict or set(replay) != fields:
        raise ValueError('Invalid replay fields')
    if type(replay['episode_id']) is not str or not replay['episode_id']:
        raise ValueError('Nonempty opaque episode identity required')
    thresholds = replay['thresholds']
    if type(thresholds) is not list or len(thresholds) != 100 or any(type(t) is not int or not 0 <= t <= 100 for t in thresholds):
        raise ValueError('One hundred integer initial thresholds required')
    actions, rewards = replay['actions'], replay['rewards']
    if type(actions) is not list or type(rewards) is not list or not 1 <= len(actions) <= 2000 or len(actions) != len(rewards):
        raise ValueError('Aligned bounded nonempty rounds required')
    for rounds, upper in ((actions, 99), (rewards, 1)):
        if any(type(row) is not list or len(row) != 2 or any(type(v) is not int or not 0 <= v <= upper for v in row) for row in rounds):
            raise ValueError('Invalid two-player round values')
    perspectives = replay['perspectives']
    if type(perspectives) is not list or not perspectives or any(type(p) is not int or p not in (0,1) for p in perspectives) or len(set(perspectives)) != len(perspectives):
        raise ValueError('Distinct eligible player perspectives required')


def _population(replays, *, rng, max_rows):
    normal, transformed, labels = [], [], []
    count = 0
    for replay in replays:
        histories = {p: BanditHistory() for p in replay['perspectives']}
        targets = np.asarray(replay['thresholds'], dtype=np.float64)
        for actions, rewards in zip(replay['actions'], replay['rewards']):
            # Sample and emit BEFORE seeing this round's actions or rewards.
            for player, history in histories.items():
                selected = rng.integers(0, 5, size=100) == 0
                n = int(selected.sum())
                if count + n > max_rows:
                    raise ValueError('Replay training rows exceed configured limit')
                if n:
                    a, b = history.features()
                    normal.append(a[selected]); transformed.append(b[selected]); labels.append(targets[selected])
                    count += n
            for player, history in histories.items():
                history.observe(actions[player], actions[1-player], rewards[player])
    if not count:
        raise ValueError('No sampled replay rows')
    raw = np.concatenate(labels)
    return dict(normal=np.concatenate(normal), transformed=np.concatenate(transformed),
                raw_target=raw, transformed_target=np.power(1.02, raw), rows=count)


def prepare_replay_training(training, validation, *, seed, max_rows):
    if type(seed) is not int or not 0 <= seed < 2**32 or type(max_rows) is not int or max_rows < 1:
        raise ValueError('Explicit seed and positive row limit required')
    if any(type(population) is not list or not 1 <= len(population) <= 1000 for population in (training, validation)):
        raise ValueError('Bounded nonempty replay populations required')
    identities = set()
    # Validate every source and split identity before generating model inputs.
    for replay in training + validation:
        validate_replay(replay)
        if replay['episode_id'] in identities:
            raise ValueError('Episode identities must be unique across both populations')
        identities.add(replay['episode_id'])
    streams = np.random.SeedSequence(seed).spawn(2)
    return dict(training=_population(training, rng=np.random.default_rng(streams[0]), max_rows=max_rows),
                validation=_population(validation, rng=np.random.default_rng(streams[1]), max_rows=max_rows))
