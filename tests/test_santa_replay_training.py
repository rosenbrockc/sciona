import copy
import numpy as np
import pytest
from sciona.santa_replay_training import prepare_replay_training


def replay(identity, rounds=3):
    return dict(episode_id=identity, thresholds=list(range(1,101)),
                actions=[[0, 1] for _ in range(rounds)], rewards=[[1, 0] for _ in range(rounds)], perspectives=[0,1])


def run(training=None, validation=None, **kwargs):
    return prepare_replay_training(training or [replay('synthetic-a')], validation or [replay('synthetic-b')], seed=42, max_rows=kwargs.get('max_rows',10000))


def test_shapes_and_transformed_targets():
    result = run()
    for population in result.values():
        n = population['rows']
        assert 0 < n < 600
        assert population['normal'].shape == population['transformed'].shape == (n, 10)
        np.testing.assert_array_equal(population['transformed_target'], 1.02**population['raw_target'])
        columns = [i for i in range(10) if i != 6]
        np.testing.assert_array_equal(population['normal'][:,columns],population['transformed'][:,columns])


def test_hidden_thresholds_only_change_labels():
    training = [replay('synthetic-a')]
    first = run(training)
    training[0]['thresholds'] = [100] * 100
    second = run(training)
    for key in ['normal', 'transformed']:
        np.testing.assert_array_equal(first['training'][key], second['training'][key])
    assert not np.array_equal(first['training']['raw_target'], second['training']['raw_target'])


def test_current_and_future_rewards_not_in_prior_features():
    training = [replay('synthetic-a')]
    first = run(training)['training']
    training[0]['actions'][1:] = [[8,9],[7,6]]
    training[0]['rewards'][1:] = [[0,1],[0,1]]
    second = run(training)['training']
    before = first['normal'][:,0] <= 1
    np.testing.assert_array_equal(first['normal'][before], second['normal'][before])
    assert not np.array_equal(first['normal'],second['normal'])


def test_other_player_reward_isolation_for_selected_perspective():
    training = [replay('synthetic-a')]
    training[0]['perspectives'] = [0]
    first = run(training)['training']
    training[0]['rewards'] = [[1,1] for _ in range(3)]
    second = run(training)['training']
    np.testing.assert_array_equal(first['normal'], second['normal'])
    np.testing.assert_array_equal(first['transformed'], second['transformed'])


def test_validation_changes_do_not_change_training():
    first = run()['training']
    v = replay('synthetic-c',rounds=7)
    v['thresholds'] = [50]*100
    second = run(validation=[v])['training']
    for key in first:np.testing.assert_array_equal(first[key],second[key])


def test_reproducible_without_input_mutation():
    training = [replay('synthetic-a')];before = copy.deepcopy(training)
    a,b = run(training),run(training)
    assert training == before
    for population in a:
        for key in a[population]:np.testing.assert_array_equal(a[population][key],b[population][key])


@pytest.mark.parametrize('change', ['same_game','duplicate_train','bad_threshold','bad_reward','bad_player','row_cap'])
def test_invalid_replays_and_split_rejected(change):
    t,v = [replay('synthetic-a')],[replay('synthetic-b')]
    if change == 'same_game':v[0]['episode_id']='synthetic-a'
    if change == 'duplicate_train':t.append(copy.deepcopy(t[0]))
    if change == 'bad_threshold':t[0]['thresholds'][0]=True
    if change == 'bad_reward':v[0]['rewards'][0][0]=2
    if change == 'bad_player':t[0]['perspectives']=[0,0]
    with pytest.raises(ValueError):run(t,v,max_rows=1 if change=='row_cap' else 10000)
