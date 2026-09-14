import copy
import numpy as np
import pytest
from sciona.santa_agent_features import BanditHistory, choose_action
from sciona.santa_boosting import fit_threshold_models
from sciona.santa_replay_training import prepare_replay_training


def replay(identity, seed, rounds=80, constant=None):
    rng=np.random.default_rng(seed)
    thresholds=rng.integers(1,101,size=100) if constant is None else np.full(100,constant)
    counts=np.zeros(100,dtype=int);actions=[];rewards=[]
    for _ in range(rounds):
        pair=rng.integers(0,100,size=2).tolist()
        outcomes=[int(rng.integers(0,101)<thresholds[a]*.97**counts[a]) for a in pair]
        actions.append(pair);rewards.append(outcomes)
        for a in pair:counts[a]+=1
    return dict(episode_id=identity,thresholds=thresholds.tolist(),actions=actions,rewards=rewards,perspectives=[0,1])


def fit(t=None,v=None):
    return fit_threshold_models(t or [replay('synthetic-a',1)],v or [replay('synthetic-b',2)],
                                seed=42,max_rows=10000,num_boost_round=24,stopping_rounds=4,num_leaves=15)


def test_two_models_selected_iteration_and_independent_metric():
    t,v=[replay('synthetic-a',1)],[replay('synthetic-b',2)]
    fitted=fit(t,v)
    population=prepare_replay_training(t,v,seed=42,max_rows=10000)['validation']
    for name,target in [('normal','raw_target'),('transformed','transformed_target')]:
        model=getattr(fitted,name);e=fitted.evaluation[name]
        prediction=model.predict(population[name],num_iteration=model.best_iteration,num_threads=1)
        expected=float(np.sqrt(np.mean((prediction-population[target])**2)))
        assert e['heldout_rmse']==pytest.approx(expected)
        assert 1<=e['best_iteration']<=24
        assert e['best_iteration']==int(np.argmin(e['iteration_rmse']))+1
        assert model.num_feature()==10
        assert e['heldout_rmse']==pytest.approx(min(e['iteration_rmse']),rel=1e-6)


def test_early_stopping_actually_triggers():
    fitted=fit([replay('synthetic-a',1,constant=80)],[replay('synthetic-b',2,constant=20)])
    for evidence in fitted.evaluation.values():
        assert evidence['best_iteration']==1
        assert len(evidence['iteration_rmse'])==5


def test_repeat_and_agent_blend_without_state_mutation():
    first,second=fit(),fit()
    h=BanditHistory();h.observe(3,4,1)
    before=copy.deepcopy(h.__dict__)
    a,b=h.features()
    expected=choose_action(first.normal.predict(a,num_threads=1),first.transformed.predict(b,num_threads=1),h,seed=17)
    result=first.predict(h,seed=17)
    assert result==expected==second.predict(h,seed=17)
    assert h.step==before['step'] and h.histories==before['histories']
    np.testing.assert_array_equal(h.own,before['own'])


def test_overlap_fails_before_lightgbm(monkeypatch):
    import sciona.santa_boosting as module
    def forbidden(*args,**kwargs):raise AssertionError('Training started')
    monkeypatch.setattr(module.lgb,'train',forbidden)
    with pytest.raises(ValueError,match='identities'):
        fit([replay('synthetic-a',1)],[replay('synthetic-a',2)])


@pytest.mark.parametrize('controls',[dict(num_leaves=True),dict(stopping_rounds=0),dict(num_boost_round=-1)])
def test_invalid_controls(controls):
    with pytest.raises(ValueError):
        fit_threshold_models([],[],**controls)
