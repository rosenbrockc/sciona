from dataclasses import replace
import copy
import json
import pytest
from sciona import santa_execution as execution
from tests.test_santa_boosting import replay


def payload():
    return dict(version=1,training=[replay('synthetic-train',1)],validation=[replay('synthetic-valid',2)],
                query=dict(episode_id='synthetic-query',history=[dict(own_action=0,opponent_action=1,reward=1)]),
                controls=dict(seed=42,decision_seed=12,max_rows=10000,num_boost_round=24,stopping_rounds=4,num_leaves=15))


def test_full_boundary_repeat_and_json_output():
    prepared=execution.prepare(payload())
    result=execution.execute(prepared)
    assert result==execution.execute(prepared)
    assert result==json.loads(json.dumps(result,allow_nan=False))
    assert result['models']==2 and result['query_step']==1 and len(result['scores'])==100
    assert 0<=result['action']<100 and result['training_rows']>0 and result['validation_rows']>0
    assert 'synthetic-train' not in json.dumps(result)+repr(prepared)


def test_detached_prepared_configuration():
    p=payload();prepared=execution.prepare(p);before=prepared.configuration
    p['training'][0]['thresholds'][0]=99
    assert prepared.configuration==before


def test_corrupt_prepared_rejected_before_fit(monkeypatch):
    prepared=execution.prepare(payload())
    def forbidden(*args,**kwargs):raise AssertionError('Fitting started')
    monkeypatch.setattr(execution,'fit_threshold_models',forbidden)
    with pytest.raises(ValueError,match='changed'):
        execution.execute(replace(prepared,configuration=prepared.configuration+' '))


@pytest.mark.parametrize('change',['query_overlap','validation_overlap','query_truth','opponent_reward','bad_control','extra_field','finished_query'])
def test_invalid_boundary_rejected_before_fit(change,monkeypatch):
    p=payload()
    if change=='query_overlap':p['query']['episode_id']='synthetic-train'
    if change=='validation_overlap':p['validation'][0]['episode_id']='synthetic-train'
    if change=='query_truth':p['query']['thresholds']=[50]*100
    if change=='opponent_reward':p['query']['history'][0]['opponent_reward']=1
    if change=='bad_control':p['controls']['decision_seed']=True
    if change=='extra_field':p['extra']=1
    if change=='finished_query':p['query']['history']*=2000
    def forbidden(*args,**kwargs):raise AssertionError('Fitting started')
    monkeypatch.setattr(execution,'fit_threshold_models',forbidden)
    with pytest.raises(ValueError):execution.prepare(p)


def test_query_does_not_affect_model_selection():
    p=payload();a=execution.execute(execution.prepare(p))
    p['query']['history']=[dict(own_action=8,opponent_action=9,reward=0)]*3
    b=execution.execute(execution.prepare(p))
    assert a['model_evaluation']==b['model_evaluation']
    assert a['query_step']==1 and b['query_step']==3
