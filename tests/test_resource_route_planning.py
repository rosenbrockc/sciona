"""Synthetic independent route enumeration and constraint checks."""
from dataclasses import replace
import random
import pytest
from sciona.resource_route_planning import plan,encode_state,generate_candidates,search_routes,validate_route


def payload():
    triples=[('direct','s','a',1,3),('detour','s','b',2,1),('join','b','a',0,0),('finish','a','g',1,2)]
    return dict(nodes=['s','a','b','g'],edges=[dict(id=i,source=s,target=t,cost=c,resources=[r]) for i,s,t,c,r in triples],
        start='s',goal='g',budgets=[3],blocked_nodes=[],blocked_edges=[],max_expansions=100)


def test_resource_state_preserves_feasible_more_expensive_prefix():
    result=plan(payload())
    assert result['status']=='optimal' and result['edge_ids']==['detour','join','finish']
    assert result['cost']==3 and result['resources']==[3]


def test_limit_not_misreported_as_infeasible():
    p=payload();p['max_expansions']=1
    assert plan(p)['status']=='search_limit'
    p['max_expansions']=100;p['blocked_edges']=['detour']
    assert plan(p)['status']=='infeasible'
    p['blocked_edges']=[];p['blocked_nodes']=['b']
    assert plan(p)['status']=='infeasible'


def test_zero_cycles_and_identity_route():
    p=payload();p['edges'].append(dict(id='loop',source='s',target='s',cost=0,resources=[0]))
    assert plan(p)['cost']==3
    p['goal']='s';result=plan(p)
    assert result['cost']==0 and result['edge_ids']==[]


def test_second_resource_budget_enforced():
    p=payload();p['budgets']=[3,1]
    for e in p['edges']:e['resources'].append(1)
    assert plan(p)['status']=='infeasible'


def test_forged_candidate_fails_independent_validation():
    result=search_routes(generate_candidates(encode_state(payload())))
    with pytest.raises(ValueError):validate_route(replace(result,cost=0))
    with pytest.raises(ValueError):validate_route(replace(result,path=('finish',)))


def test_random_graphs_against_exhaustive_bounded_walks():
    rng=random.Random(6)
    for case in range(30):
        p=payload();p['nodes']=['s','a','b','g'];p['edges']=[];p['budgets']=[4]
        for n in range(9):
            p['edges'].append(dict(id=str(n),source=rng.choice(p['nodes']),target=rng.choice(p['nodes']),cost=rng.randrange(5),resources=[rng.randint(1,2)]))
        feasible=[]
        def enumerate_walk(node,used,cost):
            if node=='g':feasible.append(cost)
            for e in p['edges']:
                if e['source']==node and used+e['resources'][0]<=4:
                    enumerate_walk(e['target'],used+e['resources'][0],cost+e['cost'])
        enumerate_walk('s',0,0)
        result=plan(p)
        assert result['status']==('optimal' if feasible else 'infeasible')
        if feasible:assert result['cost']==min(feasible)


@pytest.mark.parametrize('case',['negative','bool','dimension','duplicate','unknown','limit'])
def test_invalid_contract(case):
    p=payload()
    if case=='negative':p['edges'][0]['cost']=-1
    if case=='bool':p['budgets']=[True]
    if case=='dimension':p['edges'][0]['resources']=[]
    if case=='duplicate':p['edges'].append(p['edges'][0].copy())
    if case=='unknown':p['blocked_edges']=['missing']
    if case=='limit':p['max_expansions']=0
    with pytest.raises(ValueError):plan(p)
