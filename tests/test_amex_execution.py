"""Synthetic explicit raw mappings and detached private-boundary checks."""
import json
import numpy as np
import pytest
from sciona.amex_raw import preprocess
from sciona.amex_execution import prepare,execute,Prepared


def population(n):
    return dict(numerics=[[[.019],[-.019],[None]] for _ in range(n)],categories=[[[1.,'a'],[None,'b'],[2.,None]] for _ in range(n)],times=[[1,2,3] for _ in range(n)],months=[[0,0,1] for _ in range(n)])


def payload():
    return dict(version=1,training=dict(sequences=population(400),labels=[0]*200+[1]*200,identities=[f'r{i}' for i in range(400)]),query=dict(sequences=population(2),identities=['q0','q1']),controls=dict(category_rules=[None,dict(values={'a':0,'b':1},missing=-1)],zero_fill_columns=[0],seed=42))


def test_raw_scaling_mapping_and_nulls():
    r=preprocess(population(1),payload()['controls']['category_rules'])
    np.testing.assert_array_equal(r['numerics'][0].ravel(),[1.,-2.,np.nan])
    np.testing.assert_array_equal(r['categories'][0],[[100,0],[np.nan,1],[200,-1]])


def test_canonical_detached_boundary():
    p=payload();v=prepare(p);p['controls']['seed']=43
    assert json.loads(v.configuration)['controls']['seed']==42
    assert prepare(json.loads(v.configuration))==v
    with pytest.raises(ValueError):execute(Prepared(v.configuration+' ',v.fingerprint))


@pytest.mark.parametrize('case',['overlap','query_label','unknown','bool_label','fill','seed','infinity','missing','small'])
def test_rejections(case):
    p=payload()
    if case=='overlap':p['query']['identities'][0]='r0'
    if case=='query_label':p['query']['labels']=[0,1]
    if case=='unknown':p['query']['sequences']['categories'][0][0][1]='unknown'
    if case=='bool_label':p['training']['labels'][0]=False
    if case=='fill':p['controls']['zero_fill_columns']=[1]
    if case=='seed':p['controls']['seed']=True
    if case=='infinity':p['query']['sequences']['numerics'][0][0][0]=float('inf')
    if case=='missing':p['controls']['category_rules'][1]['missing']=None
    if case=='small':p['training']['labels']=[0]*400
    with pytest.raises(ValueError):prepare(p)
