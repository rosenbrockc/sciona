import copy
import numpy as np
import pytest
from sciona.future_sales_features import build_panel

ENTITIES=[dict(store=0,item=0,category=0),dict(store=1,item=0,category=0),dict(store=0,item=1,category=1)]
CONTROLS=dict(lags=[1,2],rolling_windows=[2],calendar_cycle=12,cold_start=2.)


def make(rows,bounds=None):return build_panel(rows,ENTITIES,bounds or [0,10,20,30,40],**CONTROLS)
def row(day,store,item,q):return dict(day=day,store=store,item=item,quantity=q)


def test_aggregation_precedes_clipping_and_missing_zero():
    p=make([row(1,0,0,30),row(2,0,0,-15),row(3,1,0,-2),row(10,0,0,25)])
    assert np.array_equal(p.targets[:6],[15,0,0,20,0,0])
    assert np.isnan(p.targets[-3:]).all()
    assert p.features[3,p.feature_names.index('lag_1')]==15
    assert p.features[6,p.feature_names.index('rolling_2')]==17.5


def test_expanding_group_means_are_strictly_prior():
    p=make([row(1,0,0,6),row(2,1,0,12),row(3,0,1,3),row(11,0,0,20)])
    vector=p.features[3]
    for key,value in [('entity_expanding',6),('store_expanding',4.5),('item_expanding',9),('category_expanding',9),('global_expanding',7)]:
        assert vector[p.feature_names.index(key)]==value
    assert p.features[0,p.feature_names.index('global_expanding')]==2


def test_future_target_mutation_cannot_change_current_features():
    rows=[row(1,0,0,6),row(11,0,0,10),row(21,1,0,5)]
    original=make(rows);changed=make(rows+[row(11,0,0,1000),row(21,0,0,1000)])
    assert np.array_equal(original.features[:6],changed.features[:6],equal_nan=True)
    prefix=make(rows[:2],[0,10,20,30])
    assert np.array_equal(original.features[:9],prefix.features,equal_nan=True)
    shuffled=make(rows[::-1]);assert np.array_equal(original.features,shuffled.features,equal_nan=True)


@pytest.mark.parametrize('case',['future','unknown','nonfinite','duplicate','category','boundaries','zero_lag'])
def test_invalid_boundaries(case):
    entities=copy.deepcopy(ENTITIES);bounds=[0,10,20,30];controls=copy.deepcopy(CONTROLS);rows=[]
    if case=='future':rows=[row(20,0,0,1)]
    elif case=='unknown':rows=[row(1,9,9,1)]
    elif case=='nonfinite':rows=[row(1,0,0,float('nan'))]
    elif case=='duplicate':entities.append(entities[0].copy())
    elif case=='category':entities[1]['category']=9
    elif case=='boundaries':bounds=[0,10,10,30]
    else:controls['lags']=[0]
    with pytest.raises(ValueError):build_panel(rows,entities,bounds,**controls)
