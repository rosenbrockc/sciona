import numpy as np
import pytest
from sciona.geotemporal_features import feature_matrix


def compute(points,context,history):
    return feature_matrix(points,context,history,radius=5.,lookback=10.,period=20.,neighbors=2)


def test_hand_latest_spatial_join_and_strict_past():
    points=[dict(x=0.,y=0.,time=10.)]
    context=[dict(x=3.,y=4.,time=1.,value=999.),dict(x=3.,y=4.,time=8.,value=6.),dict(x=0.,y=0.,time=10.,value=2.),dict(x=0.,y=0.,time=11.,value=999.)]
    history=[dict(x=0.,y=0.,time=0.,target=2.),dict(x=0.,y=0.,time=9.,target=4.),dict(x=0.,y=0.,time=10.,target=999.)]
    result=compute(points,context,history)[0]
    np.testing.assert_allclose(result,[0,0,0,-1,4,2.5,1,1,3,1,2],atol=1e-12)


def test_future_changes_cannot_change_features():
    points=[dict(x=0.,y=0.,time=10.)]
    context=[dict(x=0.,y=0.,time=9.,value=2.),dict(x=0.,y=0.,time=11.,value=3.)]
    history=[dict(x=0.,y=0.,time=9.,target=2.),dict(x=0.,y=0.,time=10.,target=4.)]
    first=compute(points,context,history)
    context[-1]['value']=999.;history[-1]['target']=999.
    np.testing.assert_array_equal(first,compute(points,context,history))


def test_missing_flags_radius_and_expiration():
    points=[dict(x=0.,y=0.,time=10.)]
    context=[dict(x=6.,y=0.,time=10.,value=2.),dict(x=0.,y=0.,time=-1.,value=3.)]
    history=[dict(x=6.,y=0.,time=9.,target=2.)]
    np.testing.assert_array_equal(compute(points,context,history)[0,4:],np.zeros(7))


def test_order_and_query_population_independence():
    points=[dict(x=0.,y=0.,time=10.)]
    context=[dict(x=-1.,y=0.,time=9.,value=2.),dict(x=1.,y=0.,time=9.,value=4.)]
    first=compute(points,context,[])
    np.testing.assert_array_equal(first,compute(points,context[::-1],[]))
    np.testing.assert_array_equal(first,compute(points+[dict(x=9.,y=9.,time=12.)],context,[])[:1])


def test_duplicate_site_time_rejected():
    p=dict(x=0.,y=0.,time=1.)
    with pytest.raises(ValueError):compute([p],[dict(p,value=2.),dict(p,value=3.)],[])


def test_boolean_coordinate_rejected():
    with pytest.raises(ValueError):compute([dict(x=True,y=0.,time=1.)],[dict(x=0.,y=0.,time=0.,value=1.)],[])
