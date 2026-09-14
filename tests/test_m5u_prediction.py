from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sciona.m5u_prediction import predict


class Constant:
    feature_name_=['x']
    def __init__(self,value):self.value=value
    def predict(self,frame):return np.full(len(frame),self.value)


def models():
    return [SimpleNamespace(bag=b,group=g,quantile=q,model=Constant(v+g+q))
            for b,v in enumerate([0,0,90]) for g in [1,2,3] for q in [.1,.9]]


def test_forecast_is_fold_weighted_bag_mean_with_inverse_scale():
    frame=pd.DataFrame({'x':[0.,1.]})
    result=predict(models(),frame,[2.,4.],[.1,.9])
    expected=np.array([30+(1+4+9)/6+q for q in [.1,.9]])[:,None]*[2,4]
    np.testing.assert_allclose(result,expected,rtol=1e-14)


def test_validation_selects_latest_outer_fold_only():
    result=predict(models(),pd.DataFrame({'x':[0.]}),[2.],[.1,.9],validation=True)
    np.testing.assert_allclose(result[:,0],[(30+3+.1)*2,(30+3+.9)*2],rtol=1e-14)


def test_missing_model_and_feature_reordering_reject():
    with pytest.raises(ValueError,match='inventory'):predict(models()[:-1],pd.DataFrame({'x':[0.]}),[1.],[.1,.9])
    with pytest.raises(ValueError,match='feature order'):predict(models(),pd.DataFrame({'z':[0.]}),[1.],[.1,.9])
