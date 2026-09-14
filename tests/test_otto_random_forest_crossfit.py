import numpy as np
import pytest
from sciona import otto_random_forest_crossfit as module
from pathlib import Path
LIBRARY=str(Path(__file__).resolve().parents[1]/'.venv/otto-r-library')

def fixture():
    y=np.repeat(np.arange(9),20)
    return np.eye(9)[y]*20,y,np.eye(9)*20


def inputs():
    x,y,q=fixture()
    return [x,y,np.tile(np.arange(5),36),[f'r{i}' for i in range(len(x))],q,[f'q{i}' for i in range(len(q))]]


def run(args):
    return module.crossfit_forest(*args,seed=12,ntree=32,mtry=3,nodesize=1,r_library=LIBRARY)


def test_native_oof_and_full_reference_fit(monkeypatch):
    args=inputs(); original=module.fit_predict; counts=[]
    def capture(x,y,q,**kwargs):
        counts.append((len(x),len(q)))
        return original(x,y,q,**kwargs)
    monkeypatch.setattr(module,'fit_predict',capture)
    result=run(args)
    assert counts==[(144,36)]*5+[(180,9)]
    assert result['forest_models']==6
    np.testing.assert_array_equal(result['oof'].argmax(axis=1),args[1])
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))


def test_heldout_labels_and_query_isolation():
    args=inputs(); baseline=run(args); held=args[2]==0
    args[1]=args[1].copy(); args[1][held]=(args[1][held]+1)%9
    changed=run(args)
    np.testing.assert_array_equal(baseline['oof'][held],changed['oof'][held])
    args=inputs(); args[4]=args[4]*3
    np.testing.assert_array_equal(baseline['oof'],run(args)['oof'])


def test_overlap_rejected_before_fitting(monkeypatch):
    args=inputs();args[5][0]=args[3][0]
    def forbidden(*args,**kwargs):raise AssertionError('Fitting started')
    monkeypatch.setattr(module,'fit_predict',forbidden)
    with pytest.raises(ValueError,match='overlapping'):run(args)
