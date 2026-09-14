from pathlib import Path
import numpy as np
import pytest
from sciona import otto_sofia as module
from sciona.otto_sofia_crossfit import crossfit_sofia

LIBRARY=str(Path(__file__).resolve().parents[1]/'.venv/otto-r-library')
CONTROLS=dict(seed=12,regularization=.01,iterations=10000,ddof=1,r_library=LIBRARY)


def inputs():
    y=np.tile(np.arange(9),30)
    return [np.eye(9)[y],y,np.arange(270)%5,[f'r{i}' for i in range(270)],np.eye(9),[f'q{i}' for i in range(9)]]


def test_native_learning_and_query_batch_independence():
    x,y,_,_,q,_=inputs()
    scores=module.fit_predict(x,y,q,**CONTROLS)
    np.testing.assert_array_equal(scores.argmax(axis=1),np.arange(9))
    np.testing.assert_array_equal(scores[:1],module.fit_predict(x,y,q[:1],**CONTROLS))


def test_five_fold_scaler_population_and_learning(monkeypatch):
    args=inputs();original=module.fit_scaling;seen=[]
    def capture(x,**kwargs):
        seen.append(x.copy());return original(x,**kwargs)
    monkeypatch.setattr(module,'fit_scaling',capture)
    result=crossfit_sofia(*args,**CONTROLS)
    assert len(seen)==6 and result['binary_models']==54
    for fold in range(5):np.testing.assert_array_equal(seen[fold],args[0][args[2]!=fold])
    np.testing.assert_array_equal(seen[-1],args[0])
    np.testing.assert_array_equal(result['oof'].argmax(axis=1),args[1])


def test_heldout_labels_and_query_isolation():
    args=inputs();baseline=crossfit_sofia(*args,**CONTROLS);held=args[2]==0
    args[1][held]=(args[1][held]+1)%9
    changed=crossfit_sofia(*args,**CONTROLS)
    np.testing.assert_array_equal(baseline['oof'][held],changed['oof'][held])
    args=inputs();args[4]*=100
    np.testing.assert_array_equal(baseline['oof'],crossfit_sofia(*args,**CONTROLS)['oof'])


def test_runtime_failure_removes_private_transport(monkeypatch):
    x,y,_,_,q,_=inputs();seen=[]
    def failure(command,**kwargs):
        seen.append(Path(command[4]).parent)
        return type('Result',(),dict(returncode=1))()
    monkeypatch.setattr(module.subprocess,'run',failure)
    with pytest.raises(ValueError,match='execution failed'):module.fit_predict(x,y,q,**CONTROLS)
    assert seen and all(not p.exists() for p in seen)


def test_overlap_rejected_before_runtime(monkeypatch):
    args=inputs();args[5][0]=args[3][0]
    def forbidden(*args,**kwargs):raise AssertionError('Runtime started')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='overlapping'):crossfit_sofia(*args,**CONTROLS)
