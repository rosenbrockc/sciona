from pathlib import Path
import numpy as np
import pytest
from sciona import otto_libfm_crossfit as module
from sciona import otto_libfm as native

EXECUTABLE=str(Path(__file__).resolve().parents[1]/'.venv/otto-libfm/bin/libFM')


def inputs():
    y=np.tile(np.arange(9),30)
    return [np.column_stack((y,y%3)),y,np.arange(270)%5,[f'r{i}' for i in range(270)],np.column_stack((np.arange(9),np.arange(9)%3)),[f'q{i}' for i in range(9)]]


def run(args):
    return module.crossfit_libfm(*args,executable=EXECUTABLE,seed=12,factors=4,iterations=80,learn_rate=.05,regularization=[0.,0.,.01])


def test_native_full_lifecycle_and_fold_local_vocabulary(monkeypatch):
    args=inputs();original=native.encode;references=[];process=native.subprocess.run;calls=[]
    def capture(x,q):
        references.append(x.copy());return original(x,q)
    def count(*args,**kwargs):
        calls.append(1);return process(*args,**kwargs)
    monkeypatch.setattr(native,'encode',capture)
    monkeypatch.setattr(native.subprocess,'run',count)
    result=run(args)
    assert result['binary_models']==len(calls)==54
    assert len(references)==6
    for fold in range(5):np.testing.assert_array_equal(references[fold],args[0][args[2]!=fold])
    np.testing.assert_array_equal(references[-1],args[0])
    np.testing.assert_array_equal(result['oof'].argmax(axis=1),args[1])
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))


def test_heldout_labels_and_query_isolation():
    args=inputs();baseline=run(args);held=args[2]==0
    args[1][held]=(args[1][held]+1)%9
    changed=run(args)
    np.testing.assert_array_equal(baseline['oof'][held],changed['oof'][held])
    args=inputs();args[4]=np.full_like(args[4],999)
    np.testing.assert_array_equal(baseline['oof'],run(args)['oof'])


@pytest.mark.parametrize('problem',['overlap','missing_class','negative','invalid_fold'])
def test_invalid_population_rejected_before_native_fit(problem,monkeypatch):
    args=inputs()
    if problem=='overlap':args[5][0]=args[3][0]
    elif problem=='missing_class':
        args[2][args[1]==8]=0
    elif problem=='negative':args[4][0,0]=-1
    else:args[2][0]=5
    def forbidden(*args,**kwargs):raise AssertionError('Native fit started')
    monkeypatch.setattr(module,'fit_predict',forbidden)
    with pytest.raises(ValueError):run(args)
