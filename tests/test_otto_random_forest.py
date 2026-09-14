from pathlib import Path
import numpy as np
import pytest
from sciona import otto_random_forest as module

LIBRARY=str(Path(__file__).resolve().parents[1]/'.venv/otto-r-library')


def fixture():
    y=np.repeat(np.arange(9),20);x=np.eye(9)[y]*20
    return x,y,np.eye(9)*20


def fit(x,y,q):return module.fit_predict(x,y,q,seed=12,ntree=64,mtry=3,nodesize=1,r_library=LIBRARY)


def test_native_nine_class_vote_probabilities():
    x,y,q=fixture();result=fit(x,y,q)
    assert result.shape==(9,9)
    np.testing.assert_array_equal(result.argmax(axis=1),np.arange(9))
    np.testing.assert_allclose(result*64,np.round(result*64),atol=1e-12)
    np.testing.assert_allclose(result.sum(axis=1),1.)


def test_repeat_and_query_independence():
    x,y,q=fixture();a=fit(x,y,q);b=fit(x,y,q[:1])
    np.testing.assert_array_equal(a[:1],b)


def test_private_files_removed_after_runtime_failure(monkeypatch):
    x,y,q=fixture();directories=[]
    def failure(command,**kwargs):
        directories.append(Path(command[4]).parent)
        return type('Result',(),dict(returncode=1))()
    monkeypatch.setattr(module.subprocess,'run',failure)
    with pytest.raises(ValueError,match='execution failed'):fit(x,y,q)
    assert directories and all(not p.exists() for p in directories)


def test_invalid_labels_rejected_before_process(monkeypatch):
    x,y,q=fixture();y[y==8]=7
    def forbidden(*args,**kwargs):raise AssertionError('Process started')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    with pytest.raises(ValueError):fit(x,y,q)
