from pathlib import Path
import numpy as np
import pytest
from sciona import otto_libfm as module

EXECUTABLE=str(Path(__file__).resolve().parents[1]/'.venv/otto-libfm/bin/libFM')


def fixture():
    y=np.tile(np.arange(9),30)
    return np.column_stack((y,y%3)),y,np.column_stack((np.arange(9),np.arange(9)%3))


def fit(x,y,q):
    return module.fit_predict(x,y,q,executable=EXECUTABLE,seed=12,factors=4,iterations=80,learn_rate=.05,regularization=[0.,0.,.01])


def test_reference_only_categorical_encoding():
    x,q=module.encode([[0,2],[1,2],[0,3]],[[0,2],[9,9]])
    np.testing.assert_array_equal(x.toarray(),[[1,0,1,0],[0,1,1,0],[1,0,0,1]])
    np.testing.assert_array_equal(q.toarray(),[[1,0,1,0],[0,0,0,0]])
    assert x.nnz==6


def test_native_nine_binary_classifiers_learn():
    x,y,q=fixture();scores=fit(x,y,q)
    assert scores.shape==(9,9)
    np.testing.assert_array_equal(scores.argmax(axis=1),np.arange(9))
    assert (scores.max(axis=1)>.8).all()


def test_query_population_does_not_change_native_fit():
    x,y,q=fixture();baseline=fit(x,y,q)
    changed=fit(x,y,np.concatenate((q,[[999,999]])))
    np.testing.assert_array_equal(baseline,changed[:9])
    np.testing.assert_array_equal(baseline[:1],fit(x,y,q[:1]))


def test_temporary_transport_cleanup(monkeypatch):
    x,y,q=fixture();seen=[]
    def fail(command,**kwargs):
        seen.append(Path(command[6]).parent)
        return type('Result',(),dict(returncode=1))()
    monkeypatch.setattr(module.subprocess,'run',fail)
    with pytest.raises(ValueError,match='fit failed'):fit(x,y,q)
    assert seen and all(not directory.exists() for directory in seen)


def test_missing_class_rejected_before_native_runtime(monkeypatch):
    x,y,q=fixture();y[y==8]=7
    def fail(*args,**kwargs):raise AssertionError('Runtime started')
    monkeypatch.setattr(module.subprocess,'run',fail)
    with pytest.raises(ValueError,match='nine-class'):fit(x,y,q)


def test_missing_output_cannot_reuse_previous_class_scores(monkeypatch):
    x,y,q=fixture();calls=[]
    def incomplete(command,**kwargs):
        calls.append(command)
        if len(calls)==1:np.savetxt(command[10],np.full(len(q),.5))
        return type('Result',(),dict(returncode=0))()
    monkeypatch.setattr(module.subprocess,'run',incomplete)
    with pytest.raises(ValueError,match='unavailable'):fit(x,y,q)
    assert len(calls)==2
