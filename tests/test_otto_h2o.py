from pathlib import Path
import numpy as np
import pytest
from sciona import otto_h2o as module

CONTROLS=dict(hidden=[16],epochs=30,activation='Rectifier',standardize=True)


def inputs():
    y=np.tile(np.arange(9),60)
    return np.eye(9)[y]*20,y,np.eye(9)*20


def test_native_ten_run_learning_and_seed_repeatability():
    x,y,q=inputs()
    a=module.fit_predict(x,y,q,seed=12,controls=CONTROLS)
    b=module.fit_predict(x,y,q[::-1],seed=12,controls=CONTROLS)
    np.testing.assert_array_equal(a.argmax(axis=1),np.arange(9))
    np.testing.assert_allclose(a,b[::-1],rtol=0,atol=1e-12)
    np.testing.assert_allclose(a.sum(axis=1),1.,atol=1e-6)


def test_worker_failure_removes_private_transport(monkeypatch):
    original=module.subprocess.Popen;seen=[]
    def fail(command,**kwargs):
        seen.append(Path(command[-1]));return original(['/usr/bin/false'],**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',fail)
    x,y,q=inputs()
    with pytest.raises(ValueError,match='worker failed'):module.fit_predict(x,y,q,seed=12,controls=CONTROLS)
    assert seen and all(not path.exists() for path in seen)


def test_invalid_class_population_rejected_before_worker(monkeypatch):
    x,y,q=inputs();y[y==8]=7
    def fail(*args,**kwargs):raise AssertionError('Worker started')
    monkeypatch.setattr(module.subprocess,'Popen',fail)
    with pytest.raises(ValueError,match='nine-class'):module.fit_predict(x,y,q,seed=12,controls=CONTROLS)


def test_timeout_terminates_owned_worker_and_cleans_transport(monkeypatch):
    original=module.subprocess.Popen;seen=[]
    class TimedWorker:
        def __init__(self,command,**kwargs):
            self.root=Path(command[-1])
            self.process=original([module.sys.executable,'-c','import time; time.sleep(30)'],**kwargs)
            self.pid=self.process.pid;seen.append(self)
        def communicate(self,timeout):return self.process.communicate(timeout=.05)
        def wait(self):return self.process.wait()
    monkeypatch.setattr(module.subprocess,'Popen',TimedWorker)
    x,y,q=inputs()
    with pytest.raises(ValueError,match='timed out'):module.fit_predict(x,y,q,seed=12,controls=CONTROLS)
    assert seen and all(worker.process.poll() is not None and not worker.root.exists() for worker in seen)
