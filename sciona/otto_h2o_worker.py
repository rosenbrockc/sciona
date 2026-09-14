"""Private subprocess entry point; caller owns and cleans this process group."""
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import h2o
from h2o.backend import H2OLocalServer
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


def _start_server(root):
    # H2O 3.46 starts Java with setsid. In this dedicated single-threaded
    # worker, keep every startup child in the group owned by the parent.
    original=subprocess.Popen
    def inherited_group(*args,**kwargs):
        preexec=kwargs.get('preexec_fn')
        if preexec is not None and preexec is not os.setsid:
            raise ValueError('Unexpected native launch hook')
        kwargs['preexec_fn']=None
        return original(*args,**kwargs)
    subprocess.Popen=inherited_group
    try:
        return H2OLocalServer.start(nthreads=1,max_mem_size=1024**3,ice_root=str(root),log_dir=str(root),log_level='ERRR',bind_to_localhost=True,verbose=False)
    finally:subprocess.Popen=original


def main(root):
    with np.load(root/'input.npz',allow_pickle=False) as data:
        x=data['reference'];y=data['labels'];q=data['query']
    options=json.loads((root/'controls.json').read_text())
    with _start_server(root) as server:
        process=server._process
        if os.getpgid(process.pid)!=os.getpgrp():raise ValueError('Native server escaped owned worker group')
        h2o.connect(server=server,verbose=False);h2o.no_progress()
        try:
            names=[f'x{i}' for i in range(x.shape[1])]
            train=h2o.H2OFrame(np.column_stack((x,y)),column_names=names+['target'])
            train['target']=train['target'].asfactor()
            if train['target'].levels()[0]!=[str(i) for i in range(9)]:raise ValueError('Unexpected class domain')
            query=h2o.H2OFrame(q,column_names=names);predictions=[]
            for run in range(10):
                model=H2ODeepLearningEstimator(**options['controls'],seed=(options['seed']+run)%2**31,reproducible=True,stopping_rounds=0,train_samples_per_iteration=-1,adaptive_rate=True)
                model.train(x=names,y='target',training_frame=train)
                prediction=model.predict(query)
                rows=prediction.as_data_frame(use_pandas=False)
                if rows[0]!=['predict']+[f'p{i}' for i in range(9)]:raise ValueError('Unexpected prediction column order')
                predictions.append(np.asarray(rows[1:],dtype=float)[:,1:])
                h2o.remove(prediction);h2o.remove(model)
        finally:h2o.connection().close()
    process.wait(timeout=10)
    if process.poll() is None:raise ValueError('Owned server failed to terminate')
    np.save(root/'scores.npy',np.stack(predictions),allow_pickle=False)


if __name__=='__main__':main(Path(sys.argv[1]))
